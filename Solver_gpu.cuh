/*
This file contains the SPH GPU kernels

@author Octavio Navarro
@version 1.0
*/

#ifndef Solver_cuh
#define Solver_cuh

#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "Parameters.cuh"
#include "math_functions.h"

#include <helper_cuda.h>
#include <helper_math.h>

#include <helper_string.h>
#include <drvapi_error_string.h>

#include "Solver.h"

#define PI 3.141592657f

#define SQR(x)					((x) * (x))
#define CUBE(x)					((x) * (x) * (x))
#define POW6(x)					(CUBE(x) * CUBE(x))
#define POW9(x)					(POW6(x) * CUBE(x))

__device__ __constant__ Parameters para;

/// For density calculations
__device__ float Poly6(float3 r, float h)
{
	float distance =  dot(r,r);
	if(distance <= h*h)
		return 315.0f / (64.0f * PI * POW9(h)) * CUBE(h*h - dot(r,r));
	return 0;
}

/// For Pressure calculations
__device__ float3 Spiky(float3 r, float h)
{
	float distance =  length(r);
	if(distance <= h)
		return -15.0f / (PI * POW6(h)) * SQR(h - distance) * r;
	return make_float3(0,0,0);
}

/// For Viscosity calculations
__device__ float Visco(float3 r, float h)
{
	float distance =  length(r);
	if(distance <= h)
		return 45.0f / (PI * POW6(h)) * (h - distance);
	return 0;
}

__device__ float B_spline_2(float3 r, float h)
{
	float q = length(r) / h;
	float B_spline_constant = 1.0f / (PI * CUBE(h));

	if (q >= 0 && q < 1)
		return B_spline_constant * (-3 + 4.5 * q);
	else if (q >= 1 && q < 2)
		return B_spline_constant * (1.5 * (2 - q));
	else
		return 0;
}

__device__ int3 cudaCalcGridPos(float3 pos)
{
	int3 gridPos;
	gridPos.x = floor(pos.x / para.cellSize);
	gridPos.y = floor(pos.y / para.cellSize);
	gridPos.z = floor(pos.z / para.cellSize);

	return gridPos;
}

__device__ int cudaCalcGridHash(int3 gridPos)
{
	/// wrap grid, assumes size is power of 2
	int x = (int)gridPos.x & (int)((para.gridSize.x-1));  
    int y = (int)gridPos.y & (int)((para.gridSize.y-1));
    int z = (int)gridPos.z & (int)((para.gridSize.z-1));

	return x + para.gridSize.x * (y + para.gridSize.y * z);
}

__global__ void cudaCalHash(unsigned int* dindex, unsigned int* dhash, float3* pos, unsigned int num_particles)
{
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x; 
	unsigned int stride = blockDim.x * gridDim.x;

	for(unsigned int tid = idx; tid < num_particles; tid += stride)
	{
		int hash = cudaCalcGridHash(cudaCalcGridPos(pos[tid]));

		dhash[tid] = hash;
		dindex[tid] = tid;
	}
}

__global__ void cudaReorderDataAndFindCellStart(
	unsigned int *cellstart,
	unsigned int *cellend,
	float3* spos,
	float3* svel,
	float3* scorr_vel,
	float* sVm,
	float* sIion,
	float* sStim, 
	unsigned int* dhash,
	unsigned int* dindex,
	float3* pos, 
	float3* vel,
	float3* corr_vel,
	float* Vm,
	float* Iion,
	float* stim,
	unsigned int num_particles)
{
	extern __shared__ int sharedHash[];
	
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;

	int _hash;

	for(unsigned int tid = idx; tid < num_particles; tid += stride)
	{
		_hash = dhash[tid];

		sharedHash[threadIdx.x + 1] = _hash;
		if (tid > 0 && threadIdx.x == 0)
		{
			sharedHash[0] = dhash[tid - 1];
		}

	}
	__syncthreads();

	for(unsigned int tid = idx; tid < num_particles; tid += stride)
	{
		if (tid == 0 || _hash != sharedHash[threadIdx.x])
		{
			cellstart[_hash] = tid;
			if (tid > 0)
			{
				cellend[sharedHash[threadIdx.x]] = tid;
			}
		}
		if (tid == (num_particles - 1))
		{
			cellend[_hash] = tid + 1;
		}

		int sortedIndex = dindex[tid];
		float3 _pos = pos[sortedIndex];
		float3 _corr_vel = corr_vel[sortedIndex];
		float _Vm = Vm[sortedIndex];
		float _stim = stim[sortedIndex];
		float _Iion = Iion[sortedIndex];

		spos[tid] = _pos;
		scorr_vel[tid] = _corr_vel;
		sVm[tid] = _Vm;
		sStim[tid] = _stim;
		sIion[tid] = _Iion;
	}
}

/// Cycles through all the neighbors and adds their mass to the density
__device__ float cudaAddDensity(unsigned int tid, int3 gridPos, float3 pos, unsigned int* cellstart, unsigned int* cellend, float3* spos)
{
	float _dens = 0.0f;
	unsigned int _hash = cudaCalcGridHash(gridPos);
	unsigned int startIndex = cellstart[_hash];

	if (startIndex != 0xffffffff)
	{
		unsigned int endIndex = cellend[_hash];
		for (int j = startIndex; j != endIndex; j++)
		{
			float3 nb_pos = spos[j];
			float3 deltaPos = pos - nb_pos;
			
			_dens += para.mass * Poly6(deltaPos, para.h);
		}
	}

	return _dens;
}

__global__ void cudaCalcDensityPressure(float* dens, float *press, float* Vm, unsigned int* cellstart, unsigned int* cellend, float3* spos, unsigned int num_particles)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;

	for(unsigned int tid = idx; tid < num_particles; tid += stride)
	{
		float3 _pos = spos[tid];

		float _dens = 0.0f;
		int3 gridPos = cudaCalcGridPos(_pos);

		for (int z = -1; z <= 1; z++)
		for (int y = -1; y <= 1; y++)
		for (int x = -1; x <= 1; x++)
		{
			int3 neighbour = gridPos + make_int3(x, y, z);

			if (neighbour.x >= 0 && neighbour.y >= 0 && neighbour.z >= 0)
				_dens += cudaAddDensity(tid, neighbour, _pos, cellstart, cellend, spos);
		}

		dens[tid] = _dens;
		press[tid] = para.k*(dens[tid] - para.restDens);

		press[tid] -= Vm[tid] * para.voltage_constant;

		if(press[tid] < -para.max_pressure)
			press[tid] = -para.max_pressure;
		else if(press[tid] > para.max_pressure)
			press[tid] = para.max_pressure;
	}
}

__global__ void cudaCalcForce(float3* force, float3* spos, float3* sintermediate_vel, float3* vel, float* sVm, float* Inter_Vm, float* sIion, float* sStim, float* press,
	float* dens, unsigned int* index, unsigned int* cellstart, unsigned int* cellend, unsigned int num_particles)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;

	for(unsigned int tid = idx; tid < num_particles; tid += stride)
	{
		float3 _pos = spos[tid];
		float3 _intermediate_vel = sintermediate_vel[tid];
		float _press = press[tid];
		float current_Vm = sVm[tid];

		int3 gridPos = cudaCalcGridPos(_pos);

		float3 _force = make_float3(0.0f, 0.0f, 0.0f);
		float _inter_Vm = 0.0f;

		for (int z = -1; z <= 1; z++)
		for (int y = -1; y <= 1; y++)
		for (int x = -1; x <= 1; x++)
		{
			int3 neighbour = gridPos + make_int3(x, y, z);
			if (neighbour.x >= 0 && neighbour.y >= 0 && neighbour.z >= 0)
			{
				// float3 force = make_float3(0.0f, 0.0f, 0.0f);
				unsigned int _hash = cudaCalcGridHash(neighbour);
				unsigned int startIndex = cellstart[_hash];

				if (startIndex != 0xffffffff)
				{
					unsigned int endIndex = cellend[_hash];
					for (int j = startIndex; j != endIndex; j++)
					{
						float3 nb_pos = spos[j];
						float3 deltaPos = _pos - nb_pos;
						float nb_Vm = sVm[j];
						
						if (length(deltaPos) <= para.h)
						{
							float3 nb_vel = sintermediate_vel[j];
							float nb_dens = dens[j];
							float nb_press = press[j];
							
							float3 deltaVel = nb_vel - _intermediate_vel;

							/// Calculates the froce from the viscosity
							_force += para.mu * para.mass * deltaVel / nb_dens * Visco(deltaPos, para.h);

							/// Calculates the force from the pressure
							_force -= para.mass * (_press + nb_press) / (2.0f * nb_dens) * Spiky(deltaPos, para.h);

							_inter_Vm += (nb_Vm - current_Vm) * para.mass / dens[j] * B_spline_2(deltaPos, para.h);
						}
					}
				}
			}
		}

		unsigned int originalIndex = index[tid];
		
		_inter_Vm += (para.sigma / (para.beta_md * para.Cm)) * _inter_Vm - ((sIion[tid] - sStim[tid] * para.dt / para.mass) / para.Cm);
		
		// force[originalIndex] = (_force + make_float3(0.0f, -9.8f, 0.0f) * para.mass) / para.mass;
		force[originalIndex] = _force / para.mass; //(_force + make_float3(0.0f, -9.8f, 0.0f) * para.mass) / para.mass;
		Inter_Vm[originalIndex] = _inter_Vm / para.mass;
		vel[originalIndex] = _intermediate_vel;
	}
}

__global__ void cudaUpdateVelocityAndPosition(float3* pos, float3* vel, float3* force, float* Vm, float* Inter_Vm, unsigned int num_particles)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;

	for(unsigned int tid = idx; tid < num_particles; tid += stride)
	{
		vel[tid] += force[tid] * para.dt;
		pos[tid] += vel[tid] * para.dt;
		Vm[tid] += Inter_Vm[tid] * para.dt;

		if(Vm[tid] > para.max_voltage)
			Vm[tid] = para.max_voltage;
		if(Vm[tid] < -para.max_voltage)
			Vm[tid] = -para.max_voltage;
	}
}

/// This boundary setup generates waves, to avoid that the displacement should be lowered.
__global__ void cudaHandleBoundary(float3* pos, float3* vel, unsigned int num_particles)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;

	float displacement = 0.0001;// para.cellSize / 2.f;

	for(unsigned int tid = idx; tid < num_particles; tid += stride)
	{
		float x = pos[tid].x;
		float y = pos[tid].y;
		float z = pos[tid].z;

		if ( x > para.xmax)
		{
			pos[tid].x = x-displacement; 
			vel[tid].x = -vel[tid].x;
		}

		if ( x < para.xmin)
		{
			pos[tid].x = para.xmin + displacement; 
			vel[tid].x = -vel[tid].x;
		}

		if ( y > para.ymax)
		{
			pos[tid].y = y-displacement;
			vel[tid].y = -vel[tid].y;
		}

		if (y < para.ymin)
		{
			pos[tid].y = para.ymin + displacement;
			vel[tid].y = -vel[tid].y;
		}

		if (z > para.zmax)
		{
			pos[tid].z = z - displacement; 
			vel[tid].z = -vel[tid].z;
		}

		if (z < para.zmin)
		{
			pos[tid].z = para.zmin + displacement;
			vel[tid].z = -vel[tid].z;
		}
	}
}

/// Cycles through all the neighbors and adds their mass to the density
__device__ float3 cudaAddPartialVel(unsigned int tid, int3 gridPos, float3 pos, float3 corrected_vel, unsigned int* cellstart, unsigned int* cellend, float3* spos, float3 *scorrected_vel, float* dens)
{
	float3 _partialVel = make_float3(0.f, 0.f, 0.f);
	unsigned int _hash = cudaCalcGridHash(gridPos);
	unsigned int startIndex = cellstart[_hash];

	if (startIndex != 0xffffffff)
	{
		unsigned int endIndex = cellend[_hash];
		for (int j = startIndex; j != endIndex; j++)
		{
			float3 nb_pos = spos[j];
			float3 deltaPos = pos - nb_pos;
			
			_partialVel += (scorrected_vel[j] - corrected_vel) * Poly6(deltaPos, para.h) * para.mass / dens[j];
		}
	}

	return _partialVel;
}

__global__ void cudacalcIntermediateVel(float3 *dintermediate_vel, float3 *scorrected_vel,	float3* spos, float* dens, unsigned int* cellstart, unsigned int* cellend, unsigned int num_particles)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;

	for(unsigned int tid = idx; tid < num_particles; tid += stride)
	{
		float3 _pos = spos[tid];
		float3 _corrected_vel = scorrected_vel[tid];

		int3 gridPos = cudaCalcGridPos(_pos);

		float3 _partialVel = make_float3(0.0f, 0.0f, 0.0f);

		for (int z = -1; z <= 1; z++)
		for (int y = -1; y <= 1; y++)
		for (int x = -1; x <= 1; x++)
		{
			int3 neighbour = gridPos + make_int3(x, y, z);
			if (neighbour.x >= 0 && neighbour.y >= 0 && neighbour.z >= 0)
			{
				_partialVel += cudaAddPartialVel(tid, neighbour, _pos, _corrected_vel, cellstart, cellend, spos, scorrected_vel, dens);
			}
		}

		dintermediate_vel[tid] = _corrected_vel + _partialVel * para.velocity_mixing;
	}
}

__global__ void cudaCalculate_cell_model(float *dVm, float *dIion, float *dW, unsigned int num_particles)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;

	for(unsigned int tid = idx; tid < num_particles; tid += stride)
	{
		float u = (dVm[tid] - para.FH_Vr) * para.denom;
		float w = dW[tid];

		dIion[tid] += para.dt * (para.C1*u*(u - para.cell_model_temp)*(u - 1.0) + para.C2* dW[tid]) / para.mass ;

		dW[tid] = w + para.dt * para.C3 * (u - para.C4 * w) / para.mass;
	}
}

#endif
