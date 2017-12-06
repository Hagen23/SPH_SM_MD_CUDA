#include "Solver.h"
#include <string>

using namespace std;

/// Definitions for the CUDA Kernel calls and Kernels
extern "C"
{
	/// Calculates the grid and block sizes
	void computeGridSize(unsigned int n, unsigned int blockSize, unsigned int &numBlocks, unsigned int &numThreads);

	/// Calls a function and checks if an error happened
	void HandleError(cudaError_t status, string message);

	/// Copies the simulation parameters to constant memory
	void SetParameters(Parameters *p);

	/// Calculats the hash of every particle: calculates the cell where it is in the grid
	void CalHash(unsigned int* index, unsigned int* hash, float3* pos, unsigned int num_particles);

	/// Sorts the particles according to their hash
	void SortParticles(unsigned int *hash, unsigned int *index, unsigned int num_particles);

	/// Reorders the particles according to the sorted hash
	void ReorderDataAndFindCellStart(
		unsigned int* cellstart,
		unsigned int* cellend,
		float3* spos,
		float3* svel,
		float3* scorr_vel,
		float* sVm,
		float* sIion,
		float* sStim,
		unsigned int* hash,
		unsigned int* index,
		float3* pos,
		float3* vel,
		float3* corr_vel,
		float* Vm,
		float* Iion,
		float* stim,
		unsigned int num_particles,
		unsigned int gridNum);


	void calcIntermediateVel(
		float3 *dintermediate_vel,
		float3 *scorrected_vel,
		float3* spos,
		float* dens,
		unsigned int* cellstart, 
		unsigned int* cellend, 
		unsigned int num_particles);

	void calculateCellModel(
		float *dVm, 
		float *dIion,
		float *dW, 
		unsigned int num_particles);

	void CalcDensityPressure(
		float* dens,
		float* pres,
		float* Vm,
		unsigned int* cellstart, 
		unsigned int* cellend, 
		float3 *spos, 
		unsigned int num_particles);

	void CalcForce(float3* force, 
		           float3* spos, 
				   float3* sintermediate_vel,
				   float3* vel,
				   float* sVm,
				   float* Inter_Vm, 
				   float* Iion, 
				   float* sStim, 
				   float* press, 
				   float* dens, 
				   unsigned int* index, 
				   unsigned int* cellstart, 
				   unsigned int* cellend, 
				   unsigned int num_particles);

	void UpdateVelocityAndPosition(float3* pos, 
		                           float3* vel, 
								   float3* force,
								   float* Vm,
								   float* Inter_Vm,
								   unsigned int num_particles);

	void HandleBoundary(float3* pos, 
		                float3* vel, 
						unsigned int num_particles);

}

#define CHECK(ptr, message)  {if(ptr==NULL){cerr<<message<<endl;exit(1);}}

Solver::Solver(unsigned int _num_particles) :num_particles(_num_particles)
{
	float sigma_i = 0.893, sigma_e = 0.67;
	size1 = num_particles*sizeof(float);
	size3 = num_particles*sizeof(float3);
	gridNum = GRID_SIZE * GRID_SIZE * GRID_SIZE;

	/// Set simulation parameters
	pa.mass = 30.f;
	pa.dt = 0.03f;

	pa.xmin = 0.0f;
	pa.xmax = GRID_SIZE;
	pa.ymin = 0.0f;
	pa.ymax = GRID_SIZE;
	pa.zmin = 0.0f;
	pa.zmax = GRID_SIZE;

	pa.gridSize.x = GRID_SIZE;
	pa.gridSize.y = GRID_SIZE;
	pa.gridSize.z = GRID_SIZE;
	pa.cellSize = 1;

	pa.h = 1.7f;
	pa.k = 0.008f;
	pa.restDens = 1112.0f;
	pa.mu = 100.0f;

	pa.alpha = 0.7f;
	pa.beta = 0.2f;
	pa.quadraticMatch = false;
	pa.volumeConservation = true;
	pa.allowFlip = false;

	pa.velocity_mixing = 0.5f;

	pa.max_voltage = 1000;
	pa.max_pressure = 15000;
	pa.voltage_constant = 50;
	pa.stim_strength = 80000.0f;

	pa.Cm = 0.01f;
	pa.beta_md = 50;
	pa.sigma = sigma_i * sigma_e / ( sigma_i + sigma_e);
	
	pa.FH_Vt = -75.0f;
	pa.FH_Vp = 15.0f;
	pa.FH_Vr = -85.0f;
	pa.C1 = 0.175f;
	pa.C2 = 0.03f;
	pa.C3 = 0.011f;
	pa.C4 = 0.55f;

	pa.denom = 1.0f / (pa.FH_Vp - pa.FH_Vr);
	pa.cell_model_temp = (pa.FH_Vt - pa.FH_Vr) * pa.denom;

	/// Memory allocation
	hpos=(float3*)malloc(size3);
	CHECK(hpos, "Failed to allocate memory of hpos!");

	hvel = (float3*)malloc(size3);
	CHECK(hvel, "Failed to allocate memory of hvel!");

	predicted_vel = (m3Vector*)malloc(sizeof(m3Vector) * num_particles);
	hcorrected_vel = (float3*)malloc(size3);
	mOriginalPos = (m3Vector*)malloc(sizeof(m3Vector) * num_particles);
	mGoalPos = (m3Vector*)malloc(sizeof(m3Vector) * num_particles);
	mFixed = (bool*)malloc(sizeof(bool)*num_particles);

	hVm = (float*)malloc(size1);
	hStim = (float*)malloc(size1);

	HandleError(cudaMalloc((void**) &dpos, size3), "Failed to allocate memory of dpos!");
	HandleError(cudaMalloc((void**) &dvel, size3), "Failed to allocate memory of dvel!");
	HandleError(cudaMalloc((void**) &dspos, size3), "Failed to allocate memory of dspos!");
	HandleError(cudaMalloc((void**) &dsvel, size3), "Failed to allocate memory of dsvel!");
	HandleError(cudaMalloc((void**) &ddens, size1), "Failed to allocate memory of ddens!");
	HandleError(cudaMalloc((void**) &dforce, size3), "Failed to allocate memory of dforce!");
	HandleError(cudaMalloc((void**) &dpress, size1), "Failed to allocate memory of dpress!");

	HandleError(cudaMalloc((void**) &dindex, num_particles*sizeof(unsigned int)), "Failed to allocate memory of dindex");	
	HandleError(cudaMalloc((void**) &dhash, num_particles*sizeof(unsigned int)), "Failed to allocate memory of dhash");
	HandleError(cudaMalloc((void**) &dcellStart, gridNum*sizeof(unsigned int)), "Failed to allocate memory of dcellstart");
	HandleError(cudaMalloc((void**) &dcellEnd, gridNum*sizeof(unsigned int)), "Failed to allocate memory of dcellend");

	HandleError(cudaMalloc((void**) &dcorrected_vel, size3), "Failed to allocate memory of dcorrected");
	HandleError(cudaMalloc((void**) &dscorrected_vel, size3), "Failed to allocate memory of dscorrected");
	HandleError(cudaMalloc((void**) &dintermediate_vel, size3), "Failed to allocate memory of dintermediate");

	HandleError(cudaMalloc((void**) &dVm, size1), "Failed to allocate memory of dVm");
	HandleError(cudaMalloc((void**) &dsVm, size1), "Failed to allocate memory of dVm");
	HandleError(cudaMalloc((void**) &dIntermediate_vm, size1), "Failed to allocate memory of dIntermediate_vm");
	HandleError(cudaMalloc((void**) &dIion, size1), "Failed to allocate memory of dIion");
	HandleError(cudaMalloc((void**) &dsIion, size1), "Failed to allocate memory of dsIion");
	HandleError(cudaMalloc((void**) &dW, size1), "Failed to allocate memory of dW");
	HandleError(cudaMalloc((void**) &dStim, size1), "Failed to allocate memory of dStim");
	HandleError(cudaMalloc((void**) &dsStim, size1), "Failed to allocate memory of dsStim");

	InitParticles();
	set_stim();

	// HandleError(cudaMemcpy(dpos, hpos, size3, cudaMemcpyHostToDevice), "Failed to copy memory of hpos!");
	HandleError(cudaMemset(dvel, 0, size3), "Failed to memset dvel!");
	HandleError(cudaMemset(dsvel, 0, size3), "Failed to memset dsvel!"); 
	HandleError(cudaMemset(dspos, 0, size3), "Failed to memset dspos!"); 
	HandleError(cudaMemset(ddens, 0, size1), "Failed to memset ddens!");
	HandleError(cudaMemset(dforce, 0, size3), "Failed to memset dforce!");
	HandleError(cudaMemset(dpress, 0, size1), "Failed to memset dpress!");

	HandleError(cudaMemset(dintermediate_vel, 0, size3), "Failed to memset dintermediate!");
	HandleError(cudaMemset(dcorrected_vel, 0, size3), "Failed to memset dcorrected!");
	HandleError(cudaMemset(dscorrected_vel, 0, size3), "Failed to memset dcorrected!");

	HandleError(cudaMemset(dVm, 0, size1), "Failed to memset dVm!");
	HandleError(cudaMemset(dsVm, 0, size1), "Failed to memset dVm!");
	HandleError(cudaMemset(dIntermediate_vm, 0, size1), "Failed to memset dIntermediate_vm!");
	HandleError(cudaMemset(dIion, 0, size1), "Failed to memset dIion!");
	HandleError(cudaMemset(dsIion, 0, size1), "Failed to memset dsIion!");
	HandleError(cudaMemset(dW, 0, size1), "Failed to memset dW!");
	HandleError(cudaMemset(dStim, 0, size1), "Failed to memset dStim!");
	HandleError(cudaMemset(dsStim, 0, size1), "Failed to memset dsStim!");

	HandleError(cudaMemset(dindex, 0, size1), "Failed to memset dindex!");
	HandleError(cudaMemset(dhash, 0, size1), "Failed to memset dhash!");
	HandleError(cudaMemset(dcellStart, 0, gridNum*sizeof(unsigned int)), "Failed to memset dcellstart!");
	HandleError(cudaMemset(dcellEnd, 0, gridNum*sizeof(unsigned int)), "Failed to memset dcellend!");

	SetParameters(&pa);
}

Solver::~Solver()
{
	free(hpos);
	free(hvel);

	HandleError(cudaFree(dpos), "Failed to free dpos!");
	HandleError(cudaFree(dvel), "Failed to free dvel!");
	HandleError(cudaFree(ddens), "Failed to free ddens!");
	HandleError(cudaFree(dforce), "Failed to free dforce!");
	HandleError(cudaFree(dpress), "Failed to free dpress!");
	HandleError(cudaFree(dhash), "Failed to free dhash!");
	HandleError(cudaFree(dindex), "Failed to free dindex!");
	HandleError(cudaFree(dcellStart), "Failed to free dcellStart!");
	HandleError(cudaFree(dcellEnd), "Failed to free dcellEnd!");

	HandleError(cudaFree(dspos), "Failed to free dspos!");
	HandleError(cudaFree(dsvel), "Failed to free dsvel!");
}


void Solver::InitParticles()
{
	/// Initializing a set number of particles
	int index = 0;
	for (int i = 0; i < 32; i++)
	for (int j = 0; j < 32; j++)
	for (int k = 0; k < 16; k++)
	{
		index = k * 32 * 32 + j * 32 + i;
		hpos[index].x = mOriginalPos[index].x = mGoalPos[index].x = i + 16;
		hpos[index].y = mOriginalPos[index].y = mGoalPos[index].y = j + 0;
		hpos[index].z = mOriginalPos[index].z = mGoalPos[index].z = k + 24;

		mFixed[index] = false;
	}
}

void Solver::apply_external_forces()
{
	/// Gravity
	for (int i = 0; i < num_particles; i++)
	{
		if (mFixed[i]) continue;
		predicted_vel[i] = m3Vector(hvel[i].x, hvel[i].y, hvel[i].z) + (m3Vector(0.0f, -9.8f, 0.0f) * pa.dt) / pa.mass;
	}
}

void Solver::calculate_corrected_velocity()
{
	/// Computes predicted velocity from forces except viscoelastic and pressure
	apply_external_forces();

	/// Calculates corrected velocity
	projectPositions();

	m3Real time_delta_1 = 1.0f / pa.dt;

	for (int i = 0; i < num_particles; i++)
	{
		m3Vector temp = predicted_vel[i] + (mGoalPos[i] - m3Vector(hpos[i].x, hpos[i].y, hpos[i].z)) * time_delta_1 * pa.alpha;
		hcorrected_vel[i] = make_float3(temp.x, temp.y, temp.z);
	}
}

void Solver::projectPositions()
{
	if (num_particles <= 1) return;
	int i, j, k;

	// center of mass
	m3Vector cm, originalCm;
	cm.zero(); originalCm.zero();
	float mass = 0.0f;

	for (i = 0; i < num_particles; i++)
	{
		m3Real m = pa.mass;
		if (mFixed[i]) m *= 10000.0f;
		mass += m;
		cm += m3Vector(hpos[i].x, hpos[i].y, hpos[i].z) * m;
		originalCm += mOriginalPos[i] * m;
	}

	cm /= mass;
	originalCm /= mass;

	m3Vector p, q;

	m3Matrix Apq, Aqq;

	Apq.zero();
	Aqq.zero();

	for (i = 0; i < num_particles; i++)
	{
		p = m3Vector(hpos[i].x, hpos[i].y, hpos[i].z) - cm;
		q = mOriginalPos[i] - originalCm;
		m3Real m = pa.mass;

		Apq.r00 += m * p.x * q.x;
		Apq.r01 += m * p.x * q.y;
		Apq.r02 += m * p.x * q.z;

		Apq.r10 += m * p.y * q.x;
		Apq.r11 += m * p.y * q.y;
		Apq.r12 += m * p.y * q.z;

		Apq.r20 += m * p.z * q.x;
		Apq.r21 += m * p.z * q.y;
		Apq.r22 += m * p.z * q.z;

		Aqq.r00 += m * q.x * q.x;
		Aqq.r01 += m * q.x * q.y;
		Aqq.r02 += m * q.x * q.z;

		Aqq.r10 += m * q.y * q.x;
		Aqq.r11 += m * q.y * q.y;
		Aqq.r12 += m * q.y * q.z;

		Aqq.r20 += m * q.z * q.x;
		Aqq.r21 += m * q.z * q.y;
		Aqq.r22 += m * q.z * q.z;
	}

	if (!pa.allowFlip && Apq.determinant() < 0.0f)
	{  	// prevent from flipping
		Apq.r01 = -Apq.r01;
		Apq.r11 = -Apq.r11;
		Apq.r22 = -Apq.r22;
	}

	m3Matrix R, S;
	m3Matrix::polarDecomposition(Apq, R, S);

	if (!pa.quadraticMatch)
	{	// --------- linear match

		m3Matrix A = Aqq;
		A.invert();
		A.multiply(Apq, A);

		if (pa.volumeConservation)
		{
			m3Real det = A.determinant();
			if (det != 0.0f)
			{
				det = 1.0f / sqrt(fabs(det));
				if (det > 2.0f) det = 2.0f;
				A *= det;
			}
		}

		m3Matrix T = R * (1.0f - pa.beta) + A * pa.beta;

		for (i = 0; i < num_particles; i++)
		{
			if (mFixed[i]) continue;
			q = mOriginalPos[i] - originalCm;
			mGoalPos[i] = T.multiply(q) + cm;
		}
	}
	else
	{	// -------------- quadratic match---------------------

		m3Real A9pq[3][9];

		for (int i = 0; i < 3; i++)
		for (int j = 0; j < 9; j++)
			A9pq[i][j] = 0.0f;

		m9Matrix A9qq;
		A9qq.zero();

		for (int i = 0; i < num_particles; i++)
		{
			p = m3Vector(hpos[i].x, hpos[i].y, hpos[i].z) - cm;
			q = mOriginalPos[i] - originalCm;

			m3Real q9[9];
			q9[0] = q.x; q9[1] = q.y; q9[2] = q.z; q9[3] = q.x*q.x; q9[4] = q.y*q.y; q9[5] = q.z*q.z;
			q9[6] = q.x*q.y; q9[7] = q.y*q.z; q9[8] = q.z*q.x;

			m3Real m = pa.mass;
			A9pq[0][0] += m * p.x * q9[0];
			A9pq[0][1] += m * p.x * q9[1];
			A9pq[0][2] += m * p.x * q9[2];
			A9pq[0][3] += m * p.x * q9[3];
			A9pq[0][4] += m * p.x * q9[4];
			A9pq[0][5] += m * p.x * q9[5];
			A9pq[0][6] += m * p.x * q9[6];
			A9pq[0][7] += m * p.x * q9[7];
			A9pq[0][8] += m * p.x * q9[8];

			A9pq[1][0] += m * p.y * q9[0];
			A9pq[1][1] += m * p.y * q9[1];
			A9pq[1][2] += m * p.y * q9[2];
			A9pq[1][3] += m * p.y * q9[3];
			A9pq[1][4] += m * p.y * q9[4];
			A9pq[1][5] += m * p.y * q9[5];
			A9pq[1][6] += m * p.y * q9[6];
			A9pq[1][7] += m * p.y * q9[7];
			A9pq[1][8] += m * p.y * q9[8];

			A9pq[2][0] += m * p.z * q9[0];
			A9pq[2][1] += m * p.z * q9[1];
			A9pq[2][2] += m * p.z * q9[2];
			A9pq[2][3] += m * p.z * q9[3];
			A9pq[2][4] += m * p.z * q9[4];
			A9pq[2][5] += m * p.z * q9[5];
			A9pq[2][6] += m * p.z * q9[6];
			A9pq[2][7] += m * p.z * q9[7];
			A9pq[2][8] += m * p.z * q9[8];

			for (j = 0; j < 9; j++)
			for (k = 0; k < 9; k++)
				A9qq(j, k) += m * q9[j] * q9[k];
		}

		A9qq.invert();

		m3Real A9[3][9];
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 9; j++)
			{
				A9[i][j] = 0.0f;
				for (k = 0; k < 9; k++)
					A9[i][j] += A9pq[i][k] * A9qq(k, j);

				A9[i][j] *= pa.beta;
				if (j < 3)
					A9[i][j] += (1.0f - pa.beta) * R(i, j);
			}
		}

		m3Real det =
			A9[0][0] * (A9[1][1] * A9[2][2] - A9[2][1] * A9[1][2]) -
			A9[0][1] * (A9[1][0] * A9[2][2] - A9[2][0] * A9[1][2]) +
			A9[0][2] * (A9[1][0] * A9[2][1] - A9[1][1] * A9[2][0]);

		if (!pa.allowFlip && det < 0.0f) {         		// prevent from flipping
			A9[0][1] = -A9[0][1];
			A9[1][1] = -A9[1][1];
			A9[2][2] = -A9[2][2];
		}

		if (pa.volumeConservation)
		{
			if (det != 0.0f)
			{
				det = 1.0f / sqrt(fabs(det));
				if (det > 2.0f) det = 2.0f;

				for (int i = 0; i < 3; i++)
				for (int j = 0; j < 9; j++)
					A9[i][j] *= det;
			}
		}

		for (int i = 0; i < num_particles	; i++)
		{
			if (mFixed[i]) continue;
			q = mOriginalPos[i] - originalCm;

			mGoalPos[i].x = A9[0][0] * q.x + A9[0][1] * q.y + A9[0][2] * q.z + A9[0][3] * q.x*q.x + A9[0][4] * q.y*q.y +
				A9[0][5] * q.z*q.z + A9[0][6] * q.x*q.y + A9[0][7] * q.y*q.z + A9[0][8] * q.z*q.x;

			mGoalPos[i].y = A9[1][0] * q.x + A9[1][1] * q.y + A9[1][2] * q.z + A9[1][3] * q.x*q.x + A9[1][4] * q.y*q.y +
				A9[1][5] * q.z*q.z + A9[1][6] * q.x*q.y + A9[1][7] * q.y*q.z + A9[1][8] * q.z*q.x;

			mGoalPos[i].z = A9[2][0] * q.x + A9[2][1] * q.y + A9[2][2] * q.z + A9[2][3] * q.x*q.x + A9[2][4] * q.y*q.y +
				A9[2][5] * q.z*q.z + A9[2][6] * q.x*q.y + A9[2][7] * q.y*q.z + A9[2][8] * q.z*q.x;

			mGoalPos[i] += cm;
		}
	}
}

void Solver::set_stim()
{
	isStimOn = true;
	for(int i = 0; i < num_particles; i++)
	{
		float3 position = hpos[i];
		
		if( (position.x >= 16 && position.x <=20) || (position.x >= 43 && position.x <= 47))
		{
			hStim[i] = pa.stim_strength;

			if (position.y == 0.0f)
				mFixed[i] = true;
		}
	}
}

void Solver::set_stim_off()
{
	isStimOn = false;
	for(int i = 0; i < num_particles; i++)
	{
		hStim[i] = 0.f;
	}
}

void Solver::Update()
{
	calculate_corrected_velocity();

	HandleError(cudaMemcpy(dcorrected_vel, hcorrected_vel, size3, cudaMemcpyHostToDevice), "Failed to copy memory of dcorrected_vel!");
	HandleError(cudaMemcpy(dpos, hpos, size3, cudaMemcpyHostToDevice), "Failed to copy host pos to device in update!");
	HandleError(cudaMemcpy(dStim, hStim, size1, cudaMemcpyHostToDevice), "Failed to copy host stim to device in update!");

	calculateCellModel(dVm, dIion, dW, num_particles);

	CalHash(dindex, dhash, dpos, num_particles);

	SortParticles(dhash, dindex, num_particles);

	ReorderDataAndFindCellStart(dcellStart, dcellEnd, dspos, dsvel, dscorrected_vel, dsVm, dsIion, dsStim, dhash, dindex, dpos, dvel, dcorrected_vel, dVm, dIion, dStim, num_particles, gridNum);
	
	CalcDensityPressure(ddens, dpress, dsVm, dcellStart, dcellEnd, dspos, num_particles);

	calcIntermediateVel(dintermediate_vel, dscorrected_vel, dspos, ddens, dcellStart, dcellEnd, num_particles);
	
	CalcForce(dforce, dspos, dintermediate_vel, dvel, dsVm, dIntermediate_vm, dsIion, dsStim, dpress, ddens, dindex, dcellStart, dcellEnd, num_particles);

	UpdateVelocityAndPosition(dpos, dvel, dforce, dVm, dIntermediate_vm, num_particles);

	HandleBoundary(dpos, dvel, num_particles);

	HandleError(cudaMemcpy(hpos, dpos, size3, cudaMemcpyDeviceToHost), "Failed to copy device pos to host in update!");
	HandleError(cudaMemcpy(hvel, dvel, size3, cudaMemcpyDeviceToHost), "Failed to copy device vel to host in update!");

	HandleError(cudaMemcpy(hVm, dVm, size1, cudaMemcpyDeviceToHost), "Failed to copy device Vm to host in update!");
}
