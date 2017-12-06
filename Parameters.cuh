#ifndef Parameters_cuh
#define Parameters_cuh

#include "device_launch_parameters.h"
#include "cuda_runtime.h"

struct Parameters
{
	float dt;		/// Delta time for the simulation
	
	int3 gridSize;	/// Grid size
	int	 cellSize;	/// Size of each cell in the grid

	float3 gravity;

	/// Boundaries for the simulation
	float xmin;		
	float xmax;
	float ymin;
	float ymax;
	float zmin;
	float zmax;

	float mass;		/// Mass of each particle
	float h;		/// Core radius h
	float restDens;	/// Stand density
	float k;		/// ideal pressure formulation k; Stiffness of the fluid. The lower the value, the stiffer the fluid.
	float mu;		/// Viscosity

	/// SM parameters
	float alpha;
	float beta;
	bool quadraticMatch;
	bool volumeConservation;
	bool allowFlip;

	/// SPH SM parameters
	float velocity_mixing;

	/// Monodomain parameters
	float max_voltage;
	float max_pressure;
	float voltage_constant;
	float stim_strength;

	float Cm;
	float beta_md;
	float sigma;

	/// Membrane model
	float FH_Vt;
	float FH_Vp;
	float FH_Vr;
	float C1;
	float C2;
	float C3;
	float C4;

	float denom;
	float cell_model_temp;
};

#endif