#ifndef TRI_SOLVERS_CUH
#define TRI_SOLVERS_CUH

/* This file contains triangular solvers called in main.cu */

//CPU SOLVER: used for correctness checks
__host__ void csr_solve_lower_tri_system(double AA[], int IA[], int JA[], int DA[], int nn, double x[], double r[]);




/* myGpuTriSolver_1: basic parallel tri solver on the gpu 
 * Requires two functions
 * 1) The first host function calls the kernel in a loop. 
 * 2) The second function is the kernel. Each kernel corresponds to the solution of one element of the output vector 
 *
 Params:
 * 1-4: the csc device matrix
 * numOfBlocksPerCol: host array that indicates the number of blocks used for each kernel launch
 * d_x: serves as both the input and output vector (inplace)
 */
 
__host__ void myGpuLowerTriSolver_1(double *d_csc_AA, int *d_csc_JA, int *d_csc_IA, int *d_csc_DA, int *numOfBlocksPerCol, int arrsize, int blocksize, double *d_x);
/* The kernel: performs a vector multiply-subtract operation */
__global__ void tri_VMS(double *d_csc_AA, int *d_csc_JA, int *d_csc_IA, int *d_csc_DA, double *d_x, int currCol);
/* Helper function: using the nnzPerColAbove/Below, determines the number of blocks needed for each kernel launch / column 
 * Params: 
 * numOfBlocksPerCol: array that holds the blocksizes for each col
 * nnzPerCol: either for above the diagonal or below the diagonal depending on whether you are going to launch an upper or lower tri solver
 */
__host__ void get_numOfBlocksPerCol(int *numOfBlocksPerCol, int *nnzPerCol, int blocksize, int arrsize);

/* Optimized gpu tri solver -- makes use of special storage format to maximize parallelism */
#include <cooperative_groups.h>
namespace cg = cooperative_groups;


/* Optimized gpu tri solver -- makes use of special storage format to maximize parallelism */
//Uses shared memory - so kernel launch cannot be used as global syn --> error: not all blocks are coresident so we get deadlock
template<int BLOCKSIZE>
__global__ void upperSolveShared(struct block *d_upperBlocks, double *d_x) {
	int tid = threadIdx.x;
	int bid = blockIdx.x; 
	struct block myBlock = d_upperBlocks[bid];
	cg::grid_group grid = cg::this_grid();

	//Each thread loads an element of solution vector in shared mem
	__shared__ double sx[BLOCKSIZE];
	int startRow = myBlock.startRow;
	int xidx = startRow + tid;
	if(xidx <= myBlock.endRow)
		sx[tid] = d_x[xidx];
		
	__syncthreads();

	//Each iteration does a parallel vector operation on 
	for(int i = 0; i < myBlock.numOfIterations; i++) {
		int k1 = myBlock.iterPtrs[i];
		int k2 = myBlock.iterPtrs[i+1] - 1;
		//Need loop in case more elements can be done this iteration than there are threads
		//i.e. k2-k1 > blocksize
		//Note: we can eliminate loop and its overhead if we limited each iteration
		//to have blocksize # of values, but then we would have no gaurantee that certain dependencies 
		//were resolved that iteration (that are required for the next iteration)
		for(int k = k1 + tid; k <= k2; k += BLOCKSIZE) {
			int xi = myBlock.col[i] - startRow;
			int j = myBlock.col[k];
			double term = myBlock.values[k] * d_x[j];
			atomicAdd(&sx[xi], -term);
		}
			
		grid.sync();
	}
	
	//Write the solution to the global output vector
	if(xidx <= myBlock.endRow)
		d_x[xidx] = sx[tid];
}

/* Optimized gpu tri solver -- makes use of special storage format to maximize parallelism */
//Uses kernel launch as global sync since cooperatives api was not working (deadlock)
//This requires 2 functions:
//1) The host function to call the solver in a loop 
//2) The actual kernel

__host__ void myGpuLowerTriSolver_2(struct block *h_upperBlocks, struct block *d_upperBlocks, double *d_x, int arrsize, int numOfBlocks, cudaStream_t *streams);

template<int BLOCKSIZE>
__global__ void upperSolveGlobal(struct block myBlock, double *d_x, int bid, int i) {
	int tid = threadIdx.x;

	//Each iteration does a parallel vector operation on 
	int k1 = myBlock.iterPtrs[i];
	int k2 = myBlock.iterPtrs[i+1] - 1;
	//Need loop in case more elements can be done this iteration than there are threads
	//i.e. k2-k1 > blocksize
	//Note: we can eliminate loop and its overhead if we limited each iteration
	//to have blocksize # of values, but then we would have no gaurantee that certain dependencies 
	//were resolved that iteration (that are required for the next iteration)
	for(int k = k1 + tid; k <= k2; k += BLOCKSIZE) {
		int xi = myBlock.row[k];
		int j = myBlock.col[k];
		double term = myBlock.values[k] * d_x[j];
		atomicAdd(&d_x[xi], -term);
	}
			
	
}
#endif
