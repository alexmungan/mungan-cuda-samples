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
#endif
