/* This file contains triangular solvers called in main.cu */

//CPU SOLVER: used for correctness checks
__host__ void csr_solve_lower_tri_system(double AA[], int IA[], int JA[], int DA[], int nn, double x[], double r[])
{
// Purpose: compute (I + Low_tri(A))*x, return modified x.
// x[] input as the rhs vector, output as the solution.

    int idx,k1,k2,k,j;

    for (idx = 0; idx < nn; idx++) {   // compressed sparse row format
        k1 = IA[idx];
        k2 = DA[idx] - 1;
        x[idx] = r[idx];
        for (k = k1; k <= k2; k++) {
            j = JA[k];
            x[idx] -= AA[k]*x[j];
        }
    }
}

/* myGpuTriSolver_1: basic parallel tri solver on the gpu 
 * Requires two functions
 * 1) The first host function calls the kernel in a loop. 
 * 2) The kernel. Each kernel call corresponds to the solution of one element of the output vector 
 */
 /* The kernel: performs a vector multiply-subtract operation */
__global__ void tri_VMS(double *d_csc_AA, int *d_csc_JA, int *d_csc_IA, int *d_csc_DA, double *d_x, int currCol) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
 	int k1 = d_csc_JA[currCol];
 	int k2 = d_csc_DA[currCol] - 1;
 	int k = k1 + gid;
 	int i;
 	if(k <= k2) {
 		i = d_csc_IA[k];
 		d_x[i] -= d_csc_AA[k] * d_x[currCol]; 
 	}
}

 /*Params:
 * 1-4: the csc device matrix
 * numOfBlocksPerCol: host array that indicates the number of blocks used for each kernel launch
 * d_x: serves as both the input and output vector (inplace)
 */
 
__host__ void myGpuLowerTriSolver_1(double *d_csc_AA, int *d_csc_JA, int *d_csc_IA, int *d_csc_DA, int *numOfBlocksPerCol, int arrsize, int blocksize, double *d_x) { 	
 	//TODO: loop overhead???? How to reduce each iterations cost???
 	for(int i = 0; i < arrsize; i++) {
 		tri_VMS<<<numOfBlocksPerCol[i],blocksize>>>(d_csc_AA, d_csc_JA, d_csc_IA, d_csc_DA, d_x, i);
 	}
}

/* Helper function: using the nnzPerColAbove/Below, determines the number of blocks needed for each kernel launch / column 
 * Params: 
 * numOfBlocksPerCol: array that holds the blocksizes for each col
 * nnzPerCol: either for above the diagonal or below the diagonal depending on whether you are going to launch an upper or lower tri solver
 */
__host__ void get_numOfBlocksPerCol(int *numOfBlocksPerCol, int *nnzPerCol, int blocksize, int arrsize) {
	for(int i = 0; i < arrsize; i++) {
		numOfBlocksPerCol[i] = (nnzPerCol[i] + blocksize - 1) / blocksize ;
	}
}
