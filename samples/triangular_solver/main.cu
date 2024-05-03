/* Contains main() which calls the tri solve kernels */
/* Their are multiple implementations of tri solve - this file tests and profiles them */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "mm2csr.h"

#include <cusparse.h>

#include "gpuErrHandler.cuh"
#include "cuBLASErrHandler.cuh"
#include "cuSPARSEErrHandler.cuh"

#define eps 1.0e-5

//CPU SOLVER: used for correctness checks
void csr_solve_lower_tri_system(double AA[], int IA[], int JA[], int DA[], int nn, double x[])
{
// Purpose: compute (I + Low_tri(A))*x, return modified x.
// x[] input as the rhs vector, output as the solution.

    int idx,k1,k2,k,j;

    for (idx = 0; idx < nn; idx++) {   // compressed sparse row format
        k1 = IA[idx];
        k2 = DA[idx] - 1;
        for (k = k1; k <= k2; k++) {
            j = JA[k];
            x[idx] -= AA[k]*x[j];
        }
    }
}

int main() {
	//Host Matrix containers
	double *AA;
	int *IA, *JA, *DA;
	int arrsize, nnz;

	//Get the test matrix 
	char *filepath = "../../data/matrices/sparse/posdef/bcsstk13.mtx";
	mm2csr(filepath, &AA, &IA, &JA, &DA, &arrsize, &nnz);
	printf("Matrix file read in.\n");
	
	//Device Matrix containers
	double *d_AA;
	int *d_IA, *d_JA, *d_DA;
	gpuErrchk(cudaMalloc((void**)&d_AA, nnz*sizeof(double)));
	gpuErrchk(cudaMalloc((void**)&d_IA, (arrsize+1)*sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_JA, nnz*sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_DA, arrsize*sizeof(int)));	
	//Memcpy to Device
    	gpuErrchk(cudaMemcpy(d_AA, AA, nnz*sizeof(double), cudaMemcpyHostToDevice));
    	gpuErrchk(cudaMemcpy(d_IA, IA, (arrsize+1)*sizeof(int), cudaMemcpyHostToDevice));
    	gpuErrchk(cudaMemcpy(d_JA, JA, nnz*sizeof(int), cudaMemcpyHostToDevice));
    	gpuErrchk(cudaMemcpy(d_DA, DA, arrsize*sizeof(int), cudaMemcpyHostToDevice));

	//Generate random RHS for testing
	size_t vecSize = sizeof(double) * arrsize;
	double *r = (double *)malloc(vecSize);
	srand( time(NULL) );
    	for(int i = 0; i < arrsize; i++) r[i] = rand() / (double)RAND_MAX;

/****	Run CPU solver (results used for correctness checks)	****/
	double *r_correct = (double *)malloc(vecSize);
	memcpy(r_correct, r, vecSize);
	
    	clock_t startTime = clock();
	csr_solve_lower_tri_system(AA, IA, JA, DA, arrsize, r_correct);
	clock_t stopTime = clock();
	double cpuTime = ((double)stopTime-startTime)/CLOCKS_PER_SEC;
	
/*******************************************************************/

/****	Profile cuSPARSE tri solver ****/
	double *r_cusparse = (double *)malloc(vecSize);
	memcpy(r_cusparse, r, vecSize);
	double *d_r_cusparse;
    	gpuErrchk(cudaMalloc((void**)&d_r_cusparse, vecSize));
    	double *x_cusparse = (double *)malloc(vecSize);
    	startTime = clock();
    	gpuErrchk(cudaMemcpy(d_r_cusparse, r_cusparse, vecSize, cudaMemcpyHostToDevice));
    	double *d_x_cusparse;
    	gpuErrchk(cudaMalloc((void**)&d_x_cusparse, vecSize));
    	gpuErrchk(cudaMemset(d_x_cusparse, 0.0f, vecSize));
	//cusparse handle
	cusparseHandle_t handle;
    	cusparseErrchk(cusparseCreate(&handle));
    	//cusparse matrix description
    	cusparseSpMatDescr_t matA;
    	cusparseDnVecDescr_t vecR;
    	cusparseDnVecDescr_t vecX;
    	//Create cusparse CSR data structure
    	cusparseErrchk( cusparseCreateCsr(&matA, arrsize, arrsize, nnz,
                                      d_IA, d_JA, d_AA,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) );
        // Create dense vector r 
        cusparseErrchk( cusparseCreateDnVec(&vecR, arrsize, d_r_cusparse, CUDA_R_64F) );
        // Create dense vector x 
        cusparseErrchk( cusparseCreateDnVec(&vecX, arrsize, d_x_cusparse, CUDA_R_64F) );
        // Create data structure that holds analysis data 
        cusparseSpSVDescr_t  spsvDescr;
        cusparseErrchk( cusparseSpSV_createDescr(&spsvDescr) );
        //Set fill mode attribute of matrix description
    	cusparseFillMode_t fillmode = CUSPARSE_FILL_MODE_LOWER;
    	cusparseErrchk( cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_FILL_MODE,
                                              &fillmode, sizeof(fillmode)) );
        //Set Unit|Non-Unit diagonal attribute
    	cusparseDiagType_t diagtype = CUSPARSE_DIAG_TYPE_NON_UNIT;
    	cusparseErrchk( cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_DIAG_TYPE,
                                              &diagtype, sizeof(diagtype)) );
        //Allocate External buffer for analysis
        void *dBuffer = NULL;
        size_t bufferSize = 0;
        double alpha = 1.0;
    	cusparseErrchk( cusparseSpSV_bufferSize(
                                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, vecX, vecR, CUDA_R_64F,
                                CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr,
                                &bufferSize) );
        gpuErrchk( cudaMalloc(&dBuffer, bufferSize) );
	cusparseErrchk( cusparseSpSV_analysis(
                                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, vecX, vecR, CUDA_R_64F,
                                CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr, dBuffer) );
        // execute SpSV
    	cusparseErrchk( cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       &alpha, matA, vecX, vecR, CUDA_R_64F,
                                       CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr) );
        //Memcpy to Host (results)	
	gpuErrchk(cudaMemcpy(x_cusparse, d_x_cusparse, vecSize, cudaMemcpyDeviceToHost));
	
	stopTime = clock();
	double cusparseTriSolverTime = ((double)stopTime-startTime)/CLOCKS_PER_SEC;
	//free device result
	gpuErrchk(cudaFree(d_r_cusparse));
	gpuErrchk(cudaFree(d_x_cusparse));
	
	//TEST FOR CORRECTNESS
	bool passed = true;
	for(int i = 0; i < arrsize; i++) {
		if(abs(x_cusparse[i] - r_correct[i]) > eps) {
			passed = false;
			fprintf(stderr, "cuSPARSE_tri_solver failed at cusparse_x[%d] = %f, x[%d] = %f!\n", i, r_cusparse[i], i, r_correct[i]);
		}
	}
	if(passed)
		printf("cuSPARSE_tri_solver PASS");
/***************************************/
	
/****	Output Results for Unit Triangular Solvers	****/
	printf("(Single-threaded) CPU_tri_solver execution time: %fms\n", cpuTime);
	printf("cuSPARSE_tri_solver execution time: %fms\n", cusparseTriSolverTime);
/***********************************************************/
	
/****	Free Resources	****/
	//Free host matrix
	free(AA);
	free(IA);
	free(JA);
	free(DA);
	//Free device matrix
	gpuErrchk(cudaFree(d_AA));
	gpuErrchk(cudaFree(d_IA));
	gpuErrchk(cudaFree(d_JA));
	gpuErrchk(cudaFree(d_DA));
	//Free RHS
	free(r);
	free(r_correct);
	free(r_cusparse);free(x_cusparse);
	 // destroy matrix/vector descriptors
    	cusparseErrchk( cusparseDestroySpMat(matA) );
    	cusparseErrchk( cusparseDestroyDnVec(vecR) );
    	cusparseErrchk( cusparseDestroyDnVec(vecX) );
    	cusparseErrchk( cusparseSpSV_destroyDescr(spsvDescr));
    	cusparseErrchk(cusparseDestroy(handle));
/***************************/


	return 0;
}
