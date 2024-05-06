#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "mm2csr.h"

#include <cusparse.h>

#include "gpuErrHandler.cuh"
#include "cuBLASErrHandler.cuh"
#include "cuSPARSEErrHandler.cuh"

#include "csr2csc.cuh"
#include "tri_solvers.cuh"

#define eps 1.0e-5
#define loop 1024

//Random number generator
double RNG(double min, double max) {
    double scale = rand() / (double)RAND_MAX;  // [0, 1]
    return min + scale * (max - min);          // [min, max]
}

int main() {
	//Host Matrix containers
	double *AA;
	int *IA, *JA, *DA;
	int arrsize, nnz;

	//Get the test matrix 
	char *filepath = "../../data/matrices/sparse/posdef/bmwcra_1.mtx";
	mm2csr(filepath, &AA, &IA, &JA, &DA, &arrsize, &nnz);
	printf("Matrix file read in.\n");
	
	//Device Matrix containers
	double *d_AA;
	int *d_IA, *d_JA, *d_DA;
	gpuErrchk(cudaMalloc((void**)&d_AA, nnz*sizeof(double)));
	gpuErrchk(cudaMalloc((void**)&d_IA, (arrsize+1)*sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_JA, nnz*sizeof(int)));
	//gpuErrchk(cudaMalloc((void**)&d_DA, arrsize*sizeof(int)));	
	//Memcpy to Device
    	gpuErrchk(cudaMemcpy(d_AA, AA, nnz*sizeof(double), cudaMemcpyHostToDevice));
    	gpuErrchk(cudaMemcpy(d_IA, IA, (arrsize+1)*sizeof(int), cudaMemcpyHostToDevice));
    	gpuErrchk(cudaMemcpy(d_JA, JA, nnz*sizeof(int), cudaMemcpyHostToDevice));
    	//gpuErrchk(cudaMemcpy(d_DA, DA, arrsize*sizeof(int), cudaMemcpyHostToDevice));

	//Generate random RHS for testing
	size_t vecSize = sizeof(double) * arrsize;
	double *r = (double *)malloc(vecSize);
	srand( time(NULL) );
    	for(int i = 0; i < arrsize; i++) r[i] = RNG(-DBL_MAX, DBL_MAX);
    	//for(int i = 0; i < arrsize; i++) printf("r[%d] = %f\n", i, r[i]);

/****	Run CPU solver (results used for correctness checks)	****/
	double *x_correct = (double *)malloc(vecSize);
	
	printf("Running single-threaded CPU solver...");
	clock_t startTime = clock();
	for(int i = 0; i < loop; i++) {
		csr_solve_lower_tri_system(AA, IA, JA, DA, arrsize, x_correct, r);
	}
	clock_t stopTime = clock();
	printf("DONE!\n");
	double cpuTime = ((double)stopTime-startTime)/CLOCKS_PER_SEC;
	cpuTime = cpuTime / loop;
/*******************************************************************/

/****	Profile cuSPARSE tri solver ****/
	double *d_r_cusparse;
    	gpuErrchk(cudaMalloc((void**)&d_r_cusparse, vecSize));
    	double *x_cusparse = (double *)malloc(vecSize);
    	double *d_x_cusparse;
    	gpuErrchk(cudaMalloc((void**)&d_x_cusparse, vecSize));
    	//cusparse handle
	cusparseHandle_t handle;
    	cusparseErrchk(cusparseCreate(&handle));
    	
    	printf("Running cusparse tri solver...");
    	startTime = clock();
    	
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
    	cusparseDiagType_t diagtype = CUSPARSE_DIAG_TYPE_UNIT;
    	cusparseErrchk( cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_DIAG_TYPE,
                                              &diagtype, sizeof(diagtype)) );
        //Allocate External buffer for analysis
        void *dBuffer = NULL;
       	 size_t bufferSize = 0;
        double alpha = 1.0;
    	cusparseErrchk( cusparseSpSV_bufferSize(
                                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, vecR, vecX, CUDA_R_64F,
                                CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr,
                                &bufferSize) );
        gpuErrchk( cudaMalloc(&dBuffer, bufferSize) );
        
        gpuErrchk(cudaMemcpy(d_r_cusparse, r, vecSize, cudaMemcpyHostToDevice));
    	gpuErrchk(cudaMemset(d_x_cusparse, 0.0f, vecSize));    	
    	cusparseErrchk( cusparseSpSV_analysis(
                                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, vecR, vecX, CUDA_R_64F,
                                CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr, dBuffer) );	
	for(int i = 0; i < loop; i++) {
        	// execute SpSV
    		cusparseErrchk( cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       &alpha, matA, vecR, vecX, CUDA_R_64F,
                                       CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr) );	
	}
	//Memcpy to Host (results)	
	gpuErrchk(cudaMemcpy(x_cusparse, d_x_cusparse, vecSize, cudaMemcpyDeviceToHost));	
	gpuErrchk(cudaFree(dBuffer));
	stopTime = clock();
	printf("DONE!\n");
	double cusparseTriSolverTime = ((double)stopTime-startTime)/CLOCKS_PER_SEC;
	cusparseTriSolverTime = cusparseTriSolverTime / loop;
	
	//free device result
	gpuErrchk(cudaFree(d_r_cusparse));
	gpuErrchk(cudaFree(d_x_cusparse));
	
	//TEST FOR CORRECTNESS
	bool passed = true;
	for(int i = 0; i < arrsize; i++) {
		if(abs(x_cusparse[i] - x_correct[i]) > eps) {
			passed = false;
			fprintf(stderr, "cuSPARSE_tri_solver failed at cusparse_x[%d] = %f, x[%d] = %f!\n", i, x_cusparse[i], i, x_correct[i]);
		}
	}
	if(passed)
		printf("cuSPARSE_tri_solver PASS\n"); 
/***************************************/

/****	Profile my GPU only tri solver ****/
	printf("Running myGpuSolver_1...\n");
	double *x_mygpu = (double *)malloc(vecSize);
	//TODO: 0 initialize?????
	
    	startTime = clock();
    	
    	//Memcopies 
    	
    	//Convert to CSC for fast col access
    	double *csc_AA = (double *)malloc(nnz * sizeof(double));
    	int *csc_JA = (int *)malloc((arrsize+1) * sizeof(int));
    	int *csc_IA = (int *)malloc(nnz * sizeof(int));	
    	csr2csc(AA, IA, JA, csc_AA, csc_JA, csc_IA, arrsize, nnz);
    	printf("Matrix converted to CSC format!\n");
    	
    	//
    	
	
	stopTime = clock();
	printf("DONE!\n");
	double myGpuSolverTime = ((double)stopTime-startTime)/CLOCKS_PER_SEC;

/******************************************/

	
/****	Output Results for Unit Triangular Solvers	****/
	printf("(Single-threaded) CPU_tri_solver execution time: %fms\n", cpuTime);
	printf("cuSPARSE_tri_solver execution time: %fms\n", cusparseTriSolverTime);
	printf("myGpuSolver_1 execution time: %fms\n", myGpuSolverTime);
/***********************************************************/
	
/****	Free Resources	****/
	//Free host matrix
	free(AA);
	free(IA);
	free(JA);
	free(DA);
	free(csc_AA);
	free(csc_JA);
	free(csc_IA);
	//Free device matrix
	gpuErrchk(cudaFree(d_AA));
	gpuErrchk(cudaFree(d_IA));
	gpuErrchk(cudaFree(d_JA));
	//gpuErrchk(cudaFree(d_DA));
	//Free RHS
	free(r);
	free(x_correct);
	free(x_cusparse);
	//destroy matrix/vector descriptors
    	cusparseErrchk( cusparseDestroySpMat(matA) );
    	cusparseErrchk( cusparseDestroyDnVec(vecR) );
    	cusparseErrchk( cusparseDestroyDnVec(vecX) );
    	cusparseErrchk( cusparseSpSV_destroyDescr(spsvDescr));
    	cusparseErrchk(cusparseDestroy(handle));
/***************************/


	return 0;
}
