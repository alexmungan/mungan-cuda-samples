#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "mm2csr.h"
#include "mm2ccrbtri.h"
#include "global_ccrbtri_mat.h"

#include <cusparse.h>

#include "gpuErrHandler.cuh"
#include "cuBLASErrHandler.cuh"
#include "cuSPARSEErrHandler.cuh"

#include "csr2csc.cuh"
#include "tri_solvers.cuh"

#define eps 1.0e-5
#define loop 1024 //For profiling: number of times to run some operation being profiled in a loop
#define loop_expensive 1 //For operations that take longer, make their profile loop run for less iterations

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
	//cpuTime = cpuTime / loop;
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
	//cusparseTriSolverTime = cusparseTriSolverTime / loop;
	
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
		
	//free mem
	free(x_cusparse);
/***************************************/

/**** Profile and test CPU vs GPU csr2csc routines ****/
	printf("Testing CPU vs GPU csr2csc routines.\n");
	//CPU ----------------------
	double totalTime = 0;
	
	double *cpu_csc_AA;
    	int *cpu_csc_JA, *cpu_csc_IA, *cpu_csc_DA, *cpu_colHeadPtrs, *nnzPerColAbove, *nnzPerColBelow;
    	 
    	//GPU CSC (just for profiling memcpy)
    	double *gpu_csc_AA;
    	int *gpu_csc_JA, *gpu_csc_IA, *gpu_csc_DA;
    	
    	//NOTE: the larger loop_expensive gets, the longer the average iteration takes for some reason... probably something to do w/ memory allocation
    	for(int i = 0; i < loop_expensive; i++) {
    	    	startTime = clock();
    	    	cpu_csc_AA = (double *)malloc(nnz * sizeof(double));
    		cpu_csc_JA = (int *)calloc((arrsize+1), sizeof(int));  //zeroed    		
    		cpu_csc_IA = (int *)malloc(nnz * sizeof(int));	
    		cpu_csc_DA = (int *)malloc(arrsize * sizeof(int));
		cpu_colHeadPtrs = (int *)calloc(arrsize, sizeof(int)); //zeroed
		nnzPerColAbove = (int *)calloc(arrsize, sizeof(int)); //zeroed
		nnzPerColBelow = (int *)calloc(arrsize, sizeof(int)); //zeroed
		
    	    	csr2csc(AA, IA, JA, cpu_csc_AA, cpu_csc_JA, cpu_csc_IA, cpu_csc_DA, nnzPerColAbove, nnzPerColBelow, arrsize, nnz, cpu_colHeadPtrs);
    	    	
    	    	//Include memcpy time to gpu b/c in real application we would still need to copy to gpu after converting on cpu
		gpuErrchk(cudaMalloc((void**)&gpu_csc_AA, nnz*sizeof(double)));
		gpuErrchk(cudaMalloc((void**)&gpu_csc_JA, (arrsize+1)*sizeof(int)));   	 	
    	    	gpuErrchk(cudaMalloc((void**)&gpu_csc_IA, nnz*sizeof(int)));
    	    	gpuErrchk(cudaMalloc((void**)&gpu_csc_DA, arrsize * sizeof(int)));
    		gpuErrchk(cudaMemcpy(gpu_csc_AA, cpu_csc_AA, nnz*sizeof(double), cudaMemcpyHostToDevice));
    		gpuErrchk(cudaMemcpy(gpu_csc_JA, cpu_csc_JA, (arrsize+1)*sizeof(int), cudaMemcpyHostToDevice));
    		gpuErrchk(cudaMemcpy(gpu_csc_IA, cpu_csc_IA, nnz*sizeof(int), cudaMemcpyHostToDevice));
    		gpuErrchk(cudaMemcpy(gpu_csc_DA, cpu_csc_DA, arrsize*sizeof(int), cudaMemcpyHostToDevice));
    	    	
    	    	stopTime = clock();
    	    	totalTime += (stopTime-startTime);
    	    	
    	    	free(cpu_colHeadPtrs);
    	    	gpuErrchk(cudaFree(gpu_csc_AA));
    	    	gpuErrchk(cudaFree(gpu_csc_JA));
    	    	gpuErrchk(cudaFree(gpu_csc_IA));
    	    	gpuErrchk(cudaFree(gpu_csc_DA));
    	    	
    	    	//If its the last iteration, keep results so we can use to compare against gpu results
    	    	if(i == (loop_expensive-1))
    	    		break;
    	    		
    	    	free(cpu_csc_AA);
		free(cpu_csc_JA);
		free(cpu_csc_IA);
		free(cpu_csc_DA);
		free(nnzPerColAbove);
		free(nnzPerColBelow);
    	}
    	

    	double cpu_csr2csc_time = ((double)totalTime)/CLOCKS_PER_SEC;
    	cpu_csr2csc_time = cpu_csr2csc_time / loop;
    	
    	//GPU ---------------------- NOTE: this gpu routine could possibly be significantly improved through the use of CUDA streams to hide memcpy
    	totalTime = 0;
    	//Gpu CSR
    	double *gpu_AA; 
    	int *gpu_IA, *gpu_JA;   
    	//GPU CSC
    	int *gpu_colHeadPtrs, *gpu_nnzPerColAbove, *gpu_nnzPerColBelow;
    	//CPU control variables
    	int *test_nnzPerColAbove, *test_nnzPerColBelow;
    	
    	//NOTE: the larger loop_expensive gets, the longer the average iteration takes for some reason... probably something to do w/ memory allocation
    	for(int i = 0; i < loop_expensive; i++) {
    	    	startTime = clock();
    	    	//GPU CSR 
    	    	gpuErrchk(cudaMalloc((void**)&gpu_AA, nnz*sizeof(double)));
    	    	gpuErrchk(cudaMalloc((void**)&gpu_IA, (arrsize+1)*sizeof(int)));
    	    	gpuErrchk(cudaMalloc((void**)&gpu_JA, nnz*sizeof(int)));   	 	
    		gpuErrchk(cudaMemcpy(gpu_AA, AA, nnz*sizeof(double), cudaMemcpyHostToDevice));
    		gpuErrchk(cudaMemcpy(gpu_IA, IA, (arrsize+1)*sizeof(int), cudaMemcpyHostToDevice));
    		gpuErrchk(cudaMemcpy(gpu_JA, JA, nnz*sizeof(int), cudaMemcpyHostToDevice));
     	
    	    	//GPU CSC containers
    	    	gpuErrchk(cudaMalloc((void**)&gpu_csc_AA, nnz*sizeof(double)));
    	    	gpuErrchk(cudaMalloc((void**)&gpu_csc_JA, (arrsize+1)*sizeof(int)));
    	    	gpuErrchk(cudaMemset(gpu_csc_JA, 0, (arrsize+1)*sizeof(int)));  //zeroed
    	    	gpuErrchk(cudaMalloc((void**)&gpu_csc_IA, nnz*sizeof(int)));
    	    	gpuErrchk(cudaMalloc((void**)&gpu_csc_DA, arrsize * sizeof(int)));
    	    	gpuErrchk(cudaMalloc((void**)&gpu_colHeadPtrs, arrsize * sizeof(int)));
    	    	gpuErrchk(cudaMemset(gpu_colHeadPtrs, 0, arrsize*sizeof(int))); //zeroed
    	    	gpuErrchk(cudaMalloc((void**)&gpu_nnzPerColAbove, arrsize * sizeof(int)));
    	    	gpuErrchk(cudaMemset(gpu_nnzPerColAbove, 0, arrsize*sizeof(int))); //zeroed
    	    	gpuErrchk(cudaMalloc((void**)&gpu_nnzPerColBelow, arrsize * sizeof(int)));
    	    	gpuErrchk(cudaMemset(gpu_nnzPerColBelow, 0, arrsize*sizeof(int))); //zeroed
    	    	
    	    	/* KERNEL */
		int blockSize = 128;
		int nnzGridSize = (nnz + blockSize - 1)/ blockSize;
		cu_csr2csc_part1<<<nnzGridSize, blockSize>>>(gpu_JA, gpu_csc_JA, nnz);
		
		int *host_csc_JA = (int *)malloc((arrsize+1)*sizeof(int));
		gpuErrchk(cudaMemcpy(host_csc_JA, gpu_csc_JA, (arrsize+1)*sizeof(int), cudaMemcpyDeviceToHost));
		cu_csr2csc_part2(host_csc_JA, arrsize);
		gpuErrchk(cudaMemcpy(gpu_csc_JA, host_csc_JA, (arrsize+1)*sizeof(int), cudaMemcpyHostToDevice));
		free(host_csc_JA);
		
		int nnzRow, nnzRowGridSize; 
		for(int rowidx = 0; rowidx < arrsize; rowidx++) {
			nnzRow = IA[rowidx+1] - IA[rowidx];
			nnzRowGridSize = (nnzRow + blockSize - 1) / blockSize;
			cu_csr2csc_part3<<<nnzRowGridSize, blockSize>>>(gpu_AA, gpu_IA, gpu_JA, gpu_csc_AA, gpu_csc_JA, gpu_csc_IA, gpu_csc_DA, gpu_colHeadPtrs, gpu_nnzPerColAbove, gpu_nnzPerColBelow, rowidx);
			//implicit global sync for each kernel call
			//Basic Kernel Error Checking
			gpuErrchk( cudaPeekAtLastError() );
        		gpuErrchk( cudaDeviceSynchronize() );
		}	
    	    	/**********/   	   
    	    	
    	    	//Realistically, nnzPerCol arrays are needed on the CPU
    	    	test_nnzPerColAbove = (int *)malloc(arrsize * sizeof(int)); //zeroed
		test_nnzPerColBelow = (int *)malloc(arrsize * sizeof(int)); //zeroed
		gpuErrchk(cudaMemcpy(test_nnzPerColAbove, gpu_nnzPerColAbove, arrsize*sizeof(int), cudaMemcpyDeviceToHost));
     		gpuErrchk(cudaMemcpy(test_nnzPerColBelow, gpu_nnzPerColBelow, arrsize*sizeof(int), cudaMemcpyDeviceToHost));
     		gpuErrchk(cudaFree(gpu_nnzPerColAbove));
    	    	gpuErrchk(cudaFree(gpu_nnzPerColBelow));
    	    	
    	    	stopTime = clock();
    	    	totalTime += (stopTime-startTime);
    	    	
    	    	gpuErrchk(cudaFree(gpu_AA));
    	    	gpuErrchk(cudaFree(gpu_IA));
    	    	gpuErrchk(cudaFree(gpu_JA));
    	    	gpuErrchk(cudaFree(gpu_colHeadPtrs));
    	    	
    	    	//If its the last iteration, don't free csc mem, so we can check gpu results
    	    	if(i == (loop_expensive-1))
    	    		break;
    	    	
    	    	gpuErrchk(cudaFree(gpu_csc_AA));
    	    	gpuErrchk(cudaFree(gpu_csc_JA));
    	    	gpuErrchk(cudaFree(gpu_csc_IA));
    	    	gpuErrchk(cudaFree(gpu_csc_DA));
    	}
    	
    	//GPU Correctness Check -----
    	double *test_csc_AA = (double *)malloc(nnz * sizeof(double));
    	int *test_csc_JA = (int *)malloc((arrsize+1) * sizeof(int));    		
    	int *test_csc_IA = (int *)malloc(nnz * sizeof(int));	
    	int *test_csc_DA = (int *)malloc(arrsize * sizeof(int));
    	gpuErrchk(cudaMemcpy(test_csc_AA, gpu_csc_AA, nnz*sizeof(double), cudaMemcpyDeviceToHost));
    	gpuErrchk(cudaMemcpy(test_csc_JA, gpu_csc_JA, (arrsize+1) * sizeof(int), cudaMemcpyDeviceToHost));
    	gpuErrchk(cudaMemcpy(test_csc_IA, gpu_csc_IA, nnz * sizeof(int), cudaMemcpyDeviceToHost));
    	gpuErrchk(cudaMemcpy(test_csc_DA, gpu_csc_DA, arrsize*sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(gpu_csc_AA));
    	gpuErrchk(cudaFree(gpu_csc_JA));
    	gpuErrchk(cudaFree(gpu_csc_IA));
    	gpuErrchk(cudaFree(gpu_csc_DA));
    	
    	//check
    	passed = true;
    	for(int i = 0; i < nnz; i++) {
    		if(abs(test_csc_AA[i] - cpu_csc_AA[i]) > eps) {
    			fprintf(stderr, "csr2csc test failed! CSC values arrays do not match --> \
    			CPU_AA[%d] = %f. GPU_AA[%d] = %f\n", i, cpu_csc_AA[i], i, test_csc_AA[i]);
    			passed = false;
    		}
    		if(abs(test_csc_IA[i] - cpu_csc_IA[i]) > eps) {
    			fprintf(stderr, "csr2csc test failed! CSC rowIndicies arrays do not match --> \
    			CPU_IA[%d] = %d. GPU_IA[%d] = %d\n", i, cpu_csc_IA[i], i, test_csc_IA[i]);
    			    			passed = false;
    		}
    	}
    	for(int i = 0; i < arrsize+1; i++) {
    		if(abs(test_csc_JA[i] - cpu_csc_JA[i]) > eps) {
  	    		fprintf(stderr, "csr2csc test failed! CSC colPtrs arrays do not match --> \
    			CPU_AA[%d] = %d. GPU_AA[%d] = %d\n", i, cpu_csc_JA[i], i, test_csc_JA[i]);
    			    			passed = false;
    		}
    	}
    	for(int i = 0; i < arrsize; i++) {
    		if(abs(test_csc_DA[i] - cpu_csc_DA[i]) > eps) {
  	    		fprintf(stderr, "csr2csc test failed! CSC diagonal indices arrays do not match --> \
    			CPU_DA[%d] = %d. GPU_DA[%d] = %d\n", i, cpu_csc_DA[i], i, test_csc_DA[i]);
    			    			passed = false;
    		}
    		if(abs(test_nnzPerColAbove[i] - nnzPerColAbove[i]) > eps) {
  	    		fprintf(stderr, "csr2csc test failed! nnzPerColAbove arrays do not match --> \
    			CPU[%d] = %d. GPU[%d] = %d\n", i, nnzPerColAbove[i], i, test_nnzPerColAbove[i]);
    			    			passed = false;
    		}
    		if(abs(test_nnzPerColBelow[i] - nnzPerColBelow[i]) > eps) {
  	    		fprintf(stderr, "csr2csc test failed! nnzPerColAbove arrays do not match --> \
    			CPU[%d] = %d. GPU[%d] = %d\n", i, nnzPerColBelow[i], i, test_nnzPerColBelow[i]);
    			    			passed = false;
    		}
    	}
    	if(!passed)
    		printf("gpu csr2csc failed!\n");
    	else
    		printf("gpu csc2csc PASS!\n");

    	
    	free(cpu_csc_AA);
	free(cpu_csc_JA);
	free(cpu_csc_IA);
	free(cpu_csc_DA);
	free(nnzPerColAbove);
	free(nnzPerColBelow);
	free(test_csc_AA);
	free(test_csc_JA);
	free(test_csc_IA);
	free(test_csc_DA);
	free(test_nnzPerColAbove);
	free(test_nnzPerColBelow);
	//----------------------------

    	double gpu_csr2csc_time = ((double)totalTime)/CLOCKS_PER_SEC;
    	gpu_csr2csc_time = gpu_csr2csc_time / loop;
    	printf("CPU csr2csc execution time: %fs\n", cpu_csr2csc_time);
    	printf("GPU csr2csc execution time: %fs\n", gpu_csr2csc_time);

/******************************************************/

/****	Profile my first unoptimized GPU tri solver ****/
	/*printf("Running myGpuLowerSolver_1...");
	double *d_x;
	gpuErrchk(cudaMalloc((void**)&d_x, vecSize));
	gpuErrchk(cudaMemcpy(d_x, r, vecSize, cudaMemcpyHostToDevice));
	
    	startTime = clock();
    	
    	//Convert to CSC for fast col access
    	double *h_csc_AA = (double *)malloc(nnz * sizeof(double));
    	int *h_csc_JA = (int *)calloc((arrsize+1), sizeof(int)); //zeroed
    	int *h_csc_IA = (int *)malloc(nnz * sizeof(int));	
    	int *h_csc_DA = (int *)malloc(arrsize * sizeof(int));
    	int *h_colHeadPtrs = (int *)calloc(arrsize, sizeof(int));//zeroed
    	int *h_nnzPerColAbove =  (int *)calloc(arrsize, sizeof(int));//zeroed
    	int *h_nnzPerColBelow =(int *)calloc(arrsize, sizeof(int));//zeroed
    	csr2csc(AA, IA, JA, h_csc_AA, h_csc_JA, h_csc_IA, h_csc_DA, h_nnzPerColAbove, h_nnzPerColBelow, arrsize, nnz, h_colHeadPtrs);
    	//Free mem not needed
    	free(h_colHeadPtrs); 
    	free(h_nnzPerColAbove);
    		
    	//Copy the CSC matrix to GPU
    	double *d_csc_AA;
    	int *d_csc_JA, *d_csc_IA, *d_csc_DA;
    	gpuErrchk(cudaMalloc((void**)&d_csc_AA, nnz * sizeof(double)));
    	gpuErrchk(cudaMalloc((void**)&d_csc_JA, (arrsize+1) * sizeof(int)));
    	gpuErrchk(cudaMalloc((void**)&d_csc_IA, nnz * sizeof(int)));
    	gpuErrchk(cudaMalloc((void**)&d_csc_DA, arrsize * sizeof(int)));
    	gpuErrchk(cudaMemcpy(d_csc_AA, h_csc_AA, nnz * sizeof(double), cudaMemcpyHostToDevice));
    	gpuErrchk(cudaMemcpy(d_csc_JA, h_csc_JA, (arrsize+1) * sizeof(int), cudaMemcpyHostToDevice));
    	gpuErrchk(cudaMemcpy(d_csc_IA, h_csc_IA, nnz * sizeof(int), cudaMemcpyHostToDevice));
    	gpuErrchk(cudaMemcpy(d_csc_DA, h_csc_DA, arrsize * sizeof(int), cudaMemcpyHostToDevice));
    	*/
    	/*** Run the triangular solver ***/
    	//Get numOfBlocks for each iteration
    	/*int blockSize = 128;
    	int *numOfBlocksPerCol = (int *)malloc(arrsize * sizeof(int));
    	get_numOfBlocksPerCol(numOfBlocksPerCol, h_nnzPerColBelow, blockSize, arrsize);
    	free(h_nnzPerColBelow);
    	//Call the solver
    	for(int i = 0; i < loop; i++) {
    	    	myGpuLowerTriSolver_1(d_csc_AA, d_csc_JA, d_csc_IA, d_csc_DA, numOfBlocksPerCol, arrsize, blockSize, d_x);
    	}*/
    	/*********************************/
	
	/*stopTime = clock();
	printf("DONE!\n");
	double myGpuSolver1Time = ((double)stopTime-startTime)/CLOCKS_PER_SEC;
	
	//Copy the results back
	double *x_myGpuSolver_1 = (double *)malloc(vecSize);
	gpuErrchk(cudaMemcpy(x_myGpuSolver_1, d_x, vecSize, cudaMemcpyDeviceToHost));
	
	//TEST FOR CORRECTNESS
	passed = true;
	for(int i = 0; i < arrsize; i++) {
		if(abs(x_myGpuSolver_1[i] - x_correct[i]) > eps) {
			passed = false;
			fprintf(stderr, "myGpuLowerSolver_1 failed at d_x[%d] = %f, x[%d] = %f!\n", i, x_myGpuSolver_1[i], i, x_correct[i]);
		}
	}
	if(passed)
		printf("myGpuLowerSolver PASS\n"); 
		
	//free device mem
	gpuErrchk(cudaFree(d_csc_AA));
	gpuErrchk(cudaFree(d_csc_JA));
	gpuErrchk(cudaFree(d_csc_IA));
	gpuErrchk(cudaFree(d_csc_DA));
	gpuErrchk(cudaFree(d_x));
	
	//free host mem
	free(h_csc_AA);
	free(h_csc_JA);
	free(h_csc_IA);
	free(h_csc_DA);
	free(x_myGpuSolver_1);
	*/
/******************************************/

	
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
	//destroy matrix/vector descriptors
    	cusparseErrchk( cusparseDestroySpMat(matA) );
    	cusparseErrchk( cusparseDestroyDnVec(vecR) );
    	cusparseErrchk( cusparseDestroyDnVec(vecX) );
    	cusparseErrchk( cusparseSpSV_destroyDescr(spsvDescr));
    	cusparseErrchk(cusparseDestroy(handle));
/***************************/

/****	Profile my Optimized GPU tri solver ****/	
	mm2ccrbtri(filepath);
	printf("Matrix file read in (ccrbtri).\n");
	
/******************************************/

/****	Output Results for Unit Triangular Solvers	****/
	printf("\nRESULTS: average timing for %d executions.\n", loop);
	printf("(Single-threaded) CPU_tri_solver execution time: %fs\n", cpuTime/loop);
	printf("cuSPARSE_tri_solver execution time: %fs\n", cusparseTriSolverTime/loop);
	//printf("myGpuLowerSolver_1 execution time: %fs\n", myGpuSolver1Time/loop);
	printf("\nRESULTS: total times for %d executions.\n", loop);
	printf("(Single-threaded) CPU_tri_solver execution time: %fs\n", cpuTime);
	printf("cuSPARSE_tri_solver execution time: %fs\n", cusparseTriSolverTime);
	//printf("myGpuLowerSolver_1 execution time: %fs\n", myGpuSolver1Time);
/***********************************************************/

/****	Free Resources	****/
	//Free Vectors
	free(r);
	free(x_correct);
/***************************/
	return 0;
}
