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
#define loop 1024 //For profiling: number of times to run some operation being profiled in a loop

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

/**** Profile and test CPU vs GPU csr2csc routines ****/
	printf("Testing CPU vs GPU csr2csc routines.\n");
	//CPU ----------------------
	double totalTime = 0;
	
	double *cpu_csc_AA;
    	int *cpu_csc_JA, *cpu_csc_IA, *cpu_colHeadPtrs;
    	
    	for(int i = 0; i < loop; i++) {
    	    	startTime = clock();
    	    	cpu_csc_AA = (double *)malloc(nnz * sizeof(double));
    		cpu_csc_JA = (int *)calloc((arrsize+1), sizeof(int));  //zeroed    		
    		cpu_csc_IA = (int *)malloc(nnz * sizeof(int));	
		cpu_colHeadPtrs = (int *)calloc(arrsize, sizeof(int)); //zeroed
		
    	    	csr2csc(AA, IA, JA, cpu_csc_AA, cpu_csc_JA, cpu_csc_IA, arrsize, nnz, cpu_colHeadPtrs);
    	    	
    	    	stopTime = clock();
    	    	totalTime += (stopTime-startTime);
    	    	
    	    	free(cpu_colHeadPtrs);
    	    	
    	    	//If its the last iteration, keep results so we can use to compare against gpu results
    	    	if(i == (loop-1))
    	    		break;
    	    		
    	    	free(cpu_csc_AA);
		free(cpu_csc_JA);
		free(cpu_csc_IA);
    	}
    	

    	double cpu_csr2csc_time = ((double)totalTime)/CLOCKS_PER_SEC;
    	cpu_csr2csc_time = cpu_csr2csc_time / loop;
    	
    	//GPU ---------------------- NOTE: this gpu routine could possibly be significantly improved through the use of CUDA streams to hide memcpy
    	totalTime = 0;
    	//Gpu CSR
    	double *gpu_AA; 
    	int *gpu_IA, *gpu_JA;   
    	//GPU CSC	
    	double *gpu_csc_AA;
    	int *gpu_csc_JA, *gpu_csc_IA, *gpu_colHeadPtrs;
    	
    	for(int i = 0; i < loop; i++) {
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
    	    	gpuErrchk(cudaMalloc((void**)&gpu_colHeadPtrs, arrsize * sizeof(int)));
    	    	gpuErrchk(cudaMemset(gpu_colHeadPtrs, 0, arrsize*sizeof(int))); //zeroed
    	    	
    	    	/* KERNEL */
		int blockSize = 128;
		dim3 nnzGridSize(((nnz + blockSize - 1)/ blockSize),1,1);
		cu_csr2csc_part1<<<nnzGridSize, blockSize>>>(gpu_JA, gpu_csc_JA, nnz);
		
		int *host_csc_JA = (int *)malloc((arrsize+1)*sizeof(int));
		gpuErrchk(cudaMemcpy(host_csc_JA, gpu_csc_JA, (arrsize+1)*sizeof(int), cudaMemcpyDeviceToHost));
		cu_csr2csc_part2(host_csc_JA, arrsize);
		gpuErrchk(cudaMemcpy(gpu_csc_JA, host_csc_JA, (arrsize+1)*sizeof(int), cudaMemcpyHostToDevice));
		free(host_csc_JA);
		
		dim3 arrsizeGridSize(((arrsize + blockSize - 1)/ blockSize),1,1);
		for(int rowidx = 0; rowidx < arrsize; rowidx++) {
			cu_csr2csc_part3<<<arrsizeGridSize, blockSize>>>(gpu_AA, gpu_IA, gpu_JA, gpu_csc_AA, gpu_csc_JA, gpu_csc_IA, gpu_colHeadPtrs, rowidx);
			//implicit global sync for each kernel call
			//Basic Kernel Error Checking
			gpuErrchk( cudaPeekAtLastError() );
        		gpuErrchk( cudaDeviceSynchronize() );
		}	
    	    	/**********/   	    	
    	    	
    	    	stopTime = clock();
    	    	totalTime += (stopTime-startTime);
    	    	
    	    	gpuErrchk(cudaFree(gpu_AA));
    	    	gpuErrchk(cudaFree(gpu_IA));
    	    	gpuErrchk(cudaFree(gpu_JA));
    	    	gpuErrchk(cudaFree(gpu_colHeadPtrs));
    	    	
    	    	//If its the last iteration, don't free csc mem, so we can check gpu results
    	    	if(i == (loop-1))
    	    		break;
    	    	
    	    	gpuErrchk(cudaFree(gpu_csc_AA));
    	    	gpuErrchk(cudaFree(gpu_csc_JA));
    	    	gpuErrchk(cudaFree(gpu_csc_IA));
    	    			
    	}
    	
    	//GPU Correctness Check -----
    	double *test_csc_AA = (double *)malloc(nnz * sizeof(double));
    	int *test_csc_JA = (int *)malloc((arrsize+1) * sizeof(int));    		
    	int *test_csc_IA = (int *)malloc(nnz * sizeof(int));	
    	gpuErrchk(cudaMemcpy(test_csc_AA, gpu_csc_AA, nnz*sizeof(double), cudaMemcpyDeviceToHost));
    	gpuErrchk(cudaMemcpy(test_csc_JA, gpu_csc_JA, (arrsize+1) * sizeof(int), cudaMemcpyDeviceToHost));
    	gpuErrchk(cudaMemcpy(test_csc_IA, gpu_csc_IA, nnz * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(gpu_csc_AA));
    	gpuErrchk(cudaFree(gpu_csc_JA));
    	gpuErrchk(cudaFree(gpu_csc_IA));
    	
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
    	if(!passed)
    		printf("gpu csr2csc failed!\n");
    	else
    		printf("gpu csc2csc PASS!\n");

    	
    	free(cpu_csc_AA);
	free(cpu_csc_JA);
	free(cpu_csc_IA);
	free(test_csc_AA);
	free(test_csc_JA);
	free(test_csc_IA);
	//----------------------------

    	double gpu_csr2csc_time = ((double)totalTime)/CLOCKS_PER_SEC;
    	gpu_csr2csc_time = gpu_csr2csc_time / loop;
    	printf("CPU csr2csc execution time: %fs\n", cpu_csr2csc_time);
    	printf("GPU csr2csc execution time: %fs\n", gpu_csr2csc_time);

/******************************************************/

/****	Profile my GPU only tri solver ****/
	printf("Running myGpuSolver_1...");
	double *x_mygpu = (double *)malloc(vecSize);
	//TODO: 0 initialize?????
	
    	startTime = clock();
    	
    	//Memcopies 
    	
    	//Convert to CSC for fast col access
    	//double *csc_AA = (double *)malloc(nnz * sizeof(double));
    	//int *csc_JA = (int *)malloc((arrsize+1) * sizeof(int));
    	//int *csc_IA = (int *)malloc(nnz * sizeof(int));	
    	//csr2csc(AA, IA, JA, csc_AA, csc_JA, csc_IA, arrsize, nnz);
    	//printf("Matrix converted to CSC format!\n");
    	
    	//
    	
	
	stopTime = clock();
	printf("DONE!\n");
	double myGpuSolverTime = ((double)stopTime-startTime)/CLOCKS_PER_SEC;
	myGpuSolverTime = myGpuSolverTime / loop;

/******************************************/

	
/****	Output Results for Unit Triangular Solvers	****/
	printf("(Single-threaded) CPU_tri_solver execution time: %fs\n", cpuTime);
	printf("cuSPARSE_tri_solver execution time: %fs\n", cusparseTriSolverTime);
	printf("myGpuSolver_1 execution time: %fs\n", myGpuSolverTime);
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
