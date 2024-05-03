/* Contains main() which calls the tri solve kernels */
/* Their are multiple implementations of tri solve - this file tests and profiles them */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cublas_v2.h>
#include <cusparse.h>

#include "gpuErrHandler.cuh"
#include "cuBLASErrHandler.cuh"
#include "cuSPARSEErrHandler.cuh"

int main() {
	//Matrix containers
	double *AA;
	int *IA, *JA, *DA;
	int arrsize, nnz;

	//Get the test matrix 
	char *filepath = "../../data/matrices/sparse/posdef/bcsstk13.mtx";
	mm2csr(filepath, AA, IA, JA, DA, &arrsize, &nnz);
	printf("Matrix file read in.\n");

	for(int i = 0; i < 20; i++)
		printf("%f", AA[i]);
	printf("\n");

	//Warm up kernel TODO???
	


	return 0;
}
