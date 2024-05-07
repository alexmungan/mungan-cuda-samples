/* This file implements routines to convert a sparse matrix stored in CSR format to CSC */

/* CPU routine
 * Params:
 * AA, IA, JA: CSR input
 * csc_AA, csc_JA, csc_IA: CSC output, Note: csc_JA should be passed in 0-initialized
 * arrsize: num of cols / rows of the symmetric posdef input
 * nnz: num of non zero elements
 * colHeadPtrs: array of pointers to the current head of each column in AA, Note: should be passed in 0-initialized
 */
 #include <stdio.h>
__host__ void csr2csc(double *AA, int *IA, int *JA, double *csc_AA, int *csc_JA, int *csc_IA, int arrsize, int nnz,
			int *colHeadPtrs) {
	
	//Loop through CSR JA[] to get size of each column
	for(int i = 0; i < nnz; i++) {
		csc_JA[JA[i] + 1]++;
	}
	
	// Cumulative sum to get column pointers
   	 for (int i = 1; i <= arrsize; i++) {
        	csc_JA[i] += csc_JA[i - 1];
    	}
	
	//Populate csc_AA and csc_IA
	for(int rowidx = 0; rowidx < arrsize; rowidx++) {
		int k1 = IA[rowidx];
		int k2 = IA[rowidx+1] - 1;
		for(int k = k1; k <= k2; k++) {
			int j = JA[k];
			int currElement = csc_JA[j] + colHeadPtrs[j];
			if(currElement <= (csc_JA[j+1]-1)) {
				csc_AA[currElement] = AA[k];
				csc_IA[currElement] = rowidx;
				colHeadPtrs[j]++;
			}
		}
	}
	
}

/*GPU routine: broken into 2 parts (implicit global synchronization b/w each kernel call) */	
/* Part 1 */ 		
__global__ void cu_csr2csc_part1(int *d_JA, int *d_csc_JA, int nnz) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	
	//Get size of each column
	if(gid < nnz) {
		atomicAdd(&d_csc_JA[d_JA[gid] + 1], 1); 
	}
	
	//Implicit global sync - CPU calls cu_csr2csc_part2 and then cu_csr2csc_part3
}
/* Part 2 */
__host__ void cu_csr2csc_part2(int *csc_JA, int arrsize) {
	// Cumulative sum to get column pointers
   	 for (int i = 1; i <= arrsize; i++) {
        	csc_JA[i] += csc_JA[i - 1];
    	}
}
/* Part 3 */			
__global__ void cu_csr2csc_part3(double *d_AA, int *d_IA, int *d_JA, double *d_csc_AA, int *d_csc_JA, int *d_csc_IA, int *d_colHeadPtrs, int rowidx) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	//Populate csc_AA and csc_IA
	int k1 = d_IA[rowidx];
	int k2 = d_IA[rowidx+1] - 1;
	int k = k1 + gid;
	if(k <= k2) {
		int j = d_JA[k];
		int currElement = d_csc_JA[j] + d_colHeadPtrs[j];
		if(currElement <= (d_csc_JA[j+1]-1)) {
			d_csc_AA[currElement] = d_AA[k];
			d_csc_IA[currElement] = rowidx;
			d_colHeadPtrs[j]++;	
		}
	}
	//Implicit global sync by multiple kernel calls by CPU	
}
