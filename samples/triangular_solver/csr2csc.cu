/* This file implements routines to convert a sparse matrix stored in CSR format to CSC */

__host__ void csr2csc(double *AA, int *IA, int *JA, double *csc_AA, int *csc_JA, int *csc_IA, int arrsize, int nnz) {
	memset(csc_JA, 0, (arrsize+1)*sizeof(int));
	
	//Loop through CSR JA[] to get size of each column
	for(int i = 0; i < nnz; i++) {
		csc_JA[JA[i] + 1]++;
	}
	
	//Initialize array of pointers to the current head of each column in AA
	int *colHeadPtrs = (int *)calloc(arrsize, sizeof(int));
	
	//Populate csc_AA and csc_IA
	for(int rowidx = 0; rowidx < arrsize; rowidx++) {
		int k1 = IA[rowidx];
		int k2 = IA[rowidx+1] - 1;
		for(int k = k1; k <= k2; k++) {
			int j = JA[k];
			int currElement = j + colHeadPtrs[j];
			csc_AA[currElement] = AA[k];
			csc_IA[currElement] = rowidx;
			colHeadPtrs[j]++;
		}
	}
	
}
