#ifndef CSR2CSC_CUH
#define CSR2CSC_CUH

/* This file implements routines to convert a sparse matrix stored in CSR format to CSC */

/* CPU routine
 * Params:
 * AA, IA, JA: CSR input
 * csc_AA, csc_JA, csc_IA: CSC output, Note: csc_JA should be passed in 0-initialized
 * arrsize: num of cols / rows of the symmetric posdef input
 * nnz: num of non zero elements
 * colHeadPtrs: array of pointers to the current head of each column in AA, Note: should be passed in 0-initialized
 * nnzPerColAbove/Below: indicates how many non zero elements there are above or below the diagonal in that column
 */
__host__ void csr2csc(double *AA, int *IA, int *JA, double *csc_AA, int *csc_JA, int *csc_IA, int *csc_DA, int *nnzPerColAbove, int *nnzPerColBelow, int arrsize, int nnz, int *colHeadPtrs);

/*GPU routine: broken into 2 parts (implicit global synchronization b/w each kernel call) */	
/* Part 1 */ 		
__global__ void cu_csr2csc_part1(int *d_JA, int *d_csc_JA, int nnz);
/* Part 2 */
__host__ void cu_csr2csc_part2(int *csc_JA, int arrsize);
/* Part 3 */			
__global__ void cu_csr2csc_part3(double *d_AA, int *d_IA, int *d_JA, double *d_csc_AA, int *d_csc_JA, int *d_csc_IA, int *d_csc_DA, int *d_colHeadPtrs, int *nnzPerColAbove, int *nnzPerColBelow, int rowidx);


#endif
