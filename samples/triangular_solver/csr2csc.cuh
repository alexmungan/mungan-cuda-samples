/* This file implements routines to convert a sparse matrix stored in CSR format to CSC */

__host__ void csr2csc(double *AA, int *IA, int *JA, double *csc_AA, int *csc_JA, int *csc_IA, int arrsize, int nnz);
