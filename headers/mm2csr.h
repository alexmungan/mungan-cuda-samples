#ifndef MM2CSR_H
#define MM2CSR_H

#ifdef __cplusplus
extern "C" {
#endif

//Function to read in matrix market file using mmio library
/* Pararms:
*  fname: file name of matrix to be read in
*  AA: stores matrix values
*  IA: stores row pointers (indicates where each rows starts and ends in AA array)
*  JA: stores column indices (the column of the original matrix)
*  DA: stores the diagonal values' corresponding indices in AA
*  arrsize: the dimension size n of an nxn square matrix
*  nnz: the total number of non zero elements in the sparse matrix
*/
void mm2csr(char *fname, double AA[], int IA[], int JA[], int DA[], int *arrsize, int *nnz);

#ifdef __cplusplus
}
#endif

#endif
