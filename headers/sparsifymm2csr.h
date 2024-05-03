#ifndef SPARSIFYMM2CSR_H
#define SPARSIFYMM2CSR_H

#ifdef __cplusplus
extern "C" {
#endif

/* This function takes the COO matrix defined by nentry, rowidx[], colidx[], and val[]
*  and converts it into CSR using the provided storage: AA[], IA[], JA[]
*  Pararms:
*  nentry: number of entries in COO matrix
*  val: stores the values of COO matrix
*  rowidx: for each value in val, rowidx stores the row of the original matrix that the values corresponds to
*  colidx: for each value in val, colidx stores the col of the original matrix that the values corresponds to
*  AA: stores matrix values
*  IA: stores row pointers (indicates where each rows starts and ends in AA array)
*  JA: stores column indices (the column of the original matrix)
*  DA: stores the diagonal values' corresponding indices in AA
*  arrsize: the dimension size n of an nxn square matrix
*  nnz: the total number of non zero elements in the sparse matrix
*/
void sparsifymm2csr(int nentry, int rowidx[], int colidx[], double val[], double **AA, 
			int **IA, int **JA, int **DA, int *arrsize_ret, int *nnz_ret);

#ifdef __cplusplus
}
#endif

#endif
