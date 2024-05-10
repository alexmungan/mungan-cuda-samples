#ifndef SPARSIFYMM2CCRBTRI_H
#define SPARSIFYMM2CCRBTRI_H

#ifdef __cplusplus
extern "C" {
#endif

/* This function takes the COO matrix defined by nentry, rowidx[], colidx[], and val[]
*  and converts it into CCRBTRI
*  Pararms:
*  nentry: number of entries in COO matrix
*  val: stores the values of COO matrix
*  rowidx: for each value in val, rowidx stores the row of the original matrix that the values corresponds to
*  colidx: for each value in val, colidx stores the col of the original matrix that the values corresponds to
*/
void sparsifymm2ccrbtri(int nentry, int rowidx[], int colidx[], double val[]);

#ifdef __cplusplus
}
#endif

#endif
