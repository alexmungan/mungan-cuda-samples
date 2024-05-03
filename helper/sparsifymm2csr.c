#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TINYNUM 1.e-14

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
void sparsifymm2csr(int nentry, int rowidx[], int colidx[], double val[], double **AA_ptr, 
			int **IA_ptr, int **JA_ptr, int **DA_ptr, int *arrsize_ret, int *nnz_ret)
{
// Inputs:
//    nentry: length of rowidx, colidx and val 
//    rowidx[n]: row index (0-based)
//    colidx[n]: row index (0-based)
//    val[n]: matrix values at each (rowidx, colidx) pair
   int n, i, j, imax, jmax, ndiag, nupper, nlower;
   int arrsize, nnz;

// count the number of nonzero entries
   imax = 0;
   jmax = 0;
   ndiag = 0;
   nupper = 0;
   nlower = 0;
   for (n = 0; n < nentry; n++) {
       i = rowidx[n]; // row index
       j = colidx[n]; // col index
       if (i > imax) imax = i;
       if (j > jmax) jmax = j;

       if (i == j) { // diagonal
          if (fabs(val[n]) > TINYNUM) {
             ndiag ++;
          }
       }
       else if (i < j) { // upper 
          nupper ++;
       }
       else { // lower
          nlower ++;
       }
   }

   if (imax != jmax) {
      printf("Error: the matrix is not a square matrix. imax = %d and jmax = %d\n",imax,jmax);
      return;
   }
   else {
      arrsize = imax + 1;
   }

   printf("\nconfirm matrix size = %d X %d\n",arrsize,arrsize);
   printf("number of nonzero diagonal elements (ndiag = %d)\n",ndiag);
   if (ndiag < arrsize) {
       printf("Warning: %d diagonal elements are zero.\n",arrsize-ndiag);
       return;
   }
   else {
       printf("FYI: all diagonal elements are nonzero.\n");
   }

   nnz = arrsize; // start with the number of diagonal elements
   if (nupper == 0 && nlower > 0) {
       printf("FYI: the input matrix stores only entries below the diagonal (nlower = %d)\n",nlower); 
       printf("     therefore, the matrix seems symmetric.\n");
       nnz += (nlower*2);
   }
   else if (nupper > 0 && nlower == 0) {
       printf("FYI: the input matrix stores only entries above the diagonal (nupper = %d)\n",nupper); 
       printf("     therefore, the matrix seems symmetric.\n");
       nnz += (nupper*2);
   }
   else if (nupper == 0 && nlower == 0) {
       printf("FYI: this is a diagonal matrix.\n");
   }
   else {
       printf("number of upper diagonal nonzeros (nupper = %d)\n",nupper);
       printf("number of lower diagonal nonzeros (nlower = %d)\n",nlower);
       nnz += (nupper + nlower);
   }
   printf("number of total nonzeros elements (nnz = %d)\n",nnz);

// allocate memory
   *AA_ptr = malloc(nnz*sizeof(double));
   *IA_ptr = malloc((arrsize+1)*sizeof(int));
   *DA_ptr = malloc(arrsize*sizeof(int));
   *JA_ptr = malloc(nnz*sizeof(int));
// Rename pointers	
   double *AA;
   int *IA, *JA, *DA;
   AA = *AA_ptr;
   IA = *IA_ptr;
   JA = *JA_ptr;
   DA = *DA_ptr;
   
   if(!AA || !IA || !DA || !JA) {
   	fprintf(stderr, "Could not allocate memory");
   	exit(EXIT_FAILURE);
   }

   // let IA temporarily store the number of entry each row
   memset(IA, 0, (arrsize+1)*sizeof(*IA));
   for (n = 0; n < nentry; n++) {
       i = rowidx[n]; // row index
       j = colidx[n]; // col index
       if (i == j) { // diagonal
          IA[i] ++; // count only once
       }
       else {
         IA[i] ++; 
         IA[j] ++; // count twice due to symmetry
       }
   }

   int k, k1, k2;

   // assemble the IA matrix to its final CSR form
   k1 = IA[0];
   IA[0] = 0;
   for (i = 1; i < arrsize; i++) {
       k2 = IA[i];
       IA[i] = IA[i-1] + k1;  // IA stores the leading index in AA for each row
       k1 = k2;
   }
   IA[arrsize] = IA[arrsize-1] + k1;

   for (k = 0; k < nnz; k++) JA[k] = -1;

   double myval;
   for (n = 0; n < nentry; n++) {
       i = rowidx[n]; // row index
       j = colidx[n]; // col index
       myval = val[n];
       k1 = IA[i];               
       k2 = IA[i+1] - 1;                
       for (k = k1; k <= k2; k++) {
           if (JA[k] == -1) {
              JA[k] = j; // col index
              AA[k] = myval;
              break;
           }
       }
       if (i == j) DA[i] = k; // diagonal
       else { // nondiagonal
          k1 = IA[j];               
          k2 = IA[j+1] - 1;                
          for (k = k1; k <= k2; k++) {
              if (JA[k] == -1) {
                 JA[k] = i; // col index
                 AA[k] = myval;
                 break;
              }
          }
       }
   }

/*
	for(int i = 0; i < 20; i++) {
		printf("%f\n", AA[i]);
	}
*/
   *arrsize_ret = arrsize;
   *nnz_ret = nnz;

}
