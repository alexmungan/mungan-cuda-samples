/* This file reads in the matrix market file into memory in coordinate list format */
/* Then function sparsifymm2csr() is called to convert into csr format */

#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"
#include "sparsifymm2csr.h"

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
void mm2csr(char *fname, double **AA_ptr, int **IA_ptr, int **JA_ptr, int **DA_ptr, int *arrsize, int *nnz)
{
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nentry;   
    int i, *I, *J;
    double *val;

    if ((f = fopen(fname, "r")) == NULL) {
       printf("Error: cannot open file %s.\n",fname);
       exit(1);
    }

    if (mm_read_banner(f, &matcode) != 0) 
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
            mm_is_sparse(matcode) )
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* find out size of sparse matrix .... */

    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nentry)) !=0)
        exit(1);

    printf("FYI: number of row = %d\n", M);
    printf("FYI: number of col = %d\n", N);
    printf("FYI: number of entries in the .mtx file = %d\n", nentry);

    /* reseve memory for matrices */

    I = (int *) malloc(nentry * sizeof(int));
    J = (int *) malloc(nentry * sizeof(int));
    val = (double *) malloc(nentry * sizeof(double));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i=0; i<nentry; i++)
    {
        fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }

    if (f !=stdin) fclose(f);

    /*************************/
    /* convert to CSR format */
    /*************************/

    sparsifymm2csr(nentry, I, J, val, AA_ptr, IA_ptr, JA_ptr, DA_ptr, arrsize, nnz);

    /************************/
    /* now write out matrix */
    /************************/

//    mm_write_banner(stdout, matcode);
//    mm_write_mtx_crd_size(stdout, M, N, nentry);
//    for (i=0; i<nentry; i++)
//        fprintf(stdout, "%d %d %20.19g\n", I[i]+1, J[i]+1, val[i]);

    free(I); I = NULL;
    free(J); J = NULL;
    free(val); val = NULL;

}
