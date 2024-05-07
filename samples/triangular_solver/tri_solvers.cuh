#ifndef TRI_SOLVERS_CUH
#define TRI_SOLVERS_CUH

/* This file contains triangular solvers called in main.cu */

//CPU SOLVER: used for correctness checks
__host__ void csr_solve_lower_tri_system(double AA[], int IA[], int JA[], int DA[], int nn, double x[], double r[]);


#endif
