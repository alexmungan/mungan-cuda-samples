/* This file contains triangular solvers called in main.cu */

//CPU SOLVER: used for correctness checks
__host__ void csr_solve_lower_tri_system(double AA[], int IA[], int JA[], int DA[], int nn, double x[], double r[])
{
// Purpose: compute (I + Low_tri(A))*x, return modified x.
// x[] input as the rhs vector, output as the solution.

    int idx,k1,k2,k,j;

    for (idx = 0; idx < nn; idx++) {   // compressed sparse row format
        k1 = IA[idx];
        k2 = DA[idx] - 1;
        x[idx] = r[idx];
        for (k = k1; k <= k2; k++) {
            j = JA[k];
            x[idx] -= AA[k]*x[j];
        }
    }
}
