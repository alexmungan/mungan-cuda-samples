This folder contains a CUDA implementation of sparse triangular solvers on the gpu (forward / backward substitution).
The diagonal of the matrix is assumed to be unit.

First, the sample test and profiles various triangular solve kernels

Then, triangular solves are optimized for a more specific context. 
The exact computation to be performed is:
solveUpperTriangularSystem(Matrix A, input y, output y);
elementWiseVectorMultiply(input1 diagonalOfMatrixA, input2 y, output w);
vectorAdd(input1 w, input2 v, output w);
solveLowerTriangularSystem(Matrix A, input W, output W);
vectorADd(input1 w, input2 y, output w);

(This particular sequence of operations is a realistic application of tri solves in an iterative solver.)

