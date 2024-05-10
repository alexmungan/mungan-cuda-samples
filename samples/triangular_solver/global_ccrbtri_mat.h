//This file stores the global ccrbtri matrix

#ifndef GLOBAL_CCRBTRI_MAT_H
#define GLOBAL_CCRBTRI_MAT_H

#ifdef __cplusplus
extern "C" {
#endif

#define TRI_SOLVER_BLOCK_SIZE 32

struct block {
	double *values; //values of the AA array
	int *iterPtrs;	//stores indices indicating (in the values arr) where each iteration starts and ends
	int iterCount;
	bool iterated;
	int numOfIterations; //number of iterations needed for this block
	int *row; //the row index of the element
	int *col; //the col index of the element
};

//upperBlocks holds information about matrix needed for upper triangle solver
extern struct block *upperBlocks;
extern struct block *lowerBlocks;

//holds the diagonal values
extern double *diag;

#ifdef __cplusplus
}
#endif

#endif
