#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include "global_ccrbtri_mat.h"

//define global variables
struct block *upperBlocks;
struct block *lowerBlocks;
double *diag;

#define TINYNUM 1.e-14

/* This function takes the COO matrix defined by nentry, rowidx[], colidx[], and val[]
*  and converts it into ccbrtri
*  Pararms:
*  nentry: number of entries in COO matrix
*  val: stores the values of COO matrix
*  rowidx: for each value in val, rowidx stores the row of the original matrix that the values corresponds to
*  colidx: for each value in val, colidx stores the col of the original matrix that the values corresponds to
*/
int sparsifymm2ccrbtri(int nentry, int rowidx[], int colidx[], double val[])
{
  int n, i, j, imax, jmax, ndiag, nupper, nlower;

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
   
   int arrsize = 0;
   if (imax != jmax) {
      printf("Error: the matrix is not a square matrix. imax = %d and jmax = %d\n",imax,jmax);
      return 1;
   }
   else {
      arrsize = imax + 1;
   }

   printf("\nconfirm matrix size = %d X %d\n",arrsize,arrsize);
   printf("number of nonzero diagonal elements (ndiag = %d)\n",ndiag);
   if (ndiag < arrsize) {
       printf("Warning: %d diagonal elements are zero.\n",arrsize-ndiag);
       return 1;
   }
   else {
       printf("FYI: all diagonal elements are nonzero.\n");
   }

   int nnz = arrsize; // start with the number of diagonal elements
   int upperLowerFull = 2; //0 indicated only the upper matrix is stored, 1 for lower, 2 for the entire symmetric matrix, 3 for diagonal
   if (nupper == 0 && nlower > 0) {
       printf("FYI: the input matrix stores only entries below the diagonal (nlower = %d)\n",nlower); 
       printf("     therefore, the matrix seems symmetric.\n");
       nnz += (nlower*2);
       upperLowerFull = 1;
   }
   else if (nupper > 0 && nlower == 0) {
       printf("FYI: the input matrix stores only entries above the diagonal (nupper = %d)\n",nupper); 
       printf("     therefore, the matrix seems symmetric.\n");
       nnz += (nupper*2);
       upperLowerFull = 0;
   }
   else if (nupper == 0 && nlower == 0) {
       printf("FYI: this is a diagonal matrix.\n");
             upperLowerFull = 3;
   }
   else {
       printf("number of upper diagonal nonzeros (nupper = %d)\n",nupper);
       printf("number of lower diagonal nonzeros (nlower = %d)\n",nlower);
       nnz += (nupper + nlower);
       upperLowerFull = 2;       
   }
   printf("number of total nonzeros elements (nnz = %d)\n",nnz);

// allocate memory
   diag = malloc(arrsize*sizeof(double));
   int blocksCount = ceil(((double)arrsize)/TRI_SOLVER_BLOCK_SIZE);
   upperBlocks = malloc(blocksCount * sizeof(*upperBlocks));
   lowerBlocks = malloc(blocksCount * sizeof(*lowerBlocks));   

// additional working buffers to perform the analysis
   bool *upper_ready = (bool *)malloc(arrsize * sizeof(*upper_ready)); //says whether the value is ready to be added to the current iteration: 0 is not ready, 1 is ready
   bool *lower_ready = (bool *)malloc(arrsize * sizeof(*lower_ready));
   memset(upper_ready, true, arrsize*sizeof(*upper_ready));
   memset(lower_ready, true, arrsize*sizeof(*lower_ready));
   bool *upper_added = (bool *)malloc(nnz * sizeof(*upper_added));     //says whether the value has already been added to values or not
   bool *lower_added = (bool *)malloc(nnz * sizeof(*upper_added));
   memset(upper_added, false, nnz*sizeof(*upper_added));
   memset(lower_added, false, nnz*sizeof(*lower_added));
   int *upper_rowSizes = (int *)malloc((arrsize+1) * sizeof(int));
   memset(upper_rowSizes, 0, (arrsize+1)*sizeof(*upper_rowSizes));
   int *lower_rowSizes = (int *)malloc((arrsize+1) * sizeof(int));
   memset(lower_rowSizes, 0, (arrsize+1)*sizeof(*lower_rowSizes));
   
   /**** coo matrix holds only upper triangle ****/
   /*if(upperLowerFull == 0) { 
   	// store the number of entries of each row and get diagonal
        for (n = 0; n < nentry; n++) {
       	   i = rowidx[n];
           j = colidx[n];
           if(i == j) { 
               diag[i] = val[n];
           }
           else {
               upper_rowSizes[i]++;
               upper_ready[i] = false;
               lower_rowSizes[j]++;   
               lower_ready[j] = false;  
           }
   	}
   } */
   /**** coo matrix holds only lower triangle *******/
   if (upperLowerFull == 1) { 
     	// store the number of entries of each row and get diagonal
        for (n = 0; n < nentry; n++) {
       	   i = rowidx[n];
           j = colidx[n];
           if(i == j) { 
               diag[i] = val[n];
           }
           else {
               lower_rowSizes[i]++;
               lower_ready[i] = false;
               upper_rowSizes[j]++;   
               upper_ready[j] = false;  
           }
   	}
   	
        for(int i = 0; i < blocksCount; i++) {
        //Initialize / allocate each LOWER block's internal fields (Note: for lower tri solver, cuda blocks map to the matrix's rows from top to bottom)
            //Get range of rows that the block is responsible for
            int startRow = blocksCount * TRI_SOLVER_BLOCK_SIZE;
            int endRow = startRow + TRI_SOLVER_BLOCK_SIZE - 1; 
            //Get total number of elements in the row's that block is responsible for
            int total = 0;
            for(int r = startRow; r <= endRow; r++) {
            	if(r < arrsize) { //Prevent out of bounds access if the last block maps past the matrix (b/c arrsize is not divisble by TRI_SOLVER_BLOCK_SIZE)
            	    total += lower_rowSizes[r]; 
            	} 
            }
            //Finally, allocate values buffer to hold 'total' # of elements
   	    lowerBlocks[i].values = (double *)malloc(total * sizeof(double));
   	    lowerBlocks[i].iterPtrs = (int *)malloc(arrsize * sizeof(int));
   	    lowerBlocks[i].iterPtrs[0] = 0;
   	    lowerBlocks[i].iterCount = 0;
   	    lowerBlocks[i].iterated = false;
   	    lowerBlocks[i].numOfIterations = 0; 
   	    lowerBlocks[i].row = (int *)malloc(nlower * sizeof(int)); 
   	    lowerBlocks[i].col = (int *)malloc(nlower * sizeof(int));
   	//Initialize / allocate each UPPER block's internal fields (Note: for upper tri solver, cuda blocks map to the matrix's rows from bottom to top)
   	    //Get range of rows that the block is responsible for
            startRow = (arrsize - 1) - (blocksCount * TRI_SOLVER_BLOCK_SIZE);
            endRow = startRow - TRI_SOLVER_BLOCK_SIZE + 1; 
            //Get total number of elements in the row's that block is responsible for
            total = 0;
            for(int r = startRow; r >= endRow; r--) {
            	if(r >= 0) { //Prevent out of bounds access if the last block maps past the matrix (b/c arrsize is not divisble by TRI_SOLVER_BLOCK_SIZE)
            	    total += upper_rowSizes[r]; 
            	} 
            }
            //Finally, allocate values buffer to hold 'total' # of elements
   	    upperBlocks[i].values = (double *)malloc(total * sizeof(double));
   	    upperBlocks[i].iterPtrs = (int *)malloc(arrsize * sizeof(int));
   	    upperBlocks[i].iterPtrs[0] = 0;
   	    upperBlocks[i].iterCount = 0;
   	    upperBlocks[i].iterated = false;
   	    upperBlocks[i].numOfIterations = 0; 
   	    upperBlocks[i].row = (int *)malloc(nlower * sizeof(int)); 
   	    upperBlocks[i].col = (int *)malloc(nlower * sizeof(int));
        }
        /*
   	//Get the ccrbri matrix 
   	bool finished = false;
   	int iterno = 0;
   	while(!finished) {
   	    finished = true; 
   	    iterno++;
   	    //Each loop iteration corresponds to an parallel iteration to be stored in each block's storage
   	    for(n = 0; n < nentry; n++) {
   	    	int myval = val[n];
   	    	i = rowidx[n];
           	j = colidx[n];
           	if(i == -1) continue;
           	if(lower_ready[j] && !lower_added[n]) {
           	        finished = false;
           		int blockIdx = floor((double)i / TRI_SOLVER_BLOCK_SIZE);
           		struct block *temp = &lowerBlocks[blockIdx];
           		temp->iterated = true;
           		temp->values[temp->iterCount] = myval;
           		temp->row[temp->iterCount] = i;
           		temp->col[temp->iterCount] = j;
           		temp->iterCount++;
           		lower_added[n] = true;       		
           	}
           	
           	if(upper_ready[i] && !upper_added[n]) {
           	        finished = false;
           	        int blockIdx = blocksCount - 1 - floor((double)j / TRI_SOLVER_BLOCK_SIZE);
           	        struct block *temp = &upperBlocks[blockIdx];
           	        temp->iterated = true;
           	        temp->values[temp->iterCount] = myval;
           	        temp->row[temp->iterCount] = j;
           	        temp->col[temp->iterCount] = i;
           	        temp->iterCount++;
           	        upper_added[n] = true;
           	}
   	    }
   	    
   	    if(finished) 
   	    	break;
   	    
   	  
   	    //Set lower iter array
   	    for(int b = 0; b < blocksCount; b++) {
   	     	struct block *tempLower = &lowerBlocks[b];
   	     	if(tempLower->iterated) {
   	     		tempLower->iterPtrs[iterno] = tempLower->iterCount;
   	    		tempLower->numOfIterations++;
   	    		tempLower->iterated = false;
   	     	}
   	     	
   	     	struct block *tempUpper = &upperBlocks[b];
   	     	if(tempUpper->iterated) {
   	     		tempUpper->iterPtrs[iterno] = tempUpper->iterCount;
   	    		tempUpper->numOfIterations++;
   	    		tempUpper->iterated = false;
   	     	}
            }
   	
   	}
   	*/
   }
   /**** coo matrix holds the entire symmetric matrix (upper and lower) *******/
   /*else if (upperLowerFull == 2) { 
   	// store the number of entries of each row and get diagonal
        for (n = 0; n < nentry; n++) {
       	   i = rowidx[n];
           j = colidx[n];
           if(i == j) { 
               diag[i] = val[n];
           }
           else if (j < i) { //lower
               lower_rowSizes[i]++;
               lower_ready[i] = false;
           }
           else { //upper
               upper_rowSizes[i]++;   
               lower_ready[i] = false;  
           }
   	}
   }*/
   return 0; //success
}
