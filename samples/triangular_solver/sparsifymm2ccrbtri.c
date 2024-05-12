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
int numOfBlocks;

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
       //printf("(%d,%d) = %f\n", i, j, val[n]);
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
   //printf("BLOCKSCOUNT = %d\n", blocksCount);
   numOfBlocks = blocksCount;
   upperBlocks = malloc(blocksCount * sizeof(struct block));
   lowerBlocks = malloc(blocksCount * sizeof(struct block));   
   if(!upperBlocks || !lowerBlocks) {
   	fprintf(stderr, "Failed memory allocations!\n");
   	exit(EXIT_FAILURE);
   }

// additional working buffers to perform the analysis
   bool *upper_ready = malloc(arrsize * sizeof(bool)); //says whether the value is ready to be added to the current iteration: 0 is not ready, 1 is ready
   bool *lower_ready = malloc(arrsize * sizeof(bool));
   memset(upper_ready, true, arrsize*sizeof(bool));
   memset(lower_ready, true, arrsize*sizeof(bool));
   printf("nLower = %d, nentry = %d\n", nlower, nentry);
   bool *upper_added = malloc(nentry * sizeof(*upper_added));     //says whether the value has already been added to values or not
   bool *lower_added = malloc(nentry * sizeof(*lower_added));
   memset(upper_added, false, nentry*sizeof(*upper_added));
   memset(lower_added, false, nentry*sizeof(*lower_added));
   int *upper_rowSizes = malloc(arrsize * sizeof(int));
   memset(upper_rowSizes, 0, arrsize*sizeof(*upper_rowSizes));
   int *lower_rowSizes = malloc(arrsize * sizeof(int));
   memset(lower_rowSizes, 0, arrsize*sizeof(*lower_rowSizes));
   if(!upper_ready || !lower_ready || !upper_added || !lower_added || !upper_rowSizes || !lower_rowSizes) {
   	fprintf(stderr, "Failed memory allocations!\n");
   	exit(EXIT_FAILURE);
   }
   
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
   	
   	/*for(int i = 0; i < arrsize; i++) {
   		printf("upper_ready[%d] = %d\n", i, upper_ready[i]);
   	}
   	for(int i = 0; i < arrsize; i++) {
   		printf("lower_ready[%d] = %d\n", i, lower_ready[i]);
   	}
   	for(int i = 0; i < arrsize; i++) {
   		printf("diag[%d] = %f\n", i, diag[i]);
   	}
   	for(int i = 0; i < arrsize; i++) {
   		printf("size of (upper) row %d is %d\n", i, upper_rowSizes[i]);
   	}
   	for(int i = 0; i < arrsize; i++) {
   		printf("size of (lower) row %d is %d\n", i, lower_rowSizes[i]);
   	}*/
   	
        for(int i = 0; i < blocksCount; i++) {
        //Initialize / allocate each LOWER block's internal fields (Note: for lower tri solver, cuda blocks map to the matrix's rows from top to bottom)
            //Get range of rows that the block is responsible for
            int startRow = i * TRI_SOLVER_BLOCK_SIZE;
            int endRow = startRow + TRI_SOLVER_BLOCK_SIZE - 1; 
            //printf("blockIDX = %d\n", i);
            //printf("startRow = %d\n", startRow);
            //printf("endRow = %d\n", endRow);
            //Get total number of elements in the row's that block is responsible for
            int total = 0;
            for(int r = startRow; r <= endRow; r++) {
            	if(r < arrsize) { //Prevent out of bounds access if the last block maps past the matrix (b/c arrsize is not divisble by TRI_SOLVER_BLOCK_SIZE)
            	    total += lower_rowSizes[r]; 
            	    //printf("lower_rowSizes[%d] = %d\n", r, lower_rowSizes[r]);
            	} 
            }
            //printf("Total elements for lower block %d is %d\n", i, total);
            //printf("total = %d\n", total);
            //Finally, allocate values buffer to hold 'total' # of elements
            struct block *temp = &lowerBlocks[i];
            temp->startRow = startRow;
            temp->endRow = endRow;
            temp->total = total;
   	    temp->values = malloc(total * sizeof(double));
   	    temp->iterPtrs = malloc((arrsize+1) * sizeof(int));
   	    temp->iterPtrs[0] = 0;
   	    temp->iterCount = 0;
   	    temp->iterated = false;
   	    temp->numOfIterations = 0; 
   	    temp->row = malloc(total * sizeof(int)); 
   	    temp->col = malloc(total * sizeof(int));
   	    if(!temp->values || !temp->iterPtrs || !temp->row || !temp->col) {
   		fprintf(stderr, "Failed memory allocations!\n");
   		exit(EXIT_FAILURE);
   	    }
   	//Initialize / allocate each UPPER block's internal fields (Note: for upper tri solver, cuda blocks map to the matrix's rows from bottom to top)
            //Get total number of elements in the row's that block is responsible for
            total = 0;
            for(int r = startRow; r <= endRow; r++) {
            	if(r < arrsize) { //Prevent out of bounds access if the last block maps past the matrix (b/c arrsize is not divisble by TRI_SOLVER_BLOCK_SIZE)
            	    total += upper_rowSizes[r]; 
            	} 
            }
            //printf("Total elements for upper block %d is %d\n", i, total);
            //Finally, allocate values buffer to hold 'total' # of elements
            struct block *tempUpper = &upperBlocks[i];
            tempUpper->startRow = startRow;
            tempUpper->endRow = endRow;
            tempUpper->total = total;
   	    tempUpper->values = malloc(total * sizeof(double));
   	    tempUpper->iterPtrs = malloc((arrsize+1) * sizeof(int));
   	    tempUpper->iterPtrs[0] = 0;
   	    tempUpper->iterCount = 0;
   	    tempUpper->iterated = false;
   	    tempUpper->numOfIterations = 0; 
   	    tempUpper->row = malloc(total * sizeof(int)); 
   	    tempUpper->col = malloc(total * sizeof(int));
   	    if(!tempUpper->values || !tempUpper->iterPtrs || !tempUpper->row || !tempUpper->col) {
   		fprintf(stderr, "Failed memory allocations!\n");
   		exit(EXIT_FAILURE);
   	    }
        }
        
   	//Get the ccrbri matrix 
   	bool finished = false;
   	int iterno = 0;
   	while(!finished) {
   	    //printf("\nIteration: %d\n",iterno);
   	    finished = true; 
   	    iterno++;
   	    //Each loop iteration corresponds to an parallel iteration to be stored in each block's storage
   	    for(n = 0; n < nentry; n++) {
   	    	double myval = val[n];
   	    	i = rowidx[n];
           	j = colidx[n];
           	//printf("(%d,%d) = %f\n", i, j, myval);
           	if(i == j) continue;
           	//printf("lower_ready[%d] = %d, lower_added[%d] = %d\n", j, lower_ready[j], n, lower_added[n]);
           	if(lower_ready[j] && !lower_added[n]) {
           		//printf("Entered if statement\n");
           	        finished = false;
           		int blockIdx = floor((double)i / TRI_SOLVER_BLOCK_SIZE);
           		//printf("blockIdx = %d\n", blockIdx);
           		struct block *temp = &lowerBlocks[blockIdx];
           		temp->iterated = true;
           		//printf("iterCount = %d, total = %d\n", temp->iterCount, temp->total);
           		if(temp->iterCount >= temp->total) {
           			fprintf(stderr, "Bug: Exceeded num of elements allocated vals/rows/cols. iterCount = %d, total = %d\n", temp->iterCount, temp->total);
           			exit(EXIT_FAILURE);
           		}
           		temp->values[temp->iterCount] = myval;
           		temp->row[temp->iterCount] = i;
           		temp->col[temp->iterCount] = j;
           		temp->iterCount++;
           		lower_added[n] = true;
           		//check to see if we just finished solving for some x that iteration
           		lower_rowSizes[i]--;
           		//printf("lower_rowSizes[%d] = %d\n", i, lower_rowSizes[i]);
           		if(lower_rowSizes[i] == 0) {
           			lower_ready[i] = true; 
           			//printf("lower_ready[%d] = %d\n", i, lower_ready[i]);
           		} else if (lower_rowSizes[i] < 0) {
           			fprintf(stderr, "Error reading in matrix\n");
           			exit(EXIT_FAILURE);
           		}       		
           	}
           	//printf("upper_ready[%d] = %d, upper_added[%d] = %d\n", i, upper_ready[i], n, upper_added[n]);
           	if(upper_ready[i] && !upper_added[n]) {
 			//printf("Entered if statement\n");          	
           	        finished = false;
           	        int blockIdx = floor((double)j / TRI_SOLVER_BLOCK_SIZE);
           	        //printf("blockIdx = %d\n", blockIdx);
           	        struct block *temp = &upperBlocks[blockIdx];
           	        temp->iterated = true;
           	        //printf("iterCount = %d, total = %d\n", temp->iterCount, temp->total);
           		if(temp->iterCount >= temp->total) {
           			fprintf(stderr, "Bug: Exceeded num of elements allocated vals/rows/cols. iterCount = %d, total = %d\n", temp->iterCount, temp->total);
           			exit(EXIT_FAILURE);
           		}
           	        temp->values[temp->iterCount] = myval;
           	        temp->row[temp->iterCount] = j;
           	        temp->col[temp->iterCount] = i;
           	        temp->iterCount++;
           	        upper_added[n] = true;
           	        //check to see if we just finished solving for some x that iteration
           	        upper_rowSizes[j]--;
           	        //printf("upper_rowSizes[%d] = %d\n", j, upper_rowSizes[j]);
           	        if(upper_rowSizes[j] == 0) {
 				//printf("Element ready!\n");	          	        
           	        	upper_ready[j] = true;
           	        	//printf("upper_ready[%d] = %d\n", j, upper_ready[j]);
           	        } else if (upper_rowSizes[j] < 0) {
           			fprintf(stderr, "Error reading in matrix\n");
           			exit(EXIT_FAILURE);
           		}   
           	}
           	
   	    }
   	    
   	    //lower_ready[iterno] = true;
   	    //upper_ready[(arrsize-1) - iterno] = true;
   	    
   	    if(finished) 
   	    	break;
   	    
   	    
   	    //Set lower iter array
   	    for(int b = 0; b < blocksCount; b++) {
   	     	struct block *tempLower = &lowerBlocks[b];
   	     	if(tempLower->iterated) {
   	     		if(iterno >= (arrsize+1)) {
   	     			fprintf(stderr, "Bug: out of bounds access. iterno = %d, arrsize+1 = %d\n", iterno, arrsize+1);
   	     		}
   	     		tempLower->iterPtrs[iterno] = tempLower->iterCount;
   	    		tempLower->numOfIterations++;
   	    		tempLower->iterated = false;
   	     	}
   	     	
   	     	struct block *tempUpper = &upperBlocks[b];
   	     	if(tempUpper->iterated) {
   	     		if(iterno >= (arrsize+1)) {
   	     			fprintf(stderr, "Bug: out of bounds access. iterno = %d, arrsize+1 = %d\n", iterno, arrsize+1);
   	     		}
   	     		tempUpper->iterPtrs[iterno] = tempUpper->iterCount;
   	    		tempUpper->numOfIterations++;
   	    		tempUpper->iterated = false;
   	     	}
            }
            
   	}
   	
   }
   //printf("HERE\n");
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
   
   //free resources
   free(upper_ready);
   //printf("HERE\n");
   free(lower_ready);
   //printf("HERE\n");
   free(upper_added);
   //printf("HERE\n");
   free(lower_added);
   //printf("HERE\n");
   free(upper_rowSizes);
   //printf("HERE\n");
   free(lower_rowSizes); 
   //printf("HERE\n");  
   return 0; //success
}
