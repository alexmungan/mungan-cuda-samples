#ifndef MM2CCRBTRI_H
#define MM2CCRBTRI_H

#ifdef __cplusplus
extern "C" {
#endif

//Function to read in matrix market file using mmio library into special storage format: compressed column row block triangular storage format
void mm2ccrbtri(char *fname);

#ifdef __cplusplus
}
#endif

#endif
