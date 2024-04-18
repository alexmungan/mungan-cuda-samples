#include <stdio.h>
#include <cublas_v2.h>

#define cublasErrchk(ans) cublasAssert((ans), __FILE__, __LINE__)

inline void cublasAssert(cublasStatus_t status, const char *file, int line, bool abort=true)
{
   #ifndef NDEBUG
   if (status != CUBLAS_STATUS_SUCCESS)
   {
      fprintf(stderr, "cuBLAS Error: %d at %s:%d\n", status, file, line);
      if (abort) exit(status);
   }
   #endif
}
