#include <stdio.h>
#include <cusparse.h>

#define cusparseErrchk(ans) cusparseAssert((ans), __FILE__, __LINE__)

inline void cusparseAssert(cublasStatus_t status, const char *file, int line, bool abort=true)
{
   #ifndef NDEBUG
   if (status != CUSPARSE_STATUS_SUCCESS)
   {
      fprintf(stderr, "cuSPARSE Error: %d at %s:%d\n", cusparseGetErrorString(status), file, line);
      if (abort) exit(status);
   }
   #endif
}
