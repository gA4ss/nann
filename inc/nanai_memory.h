#ifndef nanai_memory_h
#define nanai_memory_h

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace nanai {

  void *nai_memory_calloc(size_t count, size_t size);
  void nai_memory_free(void *ptr);
  void *nai_memory_malloc(size_t size);
  void *nai_memory_realloc(void *ptr, size_t size);
  
}


#endif /* nanai_memory_h */
