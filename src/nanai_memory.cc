#include <nanai_memory.h>

namespace nanai {
  
  void *nai_memory_calloc(size_t count, size_t size) {
    void *r = calloc(count, size);
    if (r != NULL) {
      memset(r, 0, size * count);
    }
    return r;
  }
  
  void nai_memory_free(void *ptr) {
    return free(ptr);
  }
  
  void *nai_memory_malloc(size_t size) {
    void *r = malloc(size);
    if (r != NULL) {
      memset(r, 0, size);
    }
    return r;
  }
  
  void *nai_memory_realloc(void *ptr, size_t size) {
    return realloc(ptr, size);
  }
}