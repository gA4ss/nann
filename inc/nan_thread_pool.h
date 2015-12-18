#ifndef nan_thread_pool_h
#define nan_thread_pool_h

#include <pthread.h>
#include <nan_object.h>

namespace nanan {
  class nan_thread_pool : public nan_object {
  public:
    nan_thread_pool();
    virtual ~nan_thread_pool();
    
  public:
    
  };
}

#endif /* nan_thread_pool_h */
