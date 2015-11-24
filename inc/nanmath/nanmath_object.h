#ifndef nanmath_object_h
#define nanmath_object_h

#include <nan_object.h>

#define NANMATH_ERROR_LOGIC                           0x82200000
#define NANMATH_ERROR_LOGIC_INVALID_VECTOR_DEGREE     0x82200001
#define NANMATH_ERROR_LOGIC_INVALID_MATRIX_DEGREE     0x82200002

namespace nanmath {

  class nanmath_object : public nanan::nan_object {
  public:
    nanmath_object();
    virtual ~nanmath_object();
  };
  
  void error(size_t err);
}

#endif /* nanmath_object_h */
