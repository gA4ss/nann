#include <nanmath_object.h>

namespace nanmath {  
  nanmath_object::nanmath_object() : nanan::nan_object(){
    register_error(NANMATH_ERROR_LOGIC_INVALID_VECTOR_DEGREE, "invalid vector degree");
    register_error(NANMATH_ERROR_LOGIC_INVALID_MATRIX_DEGREE, "invalid matrix degree");
  }
  
  nanmath_object::~nanmath_object() {
    
  }
  
  void error(size_t err) {
    nanmath_object obj;
    obj.error(err);
  }
}