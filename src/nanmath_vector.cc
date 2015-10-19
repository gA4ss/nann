#include <stdio.h>
#include <stdlib.h>

#include "nanmath_vector.h"

namespace nanmath {
  
  nanmath_vector::nanmath_vector() {
    _number = 0;
    _vector = NULL;
  }
  
  nanmath_vector::nanmath_vector(int n) {
    create(n);
  }
  
  nanmath_vector::~nanmath_vector() {
    destroy();
  }
  
  int nanmath_vector::create(int n) {
    
    destroy();
    
    _number = n;
    
    _vector = alloc_db_vector(n);
    if (_vector == NULL) {
      return -1;
    }
    
    return 0;
  }
  
  void nanmath_vector::destroy() {
    if ((_number != 0) && (_vector != NULL)) {
      free(_vector);
      _vector = NULL;
      _number = 0;
    }
  }
  
  double nanmath_vector::at(int i) {
    return 0.0;
  }
  
  double *nanmath_vector::alloc_db_vector(int n) {
    double *p = (double *) malloc ((unsigned) (n * sizeof (double)));
    if (p == NULL) {
      return NULL;
    }
    return p;
  }
}
