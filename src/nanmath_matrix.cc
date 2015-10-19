#include <stdio.h>
#include <stdlib.h>

#include "nanmath_matrix.h"

namespace nanmath {
  
  nanmath_matrix::nanmath_matrix() {
    _row = 0;
    _col = 0;
    _matrix = NULL;
  }
  
  nanmath_matrix::nanmath_matrix(int r, int c) {
    create(r, c);
  }
  
  nanmath_matrix::~nanmath_matrix() {
    destroy();
  }
  
  int nanmath_matrix::create(int r, int c) {
    
    destroy();
    
    _row = r;
    _col = c;
    
    _matrix = alloc_db_matrix(_row, _col);
    if (_matrix == NULL) {
      return -1;
    }
    
    return 0;
  }
  
  void nanmath_matrix::destroy() {
    if ((_row != 0) && (_col != 0) && (_matrix != NULL)) {
      for (int i = 0; i < _col; i++) {
        free(_matrix[i]);
      }
      
      free(_matrix);
      _matrix = NULL;
      _row = 0;
      _col = 0;
    }
  }
  
  double nanmath_matrix::at(int r, int c) {
    return 0.0;
  }
  
  double *nanmath_matrix::alloc_db_vector(int n) {
    double *p = (double *) malloc ((unsigned) (n * sizeof (double)));
    if (p == NULL) {
      return NULL;
    }
    return p;
  }
  
  double **nanmath_matrix::alloc_db_matrix(int r, int c) {
    double **p = (double **) malloc ((unsigned) (r * sizeof (double *)));
    if (p == NULL) {
      return NULL;
    }
    
    for (int i = 0; i < c; i++) {
      p[i] = alloc_db_vector(c);
    }
    
    return p;
  }
}
