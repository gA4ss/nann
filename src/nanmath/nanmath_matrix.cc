#include <stdio.h>
#include <stdlib.h>

#include <stdexcept>
#include <nanmath_matrix.h>

namespace nanmath {
  
  nanmath_matrix nm_null;
  
  nanmath_matrix::nanmath_matrix() {
    create(4, 4);
  }
  
  nanmath_matrix::nanmath_matrix(size_t r, size_t c) {
    create(r, c);
  }
  
  nanmath_matrix::nanmath_matrix(const std::vector<std::vector<double> > &mat) {
    destroy();
    size_t nr = mat.size(), nc = mat[0].size();
    
    _matrix.resize(nr);
    for (int i = 0; i < mat.size(); i++) {
      _matrix[i].resize(nc);
      for (int j = 0; j < mat[0].size(); j++) {
        _matrix[i][j] = mat[i][j];
      }
    }
  }
  
  nanmath_matrix::nanmath_matrix(const nanmath_matrix &t) {
    set(t);
  }
  
  nanmath_matrix::~nanmath_matrix() {
    destroy();
  }
  
  void nanmath_matrix::create(size_t r, size_t c) {
    
    try {
      destroy();
      _matrix.resize(r);
      
      for (int i = 0; i < r; i++) {
        _matrix[i].resize(c);
      }
    } catch (...) {
      // error
      throw;
    }
  }
  
  void nanmath_matrix::destroy() {
    clear();
  }
  
  void nanmath_matrix::clear() {
    _matrix.clear();
  }
  
  double nanmath_matrix::at(size_t r, size_t c) const {
    double res = 0.0;
    try {
      res = _matrix[r][c];
    } catch (...) {
      // error
      throw;
    }
    return res;
  }
  
  size_t nanmath_matrix::row_size() const {
    return _matrix.size();
  }
  
  size_t nanmath_matrix::col_size() const {
    return _matrix[0].size();
  }
  
  void nanmath_matrix::set(size_t r, size_t c, double v) {
    try {
      _matrix[r][c] = v;
    } catch (...) {
      // error
      throw;
    }
  }
  
  void nanmath_matrix::set(const nanmath_matrix &mat) {
    try {
      destroy();
      size_t r = mat.row_size(), c = mat.col_size();
      _matrix.resize(r);
      
      for (int i = 0; i < r; i++) {
        _matrix[i].resize(c);
        for (int j = 0; j < c; j++) {
          _matrix[r][c] = mat.at(i, j);
        }
      }
    } catch (...) {
      // error
      throw;
    }
  }
  
  std::vector<std::vector<double> > nanmath_matrix::get() {
    return _matrix;
  }
  
  void nanmath_matrix::resize(size_t r, size_t c) {
    destroy();
    create(r, c);
  }
  
  void nanmath_matrix::zero() {
    for (auto &i : _matrix) {
      for (int j = 0; j < i.size(); j++) {
        i[j] = 0;
      }
    }
  }
  
  /* 返回 0.0 到 1.0的随机值 */
  static double s_drnd() {
    return ((double) random() / (double) 0x7fffffff);
  }
  
  /* 返回 -1.0 到 1.0的随机值 */
  static double s_dpn1() {
    return ((s_drnd() * 2.0) - 1.0);
  }
  
  void nanmath_matrix::random(int t) {
    srandom((int)time(NULL));
    
    for (auto &i : _matrix) {
      for (int j = 0; j < i.size(); j++) {
        if (t == 1) i[j] = s_dpn1();
        else i[j] = s_drnd();
      }
    }
  }
  
  nanmath_matrix nanmath_matrix::T() const {
    nanmath_matrix res;
    try {
      size_t c = col_size();
      size_t r = row_size();

      res.resize(c, r);
      
      for (int i = 0; i < c; i++) {
        for (int j = 0; j < r; j++) {
          res[i][j] = _matrix[j][i];
        }
      }
    } catch (...) {
      // error
      throw;
    }
    
    return res;
  }
  
  nanmath_vector nanmath_matrix::left_mul(const nanmath_vector &v) {
    nanmath_vector res;
    /* 向量与矩阵左乘，向量的个数必须与矩阵的行相同 */
    size_t r = row_size();
    size_t c = col_size();
    
    if (v.size() == r) {
      res.resize(c);
      
      for (int i = 0; i < c; i++) {
        for (int j = 0; j < r; j++) {
          res.set(i, v[j] * _matrix[j][i]);
        }
      }
      
    } else {
      throw std::invalid_argument("inner matrix dimensions must agree");
    }
    return res;
  }
  
  nanmath_vector nanmath_matrix::right_mul(const nanmath_vector &v) {
    nanmath_vector res;
    /* 向量与矩阵右乘，向量的个数必须与矩阵的列相同 */
    size_t r = row_size();
    size_t c = col_size();
    
    if (v.size() == c) {
      res.resize(r);
      
      for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
          res.set(i, v[j] * _matrix[i][j]);
        }
      }
      
    } else {
      throw std::invalid_argument("inner matrix dimensions must agree");
    }
    return res;

  }
  
  nanmath_vector nanmath_matrix::mul(const nanmath_vector &v) {
    nanmath_vector res;
    size_t c = 0, r = 0;
    /* 矩阵与向量相乘，向量的个数或者等于矩阵的列数或者等于列数 */
    try {
      
      c = col_size();
      r = row_size();
      
      if (v.size() == r) {
        res.resize(c);
        
        for (int i = 0; i < c; i++) {
          for (int j = 0; j < r; j++) {
            res.set(i, v[j] * _matrix[j][i]);
          }
        }
        
      } else if (v.size() == c) {
        res.resize(r);
        
        for (int i = 0; i < r; i++) {
          for (int j = 0; j < c; j++) {
            res.set(i, v[j] * _matrix[i][j]);
          }
        }
        
      } else {
        throw std::invalid_argument("inner matrix dimensions must agree");
      }
      
    } catch (...) {
      // error
      throw;
    }
    return res;
  }
  
  nanmath_matrix nanmath_matrix::mul(const nanmath_matrix &mat) {
    nanmath_matrix res;
    
    return res;
  }
  
  void nanmath_matrix::print() {
#if 0
    for (int i = 0; i < row_size(); i++) {
      for (int j = 0; j < col_size(); j++) {
        
      }
    }
#endif
  }
  
  std::vector<double> nanmath_matrix::operator [](size_t r) const {
    std::vector<double> res;
    
    try {
      res = _matrix[r];
    } catch (...) {
      // error
      throw;
    }
    return res;
  }
}