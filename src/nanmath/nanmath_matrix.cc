#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <nanmath_matrix.h>

namespace nanmath {
  
  nanmath_matrix nm_null;
  
  nanmath_matrix::nanmath_matrix() : nanmath_object() {
  }
  
  nanmath_matrix::nanmath_matrix(size_t r, size_t c) : nanmath_object() {
    create(r, c);
  }
  
  nanmath_matrix::nanmath_matrix(const std::vector<std::vector<double> > &mat) {
    destroy();
    size_t nr = mat.size(), nc = mat[0].size();
    
    _matrix.resize(nr);
    for (size_t i = 0; i < mat.size(); i++) {
      _matrix[i].resize(nc);
      for (size_t j = 0; j < mat[0].size(); j++) {
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
    destroy();
    _matrix.resize(r);
      
    for (size_t i = 0; i < r; i++) {
        _matrix[i].resize(c);
    }
  }
  
  void nanmath_matrix::destroy() {
    clear();
  }
  
  bool nanmath_matrix::empty() const {
    return _matrix.empty();
  }
  
  void nanmath_matrix::clear() {
    _matrix.clear();
  }
  
  double nanmath_matrix::at(size_t r, size_t c) const {
    return _matrix[r][c];
  }
  
  size_t nanmath_matrix::row_size() const {
    return _matrix.size();
  }
  
  size_t nanmath_matrix::col_size() const {
    return _matrix[0].size();
  }
  
  void nanmath_matrix::set(size_t r, size_t c, double v) {
    check(r, c);
    _matrix[r][c] = v;
  }
  
  void nanmath_matrix::set(const nanmath_matrix &mat) {
    destroy();
    size_t r = mat.row_size();
    _matrix.clear();
    for (size_t i = 0; i < r; i++) {
      _matrix.push_back(mat[i]);
    }
  }
  
  void nanmath_matrix::set_row(size_t r, const std::vector<double> &row) {
    if (r > row_size()) {
      error(NANMATH_ERROR_LOGIC_INVALID_MATRIX_DEGREE);
    }
    
    _matrix[r] = row;
  }
  
  void nanmath_matrix::set_col(size_t c, const std::vector<double> &col) {
    if (c > col_size()) {
      error(NANMATH_ERROR_LOGIC_INVALID_MATRIX_DEGREE);
    }
  }
  
  void nanmath_matrix::push_row(const std::vector<double> &row) {
    _matrix.push_back(row);
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
      for (size_t j = 0; j < i.size(); j++) {
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
      for (size_t j = 0; j < i.size(); j++) {
        if (t == 1) i[j] = s_dpn1();
        else i[j] = s_drnd();
      }
    }
  }
  
  nanmath_matrix nanmath_matrix::T() const {
    nanmath_matrix res;
    
    size_t c = col_size();
    size_t r = row_size();

    res.resize(c, r);
      
    for (size_t i = 0; i < c; i++) {
      for (size_t j = 0; j < r; j++) {
        res[i][j] = _matrix[j][i];
      }
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
      
      for (size_t i = 0; i < c; i++) {
        for (size_t j = 0; j < r; j++) {
          res.set(i, v[j] * _matrix[j][i]);
        }
      }
      
    } else {
      error(NANMATH_ERROR_LOGIC_INVALID_MATRIX_DEGREE);
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
      
      for (size_t i = 0; i < r; i++) {
        for (size_t j = 0; j < c; j++) {
          res.set(i, v[j] * _matrix[i][j]);
        }
      }
      
    } else {
      error(NANMATH_ERROR_LOGIC_INVALID_MATRIX_DEGREE);
    }
    return res;

  }
  
  nanmath_matrix nanmath_matrix::mul(const double v) {
    nanmath_matrix res;
    size_t r = row_size(), c = col_size();
    res.create(r, c);
    
    for (size_t i = 0; i < r; i++) {
      for (size_t j = 0; j < c; j++) {
        res.set(i, j, _matrix[i][j] * v);
      }
    }
    
    return res;
  }
  
  nanmath_vector nanmath_matrix::mul(const nanmath_vector &v) {
    nanmath_vector res;
    size_t c = 0, r = 0;
    /* 矩阵与向量相乘，向量的个数或者等于矩阵的列数或者等于列数 */
    c = col_size();
    r = row_size();
      
    if (v.size() == r) {
      res.resize(c);
        
      for (size_t i = 0; i < c; i++) {
        for (size_t j = 0; j < r; j++) {
          res.set(i, v[j] * _matrix[j][i]);
        }
      }
        
    } else if (v.size() == c) {
      res.resize(r);
        
      for (size_t i = 0; i < r; i++) {
        for (size_t j = 0; j < c; j++) {
          res.set(i, v[j] * _matrix[i][j]);
        }
      }
        
    } else {
      error(NANMATH_ERROR_LOGIC_INVALID_MATRIX_DEGREE);
    }
    
    return res;
  }
  
  nanmath_matrix nanmath_matrix::mul(const nanmath_matrix &mat) {
    nanmath_matrix res;
    
    return res;
  }
  
  void nanmath_matrix::print() const {
    for (size_t i = 0; i < row_size(); i++) {
      for (size_t j = 0; j < col_size(); j++) {
        std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(2)
                  << std::setiosflags(std::ios::left) << std::setw(8) << _matrix[i][j];
        if (j == col_size() - 1) {
          std::cout << std::endl;
        } else {
          std::cout << " ";
        }
      }
    }
  }
  
  std::vector<double> nanmath_matrix::operator [](size_t r) const {
    return _matrix[r];
  }
  
  void nanmath_matrix::check(size_t r, size_t c) {
    if ((r >= row_size()) || (c >= col_size())) {
      error(NANMATH_ERROR_LOGIC_INVALID_MATRIX_DEGREE);
    }
  }
}
