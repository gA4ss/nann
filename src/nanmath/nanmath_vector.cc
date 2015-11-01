#include <stdio.h>
#include <stdlib.h>

#include <stdexcept>
#include <nanmath_vector.h>
#include <nanmath_matrix.h>

namespace nanmath {
  nanmath_vector nv_null;
  
  nanmath_vector::nanmath_vector() {
    create(4);
  }
  
  nanmath_vector::nanmath_vector(size_t n) {
    create(n);
  }
  
  nanmath_vector::nanmath_vector(const nanmath_vector &t) {
    set(t);
  }
  
  nanmath_vector::~nanmath_vector() {
    destroy();
  }
  
  void nanmath_vector::create(size_t n) {
    destroy();
    _vector.resize(n);
  }
  
  void nanmath_vector::destroy() {
    clear();
  }
  
  void nanmath_vector::clear() {
    _vector.clear();
  }
  
  double nanmath_vector::at(size_t i) const {
    double res = 0.0;
    try {
      res = _vector[i];
    } catch (...) {
      // error
      throw;
    }
    
    return res;
  }
  
  void nanmath_vector::set(size_t i, double v) {
    try {
      _vector[i] = v;
    } catch (...) {
      // error
      throw;
    }
  }
  
  void nanmath_vector::set(double *v, int vs) {
    try {
      destroy();
      _vector.resize(vs);
      for (int i = 0; i < vs; i++) {
        _vector[i] = v[i];
      }
    } catch (...) {
      // error
      throw;
    }
  }
  
  void nanmath_vector::set(const std::vector<double> &v) {
    try {
      destroy();
      _vector.resize(v.size());
      for (int i = 0; i < v.size(); i++) {
        _vector[i] = v[i];
      }
    } catch (...) {
      // error
      throw;
    }
  }
  
  void nanmath_vector::set(const nanmath_vector &v) {
    try {
      destroy();
      _vector.resize(v.size());
      for (int i = 0; i < v.size(); i++) {
        _vector[i] = v.at(i);
      }
    } catch (...) {
      // error
      throw;
    }
  }
  
  void nanmath_vector::push(double v) {
    _vector.push_back(v);
  }
  
  void nanmath_vector::pop() {
    _vector.pop_back();
  }
  
  double nanmath_vector::back() {
    return _vector.back();
  }
  
  size_t nanmath_vector::size() const {
    return _vector.size();
  }
  
  void nanmath_vector::resize(size_t s) {
    _vector.resize(s);
  }
  
  void nanmath_vector::zero() {
    for (int i = 0; i < _vector.size(); i++) {
      _vector[i] = 0;
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
  
  void nanmath_vector::random(int t) {
    for (int i = 0; i < _vector.size(); i++) {
      if (t == 1) _vector[i] = s_dpn1();
      else _vector[i] = s_drnd();
    }
  }
  
  nanmath_vector nanmath_vector::add(const nanmath_vector &v) {
    if (_vector.size() != v.size()) {
      // error
    }
    
    nanmath_vector res;
    int j = 0;
    for (auto i : _vector) {
      res.push(i + v[j++]);
    }
    return res;
  }
  
  nanmath_vector nanmath_vector::sub(const nanmath_vector &v) {
    if (_vector.size() != v.size()) {
      // error
    }
    
    nanmath_vector res;
    int j = 0;
    for (auto i : _vector) {
      res.push(i - v[j++]);
    }
    return res;
  }
 
  std::vector<std::vector<double> > nanmath_vector::mul(const nanmath_vector &v) {
    nanmath_matrix res(_vector.size(), v.size());
    
    try {
      for (int i = 0; i < _vector.size(); i++) {
        for (int j = 0; j < v.size(); j++) {
          double p = _vector[i] * v.at(j);
          res.set(i, j, p);
        }
      }
    } catch (...) {
      throw;
    }
    
    return res.get();
  }
  
  double nanmath_vector::dot(const nanmath_vector &v) {
    if (_vector.size() != v.size()) {
      throw std::invalid_argument("not match argument size");
    }
    
    double res = 0.0;
    try {
      for (int i = 0; i < _vector.size(); i++) {
        res += _vector[i] * v.at(i);
      }
    } catch (...) {
      throw;
    }
    
    return res;
  }
  
  void nanmath_vector::print() {
    
  }
  
  double nanmath_vector::operator [](size_t i) const {
    return at(i);
  }
  
}
