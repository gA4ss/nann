#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <nanmath_vector.h>
#include <nanmath_matrix.h>

namespace nanmath {
  nanmath_vector nv_null;
  
  nanmath_vector::nanmath_vector() : nanmath_object() {
  }
  
  nanmath_vector::nanmath_vector(size_t n) : nanmath_object() {
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
  
  bool nanmath_vector::empty() const {
    return _vector.empty();
  }
  
  void nanmath_vector::clear() {
    _vector.clear();
  }
  
  double nanmath_vector::at(size_t i) const {
    return _vector[i];
  }
  
  void nanmath_vector::set(size_t i, double v) {
    _vector[i] = v;
  }
  
  void nanmath_vector::set(double *v, int vs) {
    destroy();
    _vector.resize(vs);
    for (int i = 0; i < vs; i++) {
      _vector[i] = v[i];
    }
  }
  
  void nanmath_vector::set(const std::vector<double> &v) {
    destroy();
    _vector.resize(v.size());
    for (int i = 0; i < v.size(); i++) {
      _vector[i] = v[i];
    }
  }
  
  void nanmath_vector::set(const nanmath_vector &v) {
    destroy();
    _vector.resize(v.size());
    for (size_t i = 0; i < v.size(); i++) {
      _vector[i] = v.at(i);
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
    for (size_t i = 0; i < _vector.size(); i++) {
      if (t == 1) _vector[i] = s_dpn1();
      else _vector[i] = s_drnd();
    }
  }
  
  nanmath_vector nanmath_vector::add(const nanmath_vector &v) {
    if (_vector.size() != v.size()) {
      error(NANMATH_ERROR_LOGIC_INVALID_VECTOR_DEGREE);
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
      error(NANMATH_ERROR_LOGIC_INVALID_VECTOR_DEGREE);
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
    
    for (size_t i = 0; i < _vector.size(); i++) {
      for (size_t j = 0; j < v.size(); j++) {
        double p = _vector[i] * v.at(j);
        res.set(i, j, p);
      }
    }
    return res.get();
  }
  
  nanmath_vector nanmath_vector::mul(const double v) {
    nanmath_vector res;
    for (auto i : _vector) {
      res.push(i * v);
    }
    return res;
  }
  
  double nanmath_vector::dot(const nanmath_vector &v) {
    if (_vector.size() != v.size()) {
      error(NANMATH_ERROR_LOGIC_INVALID_VECTOR_DEGREE);
    }
    
    double res = 0.0;
    for (size_t i = 0; i < _vector.size(); i++) {
      res += _vector[i] * v.at(i);
    }
    
    return res;
  }
  
  void nanmath_vector::print() {
    for (auto i : _vector) {
      std::cout << std::setiosflags(std::ios::fixed) << std::setiosflags(std::ios::left)
                << std::setprecision(2) << std::setw(8) << i;
    }
  }
  
  double nanmath_vector::operator [](size_t i) const {
    return at(i);
  }
  
  void nanmath_vector::check(size_t i) {
    if (i >= size()) {
      error(NANMATH_ERROR_LOGIC_INVALID_VECTOR_DEGREE);
    }
  }
}
