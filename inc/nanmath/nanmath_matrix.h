#ifndef nanmath_matrix_h
#define nanmath_matrix_h

#include <vector>
#include <nanmath_vector.h>

namespace nanmath {
  
  class nanmath_matrix {
  public:
    nanmath_matrix();
    nanmath_matrix(size_t r, size_t c);
    nanmath_matrix(const std::vector<std::vector<double> > &mat);
    nanmath_matrix(const nanmath_matrix &t);
    virtual ~nanmath_matrix();
    
  public:
    virtual void create(size_t r, size_t c);
    virtual void destroy();
    virtual void clear();
    virtual double at(size_t r, size_t c) const;
    virtual size_t row_size() const;
    virtual size_t col_size() const;
    virtual void set(size_t r, size_t c, double v);
    virtual void set(const nanmath_matrix &mat);
    virtual void set_row(size_t r, const std::vector<double> &row);
    virtual void set_col(size_t c, const std::vector<double> &col);
    virtual void push_row(const std::vector<double> &row);
    virtual std::vector<std::vector<double> > get();
    virtual void resize(size_t r, size_t c);
    virtual void print();
    
  public:
    virtual void zero();
    virtual void random(int t=0);
    virtual nanmath_matrix T() const;
    virtual nanmath_vector left_mul(const nanmath_vector &v);
    virtual nanmath_vector right_mul(const nanmath_vector &v);
    virtual nanmath_vector mul(const nanmath_vector &v);
    virtual nanmath_matrix mul(const nanmath_matrix &mat);
    
  public:
    std::vector<double> operator [](size_t r) const;
    
  protected:
    std::vector<std::vector<double> > _matrix;
  };
  
  extern nanmath_matrix nm_null;
  
}

#endif /* nanmath_matrix_h */
