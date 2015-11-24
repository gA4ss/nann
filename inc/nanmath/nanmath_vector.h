#ifndef nanmath_vector_h
#define nanmath_vector_h

#include <vector>
#include <nanmath_object.h>

namespace nanmath {
  
  class nanmath_vector :public nanmath_object {
  public:
    nanmath_vector();
    nanmath_vector(size_t n);
    nanmath_vector(const nanmath_vector &t);
    virtual ~nanmath_vector();
    
  public:
    virtual void create(size_t n);
    virtual void destroy();
    virtual bool empty() const;
    virtual void clear();
    virtual double at(size_t i) const;
    virtual void set(size_t i, double v);
    virtual void set(double *v, int vs);
    virtual void set(const std::vector<double> &v);
    virtual void set(const nanmath_vector &v);
    virtual void push(double v);
    virtual void pop();
    virtual double back();
    virtual size_t size() const;
    virtual void resize(size_t s);
    virtual void print();
    
  public:
    virtual void zero();
    virtual void random(int t=0);
    virtual nanmath_vector add(const nanmath_vector &v);
    virtual nanmath_vector sub(const nanmath_vector &v);
    virtual std::vector<std::vector<double> > mul(const nanmath_vector &v);
    virtual nanmath_vector mul(const double v);
    virtual double dot(const nanmath_vector &v);
    
  public:
    double operator [](size_t i) const;
    
  protected:
    void check(size_t i);
    
  protected:
    std::vector<double> _vector;
  };
 
  extern nanmath_vector nv_null;
}

#endif /* nanmath_vector_h */
