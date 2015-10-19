#ifndef nanmath_vector_h
#define nanmath_vector_h

namespace nanmath {
  
  class nanmath_vector {
  public:
    nanmath_vector();
    nanmath_vector(int n);
    virtual ~nanmath_vector();
    
  public:
    virtual int create(int n);
    virtual void destroy();
    virtual double at(int i);
    
  private:
    double *alloc_db_vector(int n);
    
  protected:
    int _number;
    double *_vector;
  };
  
}

#endif /* nanmath_vector_h */
