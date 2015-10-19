#ifndef nanmath_matrix_h
#define nanmath_matrix_h

namespace nanmath {

  class nanmath_vector {
    
  };
  
  class nanmath_matrix {
  public:
    nanmath_matrix();
    nanmath_matrix(int r, int c);
    virtual ~nanmath_matrix();
    
  public:
    virtual int create(int r, int c);
    virtual void destroy();
    virtual double at(int r, int c);
    
  private:
    double *alloc_db_vector(int n);
    double **alloc_db_matrix(int r, int c);
    
  protected:
    int _row;
    int _col;
    double **_matrix;
  };
  
}

#endif /* nanmath_matrix_h */
