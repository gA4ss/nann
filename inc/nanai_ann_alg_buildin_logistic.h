#ifndef nanai_ann_alg_buildin_logistic_h
#define nanai_ann_alg_buildin_logistic_h

#include <nanai_object.h>
#include <nanmath_vector.h>
#include <nanmath_matrix.h>
#include <nlang.h>
#include <nanai_ann_alg_buildin_plugin.h>

class nanai_ann_alg_buildin_logistic : public nanai_ann_alg_buildin_plugin {
public:
  nanai_ann_alg_buildin_logistic();
  virtual ~nanai_ann_alg_buildin_logistic();
  
public:
  virtual double hidden_calc(nlang::nlang &nl,
                             double input);
  
  virtual void hidden_error(nlang::nlang &nl,
                            nanmath::nanmath_vector *delta_k,
                            nanmath::nanmath_matrix *w_kh,
                            nanmath::nanmath_vector *o_h,
                            nanmath::nanmath_vector *delta_h);
  
  virtual double output_error(nlang::nlang &nl,
                              double target,
                              double output);
  
  virtual void hidden_adjust_weight(nlang::nlang &nl,
                                    nanmath::nanmath_vector *layer,
                                    nanmath::nanmath_vector *delta,
                                    nanmath::nanmath_matrix *wm,
                                    nanmath::nanmath_matrix *prev_dwm);
};


#endif /* nanai_ann_alg_buildin_logistic_h */
