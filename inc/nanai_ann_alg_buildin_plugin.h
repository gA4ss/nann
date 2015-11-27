#ifndef nanai_ann_alg_buildin_plugin_h
#define nanai_ann_alg_buildin_plugin_h

#include <nanai_object.h>
#include <nanmath_vector.h>
#include <nanmath_matrix.h>
#include <nlang.h>

class nanai_ann_alg_buildin_plugin : public nanai::nanai_object {
public:
  nanai_ann_alg_buildin_plugin();
  virtual ~nanai_ann_alg_buildin_plugin();
  
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


#endif /* nanai_ann_alg_buildin_plugin_h */
