#include <nanai_object.h>
#include <nanmath_vector.h>
#include <nanmath_matrix.h>
#include <nlang.h>
#include <nanai_ann_alg_buildin_plugin.h>


nanai_ann_alg_buildin_plugin::nanai_ann_alg_buildin_plugin() : nanai::nanai_object() {
  
}

nanai_ann_alg_buildin_plugin::~nanai_ann_alg_buildin_plugin() {
  
}


double nanai_ann_alg_buildin_plugin::hidden_calc(nlang::nlang &nl,
                                                 double input) {
  return 0.0;
}

void nanai_ann_alg_buildin_plugin::hidden_error(nlang::nlang &nl,
                                                nanmath::nanmath_vector *delta_k,
                                                nanmath::nanmath_matrix *w_kh,
                                                nanmath::nanmath_vector *o_h,
                                                nanmath::nanmath_vector *delta_h) {
  
}

double nanai_ann_alg_buildin_plugin::output_error(nlang::nlang &nl,
                                                  double target,
                                                  double output) {
  return 0.0;
}

void nanai_ann_alg_buildin_plugin::hidden_adjust_weight(nlang::nlang &nl,
                                                        nanmath::nanmath_vector *layer,
                                                        nanmath::nanmath_vector *delta,
                                                        nanmath::nanmath_matrix *wm,
                                                        nanmath::nanmath_matrix *prev_dwm) {
  
}
