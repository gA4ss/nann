#include <cmath>
#include <nanai_object.h>
#include <nanmath_vector.h>
#include <nanmath_matrix.h>
#include <nlang.h>
#include <nanai_ann_alg_buildin_logistic.h>

nanai_ann_alg_buildin_logistic::nanai_ann_alg_buildin_logistic() : nanai_ann_alg_buildin_plugin() {
  
}

nanai_ann_alg_buildin_logistic::~nanai_ann_alg_buildin_logistic() {
  
}

double nanai_ann_alg_buildin_logistic::hidden_calc(nlang::nlang &nl,
                                                   double input) {
  return (1.0 / (1.0 + exp(-input)));
}

void nanai_ann_alg_buildin_logistic::hidden_error(nlang::nlang &nl,
                                                  nanmath::nanmath_vector *delta_k,
                                                  nanmath::nanmath_matrix *w_kh,
                                                  nanmath::nanmath_vector *o_h,
                                                  nanmath::nanmath_vector *delta_h) {
  delta_h->create(o_h->size());
  nanmath::nanmath_vector delta_sum = w_kh->right_mul(*delta_k);
  
  for (size_t i = 0; i < o_h->size(); i++) {
    delta_h->set(i, o_h->at(i) * (1 - o_h->at(i)) * delta_sum[i]);
  }
}

double nanai_ann_alg_buildin_logistic::output_error(nlang::nlang &nl,
                                                    double target,
                                                    double output) {
  double delta = output * (1.0 - output) * (target - output);
  return delta;
}

void nanai_ann_alg_buildin_logistic::hidden_adjust_weight(nlang::nlang &nl,
                                                          nanmath::nanmath_vector *layer,
                                                          nanmath::nanmath_vector *delta,
                                                          nanmath::nanmath_matrix *wm,
                                                          nanmath::nanmath_matrix *prev_dwm) {
  if (wm->col_size() != delta->size() ||
      (wm->row_size() != layer->size())) {
    nanai::error(NANAI_ERROR_LOGIC_ANN_INVALID_MATRIX_DEGREE);
  }
  
  double eta = 0.0, momentum = 0.0;
  if (nl.get("eta", eta)) {
    nanai::error(NANAI_ERROR_LOGIC_INVALID_CONFIG);
  }
  
  if (nl.get("momentum", momentum)) {
    nanai::error(NANAI_ERROR_LOGIC_INVALID_CONFIG);
  }
  
  /* 这里是遍历列向量 */
  for (size_t i = 0; i < delta->size(); i++) {          /* 矩阵的列 */
    for (size_t j = 0; j < layer->size(); j++) {        /* 矩阵的行 */
      /* 让上一层的每个输入向量都乘以当前的偏差值
       * 然后在修订这个偏差值的权向量
       */
      double new_dw = (eta * delta->at(i) * layer->at(j)) + (momentum * prev_dwm->at(j, i));
      double t = wm->at(j, i) + new_dw;
      wm->set(j, i, t);
      prev_dwm->set(j, i, new_dw);
    }
  }
}