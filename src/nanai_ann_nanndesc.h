#ifndef nanai_ann_nanndesc_h
#define nanai_ann_nanndesc_h

#include "nanmath_matrix.h"

namespace nanai {

  typedef int (*fptr_ann_result)(nanmath::nanmath_vector &vec_output, unsigned char *result, int *size_result, int *count_result);
  typedef int (*fptr_hidden_init)(int idx_hidden, nanmath::nanmath_matrix &mat_weight);
  typedef int (*fptr_hidden_calc)(int idx_hidden, nanmath::nanmath_matrix &mat_weight,
                                  nanmath::nanmath_vector &vec_output,
                                  double* filter_input);
  typedef int (*fptr_hidden_error)(int idx_hidden, double* de);

  typedef void (*fptr_ann_monitor_except)(int tid, int errcode);
  typedef void (*fptr_ann_monitor_trained)(int tid, nanmath::nanmath_matrix &ann);
  typedef void (*fptr_ann_monitor_calculated)(int tid, double res);
  typedef void (*fptr_ann_monitor_progress)(int tid, int progress, nanmath::nanmath_matrix &ann);

  #define MAX_NETLAYER_NUMBER         32          // 最大容纳32个网络层

  typedef struct _nanai_ann_nanndesc {
    char *name;
    char *description;
    int nnetlayer;
    int nneure[MAX_NETLAYER_NUMBER];
    fptr_netlayer_infilter netlayer_infilter[MAX_NETLAYER_NUMBER];
    fptr_netlayer_outfilter netlayer_outfilter[MAX_NETLAYER_NUMBER];
    fptr_netlayer_init netlayer_init[MAX_NETLAYER_NUMBER];
    fptr_netlayer_error netlayer_error[MAX_NETLAYER_NUMBER];
    fptr_ann_result taret_outfilter;
    fptr_ann_monitor_except callback_monitor_except;
    fptr_ann_monitor_trained callback_monitor_trained;
    fptr_ann_monitor_calculated callback_monitor_calculated;
    fptr_ann_monitor_progress callback_monitor_progress;
  } nanai_ann_nanndesc;
}

#endif /* nanai_ann_nanndesc_h */
