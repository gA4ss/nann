#ifndef nanai_ann_nanndesc_h
#define nanai_ann_nanndesc_h

#include <vector>

#include <nanmath_vector.h>
#include <nanmath_matrix.h>

namespace nanai {

  typedef int (*fptr_ann_result)(nanmath::nanmath_vector &output, nanmath::nanmath_vector &result);
  
  typedef int (*fptr_ann_hidden_init)(int ihidden, nanmath::nanmath_matrix &weights);
  typedef int (*fptr_ann_hidden_calc)(int ihidden, double input, double *output);
  typedef int (*fptr_ann_hidden_error)(int ihidden, nanmath::nanmath_vector &weights, double target, double output);

  typedef void (*fptr_ann_monitor_except)(int cid, const char *task, int errcode);
  typedef void (*fptr_ann_monitor_trained)(int cid, const char *task, std::vector<nanmath::nanmath_matrix> &ann);
  typedef void (*fptr_ann_monitor_calculated)(int cid, const char *task, nanmath::nanmath_vector &output);
  typedef void (*fptr_ann_monitor_progress)(int cid, const char *task, int progress, void *arg);

  #define MAX_HIDDEN_NUMBER         31          // 最大容纳隐藏层个数

  typedef struct _nanai_ann_nanndesc {
    char *name;
    char *description;
    int ninput;
    int nhidden;
    int noutput;
    int nneure[MAX_HIDDEN_NUMBER];
    
    fptr_ann_result taret_outfilter;
    fptr_ann_hidden_init hidden_init[MAX_HIDDEN_NUMBER+1];
    fptr_ann_hidden_calc hidden_calc[MAX_HIDDEN_NUMBER+1];
    fptr_ann_hidden_error hidden_error[MAX_HIDDEN_NUMBER+1];
    
    fptr_ann_monitor_except callback_monitor_except;
    fptr_ann_monitor_trained callback_monitor_trained;
    fptr_ann_monitor_calculated callback_monitor_calculated;
    fptr_ann_monitor_progress callback_monitor_progress;
  } nanai_ann_nanndesc;
}

#endif /* nanai_ann_nanndesc_h */
