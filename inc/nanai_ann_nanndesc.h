#ifndef nanai_ann_nanndesc_h
#define nanai_ann_nanndesc_h

namespace nanai {
  
  typedef struct _nanai_ann_nanndesc nanai_ann_nanndesc;
  
  typedef void (*fptr_ann_main)();
  
  typedef void (*fptr_ann_input_filter)(void *input, void *input_filted);
  typedef void (*fptr_ann_result)(void *output, void* result);
  typedef double (*fptr_ann_output_error)(double target, double output);
  typedef void (*fptr_ann_calculate)(void *task, void *input, void *target, void *output, void *arg);
  
  typedef void (*fptr_ann_hidden_init)(int h, void *wmat);
  typedef double (*fptr_ann_hidden_calc)(int h, double input);
  typedef void (*fptr_ann_hidden_error)(int h, void *delta_k, void *w_kh, void *o_h, void *delta_h);
  typedef void (*fptr_ann_hidden_adjust_weight)(int h, void *layer, void *delta, void *wm, void *prev_dwm);

  /*
   * 一些回调函数
   */
  typedef void (*fptr_ann_monitor_except)(int cid, const char *task, int errcode, void *arg);             /* 当异常触发时回调 */
  typedef void (*fptr_ann_monitor_trained)(int cid, const char *task, void *arg);                         /* 训练完毕 */
  typedef void (*fptr_ann_monitor_trained2)(int cid, const char *task, void *arg);                        /* 训练完毕，无输出 */
  typedef void (*fptr_ann_monitor_calculated)(int cid, const char *task, void *arg);                      /* 当计算完毕后回调 */
  typedef void (*fptr_ann_monitor_progress)(int cid, const char *task, int progress, void *arg);          /* 进度控制回调 */
  typedef void (*fptr_ann_monitor_alg_uninstall)(int cid);                                                /* 当算法被卸载时调用 */

  #define MAX_HIDDEN_NUMBER         31          // 最大容纳隐藏层个数
  #define MAX_NANN_BUFFER           256         // 最大缓存数量

  struct _nanai_ann_nanndesc {
    char name[MAX_NANN_BUFFER];                 /* 名称 */
    char description[MAX_NANN_BUFFER];          /* 描述 */
    int ninput;                                 /* 输入向量的个数 */
    int nhidden;                                /* 隐藏层个数 */
    int noutput;                                /* 输出向量的个数 */
    int nneure[MAX_HIDDEN_NUMBER];              /* 每个隐藏层的神经元个数 */
    
    fptr_ann_input_filter fptr_input_filter;
    fptr_ann_result fptr_result;
    fptr_ann_output_error fptr_output_error;
    fptr_ann_calculate fptr_calculate;
    
    fptr_ann_hidden_init fptr_hidden_inits;
    fptr_ann_hidden_calc fptr_hidden_calcs;
    fptr_ann_hidden_error fptr_hidden_errors;
    fptr_ann_hidden_adjust_weight fptr_hidden_adjust_weights;
    
    fptr_ann_monitor_except callback_monitor_except;
    fptr_ann_monitor_trained callback_monitor_trained;
    fptr_ann_monitor_trained2 callback_monitor_trained_nooutput;
    fptr_ann_monitor_calculated callback_monitor_calculated;
    fptr_ann_monitor_progress callback_monitor_progress;
    fptr_ann_monitor_alg_uninstall callback_monitor_alg_uninstall;
    
    fptr_ann_main fptr_main;
  };
  
  /* ann_alg_setup */
  typedef nanai_ann_nanndesc* (*fptr_ann_alg_setup)(const char *conf_dir);
}

#endif /* nanai_ann_nanndesc_h */
