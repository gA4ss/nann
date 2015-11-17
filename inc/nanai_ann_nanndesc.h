#ifndef nanai_ann_nanndesc_h
#define nanai_ann_nanndesc_h

namespace nanai {
  
#define NANAI_ANN_DESC_RETURN               0
#define NANAI_ANN_DESC_CONTINUE             1
  
  typedef struct _nanai_ann_nanndesc nanai_ann_nanndesc;
  
  typedef void (*fptr_ann_event_added)();             /*! 发生在dlopen成功之后调用 */
  typedef void (*fptr_ann_event_close)();             /*! 发生在dlclose之前调用 */
  
  typedef int (*fptr_ann_mapreduce_map)(void *task,
                                        void *config,
                                        void *inputs,
                                        void *targets,
                                        void *ann,
                                        void *map_results);
  typedef int (*fptr_ann_mapreduce_reduce)(int wt, void *task, void *map_results, void *reduce_result);
  
  typedef void (*fptr_ann_alg_input_filter)(void *input, void *input_filted);
  typedef void (*fptr_ann_alg_result)(void *output, void* result);
  typedef double (*fptr_ann_alg_output_error)(double target, double output);
  typedef int (*fptr_ann_alg_calculate)(void *task, void *input, void *target, void *output, void *ann);
  
  typedef double (*fptr_ann_alg_hidden_calc)(int h, double input);
  typedef void (*fptr_ann_alg_hidden_error)(int h, void *delta_k, void *w_kh, void *o_h, void *delta_h);
  typedef void (*fptr_ann_alg_hidden_adjust_weight)(int h, void *layer, void *delta, void *wm, void *prev_dwm);
  
  /*
   * 一些回调函数
   */
  typedef void (*fptr_ann_monitor_except)(int cid, const char *task, int errcode, void *arg);             /* 当异常触发时回调 */
  typedef void (*fptr_ann_monitor_trained)(int cid, const char *task, void *arg);                         /* 训练完毕 */
  typedef void (*fptr_ann_monitor_progress)(int cid, const char *task, int progress, void *arg);          /* 进度控制回调 */
  typedef void (*fptr_ann_monitor_alg_uninstall)(int cid);                                                /* 当算法被卸载时调用 */
  
  #define MAX_NANN_BUFFER           256         // 最大缓存数量

  struct _nanai_ann_nanndesc {
    char name[MAX_NANN_BUFFER];                 /* 名称 */
    char description[MAX_NANN_BUFFER];          /* 描述 */
    
    /* 在nanai_ann_nanncalc中调用 */
    fptr_ann_alg_input_filter fptr_input_filter;
    fptr_ann_alg_result fptr_result;
    fptr_ann_alg_output_error fptr_output_error;
    fptr_ann_alg_calculate fptr_calculate;
    
    fptr_ann_alg_hidden_calc fptr_hidden_calcs;
    fptr_ann_alg_hidden_error fptr_hidden_errors;
    fptr_ann_alg_hidden_adjust_weight fptr_hidden_adjust_weights;
    
    fptr_ann_monitor_except callback_monitor_except;
    fptr_ann_monitor_trained callback_monitor_trained;
    fptr_ann_monitor_progress callback_monitor_progress;
    fptr_ann_monitor_alg_uninstall callback_monitor_alg_uninstall;
    
    /* 在nanai_ann_nannmgr中调用 */
    fptr_ann_event_added fptr_event_added;
    fptr_ann_event_close fptr_event_close;
    
    /* 在nanai_mapreduce_ann中调用 */
    fptr_ann_mapreduce_map fptr_map;
    fptr_ann_mapreduce_reduce fptr_reduce;
  };
  
  /* ann_alg_setup */
  typedef nanai_ann_nanndesc* (*fptr_ann_alg_setup)(const char *conf_dir);
}

#endif /* nanai_ann_nanndesc_h */
