#ifndef nanai_ann_nanncalc_h
#define nanai_ann_nanncalc_h

#include <stdio.h>
#include <pthread.h>
#include <time.h>
#include <string>
#include <vector>

#include "nanai_ann_nanndesc.h"

namespace nanai {
  
  /*
   * 状态消息
   */
  #define NANNCALC_ST_WAITING                0
  #define NANNCALC_ST_TRAINING               1
  #define NANNCALC_ST_TRAINING_NOTARGET      2
  #define NANNCALC_ST_CALCULATE              NANNCALC_ST_TRAINING_NOTARGET
  #define NANNCALC_ST_CONFIGURE              3
  #define NANNCALC_ST_STOP                   4
  
  void *nanai_ann_thread_worker(void *arg);

  class nanai_ann_nanncalc {
  public:
    nanai_ann_nanncalc(const char *lp=NULL);
    virtual ~nanai_ann_nanncalc();
    
  public:
    virtual int ann_training(nanmath::nanmath_vector &input, nanmath::nanmath_vector &target,
                             const char *task=NULL);
    virtual int ann_training_asyn(nanmath::nanmath_vector &input, nanmath::nanmath_vector &target,
                                  const char *task=NULL);
    
    virtual int ann_training_notarget(nanmath::nanmath_vector &input,
                                      const char *task=NULL);
    virtual int ann_training_notarget_asyn(nanmath::nanmath_vector &input,
                                           const char *task=NULL);
    
#define ann_calculate       ann_training_notarget
    virtual int ann_configure(nanai_ann_nanndesc &desc);
    virtual int ann_configure_asyn(nanai_ann_nanndesc &desc);
    
    virtual int ann_output(nanmath::nanmath_vector &output);
    virtual int ann_output();
    
    virtual int ann_stop();
    virtual int ann_free();
    virtual int ann_wait();
    
    virtual int read(void *nnn);
    virtual int write(void *nnn, int len);
    
    virtual void set_state(int st);
    virtual int get_state();
    
    virtual void lock();
    virtual void unlock();
    
  protected:
    //virtual int ann_excitation_function(double );
    
  private:
    static int s_ann_calculate(nanmath::nanmath_vector &input, nanmath::nanmath_vector &target,
                               nanmath::nanmath_vector &output);
    static int s_ann_layerforward();
    static int s_ann_forward();
    
    
  protected:
    /*
     * 用于配置的临时数据
     */
    nanai_ann_nanndesc _configure_desc;
    
  protected:
    /*
     * 用来描述神经网络
     */
    std::string _alg;
    int _ninput;
    int _nhidden;
    int _noutput;
    std::vector<int> _nneural;
    
  protected:
    /*
     * 临时数据
     */
    std::string _task;
    nanmath::nanmath_vector _input;
    nanmath::nanmath_vector _output;
    std::vector<nanmath::nanmath_matrix> _ann;
    
  protected:
    /*
     * 状态队列
     */
    std::vector<int> _stlist;
    pthread_mutex_t *_stlist_lock;
    pthread_attr_t _thread_attr_worker;
    pthread_t _thread_worker;
    int _state;
    int _sleep_time;
    
  protected:
    /*
     * 计算函数
     */
    //std::vector<fptr_netlayer_infilter> _netlayer_infilters;
    //std::vector<fptr_netlayer_outfilter> _netlayer_outfilters;
    //std::vector<fptr_netlayer_init> _netlayer_inits;
    //std::vector<fptr_netlayer_error> _netlayer_errors;
    //fptr_ann_outfilter _target_outfilter;
    
  protected:
    /*
     * 回调函数
     */
    fptr_ann_monitor_except _callback_monitor_except;
    fptr_ann_monitor_trained _callback_monitor_trained;
    fptr_ann_monitor_calculated _callback_monitor_calculated;
    fptr_ann_monitor_progress _callback_monitor_progress;
    
  protected:
    /*
     * 日志纪录
     */
    FILE *_log_file;
    std::string _log_dir;
    
  private:
    /*
     * 相关标示
     */
    int _cid;
    time_t _birthday;
    pthread_mutex_t *_lock;
    
  };
  
}

#endif /* nanai_ann_nanncalc_h */
