#ifndef nanai_ann_nanncalc_h
#define nanai_ann_nanncalc_h

#include <stdio.h>
#include <pthread.h>
#include <time.h>
#include <string>
#include <vector>
#include <queue>

#include <nanmath_vector.h>
#include <nanmath_matrix.h>
#include <nanai_ann_nanndesc.h>

namespace nanai {
  
  /*
   * 状态消息
   */
#define NANNCALC_ST_STOP                   0
#define NANNCALC_ST_WAITING                1
#define NANNCALC_ST_TRAINED                2
#define NANNCALC_ST_TRAINING               3
#define NANNCALC_ST_CONFIGURED             4
#define NANNCALC_ST_CONFIGURING            5
  
  /*
   * 命令
   */
#define NANNCALC_CMD_STOP                   0
#define NANNCALC_CMD_WAITING                1
#define NANNCALC_CMD_TRAINING               2
#define NANNCALC_CMD_TRAINING_NOOUTPUT      3
#define NANNCALC_CMD_TRAINING_NOTARGET      4
#define NANNCALC_CMD_CALCULATE              NANNCALC_CMD_TRAINING_NOTARGET
#define NANNCALC_CMD_CONFIGURE              5
  
  /*
   * 进度
   */
#define NANNCALC_PROCESS_CREATE             0
#define NANNCALC_PROCESS_TRAINING           1
#define NANNCALC_PROCESS_TRAINING_NOTARGET  2
#define NANNCALC_PROCESS_CONFIGURE          3
#define NANNCALC_PROCESS_STOP               4
#define NANNCALC_PROCESS_DESTROY            5
#define NANNCALC_PROCESS_LOG                6
  
  class nanai_ann_nanncalc {
  public:
    nanai_ann_nanncalc(const char *lp=NULL);
    virtual ~nanai_ann_nanncalc();
    
  public:
    virtual int ann_training(nanmath::nanmath_vector &input, nanmath::nanmath_vector &target,
                             const char *task=NULL);
    virtual int ann_training_notarget(nanmath::nanmath_vector &input,
                                      const char *task=NULL);
    
#define ann_calculate       ann_training_notarget
    virtual int ann_configure(nanai_ann_nanndesc &desc);
    virtual int ann_output(nanmath::nanmath_vector &output);
    
    virtual int ann_stop();
    virtual int ann_destroy();
    virtual int ann_wait(int st, int slt=100);
    
    virtual int read(void *nnn);
    virtual int write(void *nnn, int len);
    
  public:
    /* WARNING: 以下这些公共函数不要由外部调用，这是支持多线程
     * 函数的。
     */
    
    struct ncommand {
      int cmd;
      nanmath::nanmath_vector input;
      nanmath::nanmath_vector target;
      std::string task;
      nanai_ann_nanndesc desc;
    };
    
    virtual void set_cmd(struct ncommand &ncmd);
    virtual int get_cmd(struct ncommand &ncmd);
    virtual void set_state(int st);
    virtual int get_state();
    
    virtual void set_output(nanmath::nanmath_vector &output);
    
  protected:
    virtual int ann_create(nanai_ann_nanndesc &desc);
    virtual void ann_except(int err);
    virtual void ann_trained();
    virtual void ann_calculated();
    virtual void ann_process(int process, void *arg);
    virtual void ann_log(const char *fmt, ...);
    //virtual int ann_excitation_function(double );
    
  private:
    static int s_ann_calculate(nanmath::nanmath_vector &input, nanmath::nanmath_vector &target,
                               nanmath::nanmath_vector &output);
    static int s_ann_layerforward();
    static int s_ann_forward();
    
  private:
    static void *thread_nanai_ann_worker(void *arg);
    
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
    std::vector<nanmath::nanmath_matrix> _ann;
    std::vector<nanmath::nanmath_vector> _outputs;
    
  protected:
    /*
     * 状态与命令
     */
    
    std::queue<struct ncommand> _cmdlist;
    pthread_mutex_t *_cmdlist_lock;
    //pthread_attr_t _thread_attr_worker;
    pthread_t _thread_worker;
    int _state;
    int _command;
    int _sleep_time;
    
  protected:
    /*
     * 计算函数
     */
    fptr_ann_result _taret_outfilter;
    std::vector<fptr_ann_hidden_init> _hidden_init;
    std::vector<fptr_ann_hidden_calc> _hidden_calc;
    std::vector<fptr_ann_hidden_error> _hidden_error;
    
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
  };
  
}

#endif /* nanai_ann_nanncalc_h */
