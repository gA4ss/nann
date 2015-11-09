#ifndef nanai_ann_nanncalc_h
#define nanai_ann_nanncalc_h

#include <stdio.h>
#include <pthread.h>
#include <time.h>
#include <string>
#include <vector>
#include <queue>

#include <nanmath_matrix.h>
#include <nanai_object.h>
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
#define NANNCALC_ST_ANN_EXCHANGING         6
#define NANNCALC_ST_ANN_EXCHANGED          7
  
  /*
   * 命令
   */
#define NANNCALC_CMD_STOP                   0
#define NANNCALC_CMD_WAITING                1
#define NANNCALC_CMD_TRAINING               2
#define NANNCALC_CMD_TRAINING_NOTARGET      3
#define NANNCALC_CMD_CALCULATE              NANNCALC_CMD_TRAINING_NOTARGET
#define NANNCALC_CMD_TRAINING_NOOUTPUT      4
#define NANNCALC_CMD_CONFIGURE              5
#define NANNCALC_CMD_ANN_EXCHANGE           6
  
  /*
   * 进度
   */
#define NANNCALC_PROCESS_LOG                1
  
  class nanai_ann_nanncalc : public nanai_object {
  public:
    /* 
     * 内部类
     */
    class ann_t {
    public:
      ann_t();
      ann_t(std::vector<nanmath::nanmath_matrix> &wm,
            std::vector<nanmath::nanmath_matrix> *dwm=nullptr);
      virtual ~ann_t();
    public:
      int make(std::vector<nanmath::nanmath_matrix> &wm,
               std::vector<nanmath::nanmath_matrix> *dwm=nullptr);
      /*! 有些算法不填充每层神经元个数nneural，这里负责填充
       
          按照weight_matrixes进行填充，如果weight_matrixes为空，则清空nneural变量
       
       */
      void fill_nneural();
      void clear();
      
    public:
      size_t ninput;                                              /* 输入层个数 */
      size_t nhidden;                                             /* 隐藏层个数 */
      size_t noutput;                                             /* 输出层个数 */
      std::vector<size_t> nneural;                                /* 每个隐藏层有多少个神经元 */
      std::vector<nanmath::nanmath_matrix> weight_matrixes;       /* 权值矩阵 */
      std::vector<nanmath::nanmath_matrix> delta_weight_matrixes; /* 权值误差矩阵 */
    };
    
  public:
    nanai_ann_nanncalc(nanai_ann_nanndesc &desc,
                       const char *lp="./",
                       const char *task=nullptr);
    virtual ~nanai_ann_nanncalc();
    
  private:
    /*！禁止拷贝构造函数 */
    nanai_ann_nanncalc(const nanai_ann_nanncalc &t);
    /*！禁止拷贝构造函数 */
    nanai_ann_nanncalc& operator=(const nanai_ann_nanncalc &t);
    
  public:
    virtual void ann_default_create(int ninput,
                                    int nhidden,
                                    int output,
                                    std::vector<int> &nneure);
    
    virtual void ann_training(nanmath::nanmath_vector &input,
                              nanmath::nanmath_vector &target,
                              const char *task=NULL);
    
    virtual void ann_training_notarget(nanmath::nanmath_vector &input,
                                       const char *task=NULL);
    
    #define ann_calculating   ann_training_notarget
    virtual void ann_training_nooutput(nanmath::nanmath_vector &input,
                                       nanmath::nanmath_vector &target,
                                       const char *task=NULL);
    
    virtual void ann_configure(nanai_ann_nanndesc &desc);
    virtual void ann_exchange(const nanai_ann_nanncalc::ann_t &ann);
    virtual void ann_stop();
    virtual void ann_destroy();
    virtual void ann_wait(int st=NANNCALC_ST_STOP,
                          int slt=100);
    virtual nanai_ann_nanncalc::ann_t ann_get();
    
    /* 返回当前计算结点信息 */
  public:
    std::string get_task_name() const;
    std::string get_alg_name() const;
    size_t get_cmdlist_count() const;
    
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
      nanai_ann_nanncalc::ann_t ann;
    };
    
    virtual void set_cmd(struct ncommand &ncmd,
                         bool lock=false);
    virtual int get_cmd(struct ncommand &ncmd,
                        bool lock=false);
    virtual void set_state(int st,
                           bool lock=false);
    virtual int get_state(bool lock=false);
    virtual void set_output(nanmath::nanmath_vector &output,
                            bool lock=false);
    virtual size_t get_output(nanmath::nanmath_vector &output,
                              bool lock=false);
    virtual void do_configure(nanai_ann_nanndesc &desc);
    virtual void do_ann_exchange(const nanai_ann_nanncalc::ann_t &ann);
    virtual nanai_ann_nanncalc::ann_t get_ann(bool lock=false);
    virtual void do_stop();
    
  public:
    virtual void ann_calculate(const char *task,
                               nanmath::nanmath_vector &input,
                               nanmath::nanmath_vector *target,
                               nanmath::nanmath_vector *output);
  protected:
    virtual void ann_forward(nanmath::nanmath_vector &input,
                             nanmath::nanmath_vector &output);
    
    virtual void ann_layer_forward(int h,
                                   nanmath::nanmath_vector &l1,
                                   nanmath::nanmath_vector &l2,
                                   nanmath::nanmath_matrix &wm,
                                   fptr_ann_hidden_calc calc);
    virtual nanmath::nanmath_vector ann_output_error(nanmath::nanmath_vector &target,
                                                     nanmath::nanmath_vector &output);
    virtual void ann_hiddens_error(nanmath::nanmath_vector &input,
                                   nanmath::nanmath_vector &output_delta);
    
    virtual nanmath::nanmath_vector ann_calc_hidden_delta(size_t h,
                                                          nanmath::nanmath_vector &delta_k,
                                                          nanmath::nanmath_matrix &w_kh,
                                                          nanmath::nanmath_vector &o_h);
    
    virtual void ann_adjust_weight(size_t h,
                                   nanmath::nanmath_vector &layer,
                                   nanmath::nanmath_vector &delta);
    
  protected:
    /* 基类的虚函数 */
    virtual void on_error(int err);
    
  protected:
    virtual void ann_create(nanai_ann_nanndesc &desc);
    virtual void ann_on_except(int err);
    virtual void ann_on_trained();
    virtual void ann_on_trained_nooutput();
    virtual void ann_on_calculated();
    virtual void ann_on_alg_uninstall();
    virtual void ann_process(int process, void *arg);
    virtual void ann_log(const char *fmt, ...);
    
  private:
    static void *thread_nanai_ann_worker(void *arg);
    
  protected:
    std::string _alg;                                            /* 算法名称 */
  
  protected:
    /*
     * 临时数据
     */
    std::string _task;                                            /* 当前任务名称 - 例如:Aid_Uid_ACTid */
    std::vector<nanmath::nanmath_vector> _hiddens;                /* 隐藏层 */
    pthread_mutex_t _outputs_lock;                                /* 输出向量锁 */
    std::vector<nanmath::nanmath_vector> _outputs;                /* 输出向量队列 */
    nanmath::nanmath_vector _output_error;                        /* 输出误差向量 */
    
    /* 人工神经网络结果 */
    pthread_mutex_t _ann_lock;
    ann_t _ann;                                                   /* 神经网络 */
    
  protected:
    /*
     * 状态与命令
     */
    std::queue<struct ncommand> _cmdlist;                         /* 命令队列 */
    pthread_mutex_t _cmdlist_lock;                                /* 命令互斥锁 */
    //pthread_attr_t _thread_attr_worker;
    pthread_t _thread_worker;                                     /* 线程函数 */
    pthread_mutex_t _state_lock;
    int _state;                                                   /* 当前状态 */
    int _command;                                                 /* 命令 */
    int _sleep_time;                                              /* 线程睡眠时间 */
    
  protected:
    /*
     * 计算函数
     */
    fptr_ann_input_filter _fptr_input_filter;
    fptr_ann_result _fptr_result;
    fptr_ann_output_error _fptr_output_error;
    fptr_ann_calculate _fptr_calculate;
    
    fptr_ann_hidden_init _fptr_hidden_inits;
    fptr_ann_hidden_calc _fptr_hidden_calcs;
    fptr_ann_hidden_error _fptr_hidden_errors;
    fptr_ann_hidden_adjust_weight _fptr_hidden_adjust_weights;
    
  protected:
    /*
     * 回调函数
     */
    fptr_ann_monitor_except _callback_monitor_except;
    fptr_ann_monitor_trained _callback_monitor_trained;
    fptr_ann_monitor_trained2 _callback_monitor_trained_nooutput;
    fptr_ann_monitor_calculated _callback_monitor_calculated;
    fptr_ann_monitor_progress _callback_monitor_progress;
    fptr_ann_monitor_alg_uninstall _callback_monitor_alg_uninstall;
    
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
