#ifndef nanai_ann_nanncalc_h
#define nanai_ann_nanncalc_h

#include <stdio.h>
#include <pthread.h>
#include <time.h>
#include <string>
#include <map>
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
  /*! nann计算结点类
   
      一个类就是一条工作线程，负责调用算法进行训练，此类实现了神经网络的前馈与反馈的逻辑。但不直接
      进行运算，而是将运算部分交给实际配置的desc算法描述结果，来执行回调。
   */
  class nanai_ann_nanncalc : public nanai_object {
  public:
    /*! 内部类，实现描述神经网络结构 */
    class ann_t {
    public:
      /*! 神经网络构造函数 */
      ann_t();
      /*! 带参数的神经网络构造函数 */
      ann_t(std::vector<nanmath::nanmath_matrix> &wm,                   /*!< 权值矩阵，会根据这个矩阵产生神经网络其它配置 */
            std::vector<nanmath::nanmath_matrix> *dwm=nullptr           /*!< 可选的偏差矩阵 */
            );
      /*! 神经网络析构函数 */
      virtual ~ann_t();
    public:
      /*! 根据权值矩阵产生神经网络 */
      int make(std::vector<nanmath::nanmath_matrix> &wm,                /*!< 权值矩阵，会根据这个矩阵产生神经网络其它配置 */
               std::vector<nanmath::nanmath_matrix> *dwm=nullptr        /*!< 可选的偏差矩阵 */
               );
      
      /*! 有些算法不填充每层神经元个数nneural，这里负责填充
       
          按照weight_matrixes进行填充，如果weight_matrixes为空，则清空nneural变量
       
       */
      void fill_nneural();
      
      /*! 清除网络数据 */
      void clear();
      
    public:
      size_t ninput;                                              /*!< 输入层个数 */
      size_t nhidden;                                             /*!< 隐藏层个数 */
      size_t noutput;                                             /*!< 输出层个数 */
      std::vector<size_t> nneural;                                /*!< 每个隐藏层有多少个神经元 */
      std::vector<nanmath::nanmath_matrix> weight_matrixes;       /*!< 权值矩阵 */
      std::vector<nanmath::nanmath_matrix> delta_weight_matrixes; /*!< 权值误差矩阵 */
    };
    
  public:
    /*! 计算结点构造函数 */
    nanai_ann_nanncalc(nanai_ann_nanndesc &desc,          /*!< 算法描述结构 */
                       const std::string &task,           /*!< 当前默认的任务名 */
                       const char *lp="./"                /*!< 配置目录 */
                       );
    
    /*! 计算结点析构函数 */
    virtual ~nanai_ann_nanncalc();
    
  private:
    /*！禁止拷贝构造函数 */
    nanai_ann_nanncalc(const nanai_ann_nanncalc &t);
    /*！禁止拷贝构造函数 */
    nanai_ann_nanncalc& operator=(const nanai_ann_nanncalc &t);
    
  public:
    /*! 创建默认的神经网络 */
    virtual void ann_default_create(int ninput,                   /*!< 输入层向量个数 */
                                    int nhidden,                  /*!< 隐藏层数量 */
                                    int output,                   /*!< 输出层向量个数 */
                                    std::vector<int> &nneure      /*!< 隐藏层每层神经元个数 */
                                    );
    
    /*! 进行训练，有输出向量，调解权值 */
    virtual void ann_training(nanmath::nanmath_vector &input,     /*!< 输入向量 */
                              nanmath::nanmath_vector &target,    /*!< 目标向量 */
                              const std::string &task             /*!< 任务名 */
                              );
    /*! 进行训练，有输出向量，不调解权值 */
    virtual void ann_training_notarget(nanmath::nanmath_vector &input,    /*!< 输入向量 */
                                       const std::string &task            /*!< 任务名 */
                                       );
    
    /*! 无调解权值训练的别名 */
    #define ann_calculating   ann_training_notarget
    /*! 进行训练，有输出向量，不调解权值 */
    virtual void ann_training_nooutput(nanmath::nanmath_vector &input,    /*!< 输入向量 */
                                       nanmath::nanmath_vector &target,   /*!< 目标向量 */
                                       const std::string &task            /*!< 任务名 */
                                       );
    /*! 替换神经网络配置 */
    virtual void ann_configure(const nanai_ann_nanndesc &desc             /*!< 算法描述结构 */
                               );
    
    /*! 替换神经网络 */
    virtual void ann_exchange(const nanai_ann_nanncalc::ann_t &ann        /*!< 神经网络 */
                              );
    
    /*! 停止当前的计算线程 */
    virtual void ann_stop();
    
    /*! 销毁当前的计算结点的内部数据与内存 */
    virtual void ann_destroy();
    
    /*! 等待计算结点到某个状态 */
    virtual void ann_wait(int st=NANNCALC_ST_STOP,        /*!< 要等待的状态 */
                          int slt=100                     /*!< 要等待的时间 */
                          );
    
    /*! 获取当前计算结点内部的神经网络 */
    virtual nanai_ann_nanncalc::ann_t ann_get();
    
  public:
    /*! 返回当前任务名称 */
    std::string get_task_name() const;
    /*! 返回当前算法名称 */
    std::string get_alg_name() const;
    /*! 返回当前命令栈中的计数 */
    size_t get_cmdlist_count() const;
    
  public:
    /*! 命令格式 */
    struct ncommand {
      int cmd;
      nanmath::nanmath_vector input;
      nanmath::nanmath_vector target;
      std::string task;
      nanai_ann_nanndesc desc;
      nanai_ann_nanncalc::ann_t ann;
    };
    
    /*! 设置命令
     
        \warning 不要由外部调用，这是支持多线程函数的
     */
    virtual void set_cmd(struct ncommand &ncmd,                  /*! [in] 命令结构 */
                         bool lock=false                         /*! [in] 命令锁 */
                         );
    
    /*! 获取命令
     
        \warning 不要由外部调用，这是支持多线程函数的
     */
    virtual int get_cmd(struct ncommand &ncmd,                    /*! [in] 命令结构 */
                        bool lock=false                           /*! [in] 命令锁 */
                        );
    
    /*! 设置状态
     
        \warning 不要由外部调用，这是支持多线程函数的
     */
    virtual void set_state(int st,                                /*!< [in] 状态 */
                           bool lock=false                        /*!< [in] 线程锁 */
                           );
    
    /*! 获取状态
     
        \warning 不要由外部调用，这是支持多线程函数的
     */
    virtual int get_state(bool lock=false                          /*!< [in] 线程锁 */
                          );
    
    /*! 设置指定任务的输出向量
     
        \warning 不要由外部调用，这是支持多线程函数的
     */
    virtual void set_output(std::string &task,                      /*!< [in] 任务名 */
                            nanmath::nanmath_vector &output,        /*!< [in] 要设置的输出向量 */
                            bool lock=false                         /*!< [in] 是否进行输出锁控制 */
                            );
    
    /*! 获取指定任务的最后一个输出向量 */
    virtual nanmath::nanmath_vector get_output(std::string &task,   /*!< [in] 任务名 */
                                               bool lock=false,     /*!< [in] 是否进行输出锁控制 */
                                               bool not_pop=false   /*!< [in] 是否取走 */
                                               );
    
    /*! 按照匹配进行搜索输出 */
    typedef std::pair<std::string, nanmath::nanmath_vector> task_output_t;
    virtual std::vector<task_output_t> get_matched_outputs(std::string &rstr,     /*!< [in] 正则表达式 */
                                                           bool lock=false,       /*!< [in] 是否进行输出锁控制 */
                                                           bool not_pop=false     /*!< [in] 是否取走 */
                                                           );
    /*! 内部替换配置
     
        \warning 不要由外部调用，这是支持多线程函数的
     */
    virtual void do_configure(nanai_ann_nanndesc &desc                      /*!< [in] 算法结构 */
                              );
    
    /*! 内部替换神经网络
     
        \warning 不要由外部调用，这是支持多线程函数的
     */
    virtual void do_ann_exchange(const nanai_ann_nanncalc::ann_t &ann       /*!< [in] 要替换的神经网络 */
                                 );
    
    /*! 获取指定任务神经网络 */
    virtual nanai_ann_nanncalc::ann_t get_ann(const std::string &task,      /*!< [in] 任务名 */
                                              bool lock=false               /*!< [in] 获取神经网络锁 */
                                              );
    
    
    /*! 设置训练好的神经网络
     
        \warning 不要由外部调用，这是支持多线程函数的
     */
    virtual void set_ann(std::string &task,                       /*!< [in] 任务名 */
                         bool lock=false                          /*!< [in] 是否进行输出锁控制 */
                        );
    
    /*! 按照匹配进行搜索神经网络
     
        \warning 不要由外部调用，这是支持多线程函数的
     */
    typedef std::pair<std::string, nanai_ann_nanncalc::ann_t> task_ann_t;
    virtual std::vector<task_ann_t> get_matched_anns(std::string &rstr,     /*!< [in] 正则表达式 */
                                                     bool lock=false,       /*!< [in] 是否进行输出锁控制 */
                                                     bool not_pop=false     /*!< [in] 是否取走 */
                                                    );
    
    /*! 内部停止线程
     
        \warning 不要由外部调用，这是支持多线程函数的
     */
    virtual void do_stop();
    
  public:
    /*! 神经网络训练 */
    virtual void ann_calculate(const std::string &task,                     /*!< [in] 任务名 */
                               nanmath::nanmath_vector &input,              /*!< [in] 输入向量 */
                               nanmath::nanmath_vector *target,             /*!< [in] 目标向量 */
                               nanmath::nanmath_vector *output              /*!< [in] 输入向量 */
                               );
  protected:
    /*! 网络向前运算  */
    virtual void ann_forward(nanmath::nanmath_vector &input,                /*!< [in] 输入向量 */
                             nanmath::nanmath_vector &output                /*!< [in] 输出向量 */
                             );
    
    /*! 网络层向前运算 */
    virtual void ann_layer_forward(int h,                                   /*!< [in] 隐藏层索引 */
                                   nanmath::nanmath_vector &l1,             /*!< [in] 输入层向量 */
                                   nanmath::nanmath_vector &l2,             /*!< [in] 输出层向量 */
                                   nanmath::nanmath_matrix &wm,             /*!< [in] 权值矩阵 */
                                   fptr_ann_hidden_calc calc                /*!< [in] 计算函数指针 */
                                   );
    
    /*! 输出误差调整 */
    virtual nanmath::nanmath_vector ann_output_error(nanmath::nanmath_vector &target,       /*!< [in] 目标向量 */
                                                     nanmath::nanmath_vector &output        /*!< [out] 输出向量 */
                                                     );
    
    /*! 隐藏层误差调整 */
    virtual void ann_hiddens_error(nanmath::nanmath_vector &input,                  /*!< [in] 输入向量 */
                                   nanmath::nanmath_vector &output_delta            /*!< [out] 误差输出向量 */
                                   );
    /*! 具体隐藏层误差调整 */
    virtual nanmath::nanmath_vector ann_calc_hidden_delta(size_t h,                         /*!< [in] 隐藏层索引 */
                                                          nanmath::nanmath_vector &delta_k, /*!< [in] 偏差向量 */
                                                          nanmath::nanmath_matrix &w_kh,    /*!< [in] 要调解的权值矩阵 */
                                                          nanmath::nanmath_vector &o_h      /*!< [in] 输出的向量 */
                                                          );
    /*! 两个层之间的权值矩阵调解 */
    virtual void ann_adjust_weight(size_t h,                          /*!< [in] 隐藏层索引 */
                                   nanmath::nanmath_vector &layer,    /*!< [in] 输入向量 */
                                   nanmath::nanmath_vector &delta     /*!< [in] 误差向量 */
                                   );
    
  protected:
    /*! 基类的虚函数，当错误发生是调用 */
    virtual void on_error(int err     /*!< 错误发生时的代码 */
                          );
    
  protected:
    virtual void ann_create(nanai_ann_nanndesc &desc);
    virtual void ann_on_except(int err);
    virtual void ann_on_trained();
    virtual void ann_on_trained_nooutput();
    virtual void ann_on_calculated();
    virtual void ann_on_alg_uninstall();
    virtual void ann_process(int process, void *arg);
    
  protected:
    virtual void ann_log(const char *fmt, ...);
    
    /*! 打印创建日志 */
    virtual void ann_log_create();
    
  private:
    static void *thread_nanai_ann_worker(void *arg);
    
  protected:
    std::string _alg;                                             /*!< 算法名称 */
  
  protected:
    std::string _task;                                            /*!< 当前任务名称 - 例如:Aid_Uid_ACTid */
    std::vector<nanmath::nanmath_vector> _hiddens;                /*!< 隐藏层 */
    pthread_mutex_t _outputs_lock;                                /*!< 输出向量锁 */
    typedef std::map<std::string, nanmath::nanmath_vector> nanncalc_output_t;
    nanncalc_output_t _outputs;                                   /*!< 输出向量队列 */
    typedef std::map<std::string, nanai_ann_nanncalc::ann_t> nanncalc_ann_t;
    nanncalc_ann_t _anns;
    
    pthread_mutex_t _ann_lock;                                    /*!< 神经网络的锁 */
    ann_t _ann;                                                   /*!< 神经网络 */
    
  protected:
    std::queue<struct ncommand> _cmdlist;                         /*!< 命令队列 */
    pthread_mutex_t _cmdlist_lock;                                /*!< 命令互斥锁 */
    //pthread_attr_t _thread_attr_worker;
    pthread_t _thread_worker;                                     /*!< 线程函数 */
    pthread_mutex_t _state_lock;                                  /*!< 状态锁 */
    int _state;                                                   /*!< 当前状态 */
    int _command;                                                 /*!< 命令 */
    int _sleep_time;                                              /*!< 线程睡眠时间 */
    
  protected:
    fptr_ann_input_filter _fptr_input_filter;
    fptr_ann_result _fptr_result;
    fptr_ann_output_error _fptr_output_error;
    fptr_ann_calculate _fptr_calculate;
    
    fptr_ann_hidden_init _fptr_hidden_inits;
    fptr_ann_hidden_calc _fptr_hidden_calcs;
    fptr_ann_hidden_error _fptr_hidden_errors;
    fptr_ann_hidden_adjust_weight _fptr_hidden_adjust_weights;
    
  protected:
    fptr_ann_monitor_except _callback_monitor_except;
    fptr_ann_monitor_trained _callback_monitor_trained;
    fptr_ann_monitor_trained2 _callback_monitor_trained_nooutput;
    fptr_ann_monitor_calculated _callback_monitor_calculated;
    fptr_ann_monitor_progress _callback_monitor_progress;
    fptr_ann_monitor_alg_uninstall _callback_monitor_alg_uninstall;
    
  protected:
    FILE *_log_file;
    std::string _log_dir;
    
  private:
    int _cid;
    time_t _birthday;
  };
  
}

#endif /* nanai_ann_nanncalc_h */
