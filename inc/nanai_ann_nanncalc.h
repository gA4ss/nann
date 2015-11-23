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
   * 计算结点状态消息
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
#define NANNCALC_CMD_CONFIGURE              3
  
  /*
   * 进度
   */
#define NANNCALC_PROCESS_LOG                1
#define NANNCALC_PROCESS_CREATE             2
#define NANNCALC_PROCESS_DESTROY            3
  
  /*! nann计算结点类
   
      一个类就是一条工作线程，负责调用算法进行训练，此类实现了神经网络的前馈与反馈的逻辑。但不直接
      进行运算，而是将运算部分交给实际配置的desc算法描述结果，来执行回调。
      一次只能设定一种任务。
   */
  class nanai_ann_nanncalc : public nanai_object {
  public:
    /*! 内部类，实现描述神经网络结构 */
    class ann_t {
    public:
      /*! 神经网络构造函数 */
      ann_t();
      /*! 带参数的神经网络构造函数 */
      ann_t(const std::string &alg,                                     /*!< 算法名称 */
            std::vector<nanmath::nanmath_matrix> &wm,                   /*!< 权值矩阵，会根据这个矩阵产生神经网络其它配置 */
            std::vector<nanmath::nanmath_matrix> *dwm=nullptr           /*!< 可选的偏差矩阵 */
            );
      
      ann_t(const ann_t &t);
      
      /*! 神经网络析构函数 */
      virtual ~ann_t();
    public:
      /*! 设置神经网络 */
      void set(const ann_t &t);
      /*! 根据权值矩阵产生神经网络 */
      int make(const std::string &alg,                                  /*!< 算法名称 */
               std::vector<nanmath::nanmath_matrix> &wm,                /*!< 权值矩阵，会根据这个矩阵产生神经网络其它配置 */
               std::vector<nanmath::nanmath_matrix> *dwm=nullptr        /*!< 可选的偏差矩阵 */
               );
      
      /*! 清除网络数据 */
      void clear();
      /*! 是否为空 */
      bool empty() const;
      
      /*! 创建隐藏层 */
      std::vector<nanmath::nanmath_vector> create_hidden_layers();

      /*! 填充神经元个数 */
      void fill_nneural();
      
    public:
      std::string alg;                                            /*!< 算法名称 */
      size_t ninput;                                              /*!< 输入层个数 */
      size_t nhidden;                                             /*!< 隐藏层个数 */
      size_t noutput;                                             /*!< 输出层个数 */
      std::vector<size_t> nneural;                                /*!< 每个隐藏层有多少个神经元 */
      std::vector<nanmath::nanmath_matrix> weight_matrixes;       /*!< 权值矩阵 */
      std::vector<nanmath::nanmath_matrix> delta_weight_matrixes; /*!< 权值误差矩阵 */
    };
    
    /*! 计算结果类型 */
    typedef std::pair<nanmath::nanmath_vector, nanai_ann_nanncalc::ann_t> result_t;
    
  public:
    /*! 计算结点构造函数 */
    nanai_ann_nanncalc(const nanai_ann_nanndesc &desc,    /*!< 算法描述结构 */
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
    /*! 训练函数 */
    virtual void ann_training(const std::string &task,                            /*!< [in] 任务名称 */
                              const nanmath::nanmath_vector &input,               /*!< [in] 输入向量 */
                              const nanmath::nanmath_vector &target,              /*!< [in] 目标向量 */
                              const nanai_ann_nanncalc::ann_t &ann,               /*!< [in] 神经网络 */
                              result_t *result                                    /*!< [in] 计算完毕后输出到哪里 */
                              );
    
    /*! 替换神经网络配置 */
    virtual void ann_configure(const nanai_ann_nanndesc &desc             /*!< 算法描述结构 */
                               );
    
    /*! 停止当前的计算线程 */
    virtual void ann_stop();
    
    /*! 销毁当前的计算结点的内部数据与内存 */
    virtual void ann_destroy();
    
    /*! 等待计算结点到某个状态 */
    virtual void ann_wait(int st=NANNCALC_ST_WAITING,     /*!< 要等待的状态 */
                          int slt=100                     /*!< 要等待的时间 */
                          );
    
  public:
    /*! 返回当前算法名称 */
    virtual std::string get_alg_name() const;
    /*! 返回当前命令栈中的计数 */
    virtual size_t get_cmdlist_count() const;
    /*! 获取状态 */
    virtual int get_state() const;
    
  public:
    /*! 命令格式 */
    struct ncommand {
      int cmd;
      nanmath::nanmath_vector input;
      nanmath::nanmath_vector target;
      std::string task;
      nanai_ann_nanndesc desc;
      ann_t ann;
      result_t *result;
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
    virtual void set_state(int st                                 /*!< [in] 状态 */
                          );
    
    /*! 内部替换配置
        \warning 不要由外部调用，这是支持多线程函数的
     */
    virtual void do_configure(nanai_ann_nanndesc &desc                      /*!< [in] 算法结构 */
                              );
    
    /*! 设置训练好的结果
        \warning 不要由外部调用，这是支持多线程函数的
     */
    
    virtual void set_result(nanai_ann_nanncalc::result_t *result,                   /*!< [out] 输出结果 */
                            const nanmath::nanmath_vector &output,                  /*!< [in] 计算结果 */
                            const nanai_ann_nanncalc::ann_t &ann                    /*!< [in] 神经网络 */
                            );
    
    /*! 内部停止线程
        \warning 不要由外部调用，这是支持多线程函数的
     */
    virtual void do_stop();
    
  public:
    /*! 神经网络训练 */
    virtual void ann_calculate(const std::string &task,                     /*!< [in] 任务名称 */
                               nanmath::nanmath_vector &input,              /*!< [in] 输入向量 */
                               nanmath::nanmath_vector &target,             /*!< [in] 目标向量 */
                               nanmath::nanmath_vector &output,             /*!< [in] 输入向量 */
                               nanai_ann_nanncalc::ann_t &ann               /*!< [in] 输入的神经网络 */
                               );
  protected:
    /*! 网络向前运算 */
    virtual void ann_forward(nanmath::nanmath_vector &input,                /*!< [in] 输入向量 */
                             nanai_ann_nanncalc::ann_t &ann,                /*!< [in] 神经网络 */
                             std::vector<nanmath::nanmath_vector> &hiddens, /*!< [in] 隐藏层 */
                             nanmath::nanmath_vector &output                /*!< [in] 输出向量 */
                             );
    
    /*! 网络层向前运算 */
    virtual void ann_layer_forward(int h,                                   /*!< [in] 隐藏层索引 */
                                   nanmath::nanmath_vector &l1,             /*!< [in] 输入层向量 */
                                   nanmath::nanmath_vector &l2,             /*!< [in] 输出层向量 */
                                   nanmath::nanmath_matrix &wm,             /*!< [in] 权值矩阵 */
                                   fptr_ann_alg_hidden_calc calc            /*!< [in] 计算函数指针 */
                                   );
    
    /*! 输出误差调整 */
    virtual nanmath::nanmath_vector ann_output_error(nanmath::nanmath_vector &target,       /*!< [in] 目标向量 */
                                                     nanmath::nanmath_vector &output        /*!< [out] 输出向量 */
                                                     );
    /*! 隐藏层误差调整 */
    virtual void ann_hiddens_error(nanai_ann_nanncalc::ann_t &ann,                  /*!< [in] 神经网络 */
                                   std::vector<nanmath::nanmath_vector> &hiddens,   /*!< [in] 隐藏层 */
                                   nanmath::nanmath_vector &input,                  /*!< [in] 输入向量 */
                                   nanmath::nanmath_vector &output_delta            /*!< [out] 误差输出向量 */
                                   );
  protected:
    /*! 基类的虚函数，当错误发生是调用 */
    virtual void on_error(int err     /*!< 错误发生时的代码 */
                          );
    
  protected:
    virtual void ann_create(const nanai_ann_nanndesc &desc);
    virtual void ann_on_except(int err);
    virtual void ann_on_trained(nanmath::nanmath_vector &input,
                                nanmath::nanmath_vector &target,
                                nanmath::nanmath_vector &output,
                                nanai::nanai_ann_nanncalc::ann_t &ann);
    virtual void ann_on_alg_uninstall();
    virtual void ann_process(int process, void *arg);
    
  protected:
    virtual void ann_log(const char *fmt, ...);
    
    /*! 打印创建日志 */
    virtual void ann_log_create();

  private:
    static void *thread_nanai_ann_worker(void *arg);
  
  protected:
    std::string _task;                                              /*!< 当前任务名称 - 例如:Aid_Uid_ACTid */

  protected:
    std::queue<struct ncommand> _cmdlist;                           /*!< 命令队列 */
    pthread_mutex_t _cmdlist_lock;                                  /*!< 命令互斥锁 */
    pthread_t _thread_worker;                                       /*!< 线程函数 */

    int _state;                                                     /*!< 当前状态 */
    int _command;                                                   /*!< 命令 */
    
  protected:
    std::string _alg;                                               /*!< 算法名称 */
    
    fptr_ann_alg_input_filter _fptr_input_filter;                   /*!< 输入初始化过滤回调 */
    fptr_ann_alg_result _fptr_result;                               /*!< 结果过滤回调 */
    fptr_ann_alg_output_error _fptr_output_error;                   /*!< 输出误差调整回调 */
    fptr_ann_alg_calculate _fptr_calculate;                         /*!< 计算回调 */
    
    fptr_ann_alg_hidden_calc _fptr_hidden_calcs;                    /*!< 计算每个隐藏层的值 */
    fptr_ann_alg_hidden_error _fptr_hidden_errors;                  /*!< 隐藏层的误差调整 */
    fptr_ann_alg_hidden_adjust_weight _fptr_hidden_adjust_weights;  /*!< 调整隐藏层权值 */
    
    fptr_ann_monitor_except _callback_monitor_except;               /*!< 当发生异常时调用 */
    fptr_ann_monitor_trained _callback_monitor_trained;             /*!< 当发生训练时调用 */
    fptr_ann_monitor_progress _callback_monitor_progress;           /*!< 进程控制调用 */
    fptr_ann_monitor_alg_uninstall _callback_monitor_alg_uninstall; /*!< 当发生卸载时调用 */
    
  protected:
    FILE *_log_file;                                                /*!< 日志文件 */
    std::string _log_dir;                                           /*!< 日志目录 */
    
  private:
    int _cid;                                                       /*!< 当前的计算结点ID */
    time_t _birthday;                                               /*!< 生成时间 */
  };
  
}

#endif /* nanai_ann_nanncalc_h */
