#ifndef nanai_ann_nannmgr_h
#define nanai_ann_nannmgr_h

#include <string>
#include <vector>
#include <pthread.h>

#include <nanai_ann_nanncalc.h>

namespace nanai {
  //! 南南人工神经网络管理器类
  /*! 管理器类用于管理所有计算结点线程，为了能做到高并发，并且可以在外部获取到计算的结果，
      这里将一组task分解成n个job，例如一个任务名为:"task"，则当有多个训练样本时，管理器会
      维护一个计数器用来计数job，然后再以task.jid的形式下发给计算结点。计算结点以task.jid
      为任务名进行计算
   */
  class nanai_ann_nannmgr : public nanai_object {
  public:
    /*! 管理器构造函数. */
    nanai_ann_nannmgr(int max=1024,     /*!< [in] 最大计算结点(线程),默认为1024 */
                      int now_start=0   /*!< [in] 当前就要启动的计算结点数 */
                      );
    
    /*! 管理器构造函数. */
    nanai_ann_nannmgr(std::string alg,                  /*!< [in] 要设定的神经网络算法 */
                      nanai_ann_nanncalc::ann_t &ann,   /*!< [in] 要设定的人工神经网络 */
                      nanmath::nanmath_vector *target,  /*!< [in] 要训练的目标向量，可选*/
                      const char *task=nullptr,         /*!< [in] 直接赋予的任务名 */
                      int max=1024,                     /*!< [in] 最大计算结点(线程),默认为1024 */
                      int now_start=0                   /*!< [in] 当前就要启动的计算结点数 */
                      );
    /*! 析构函数.
     
        会向所有正在工作的线程发出停止的命令，并等待完成，最后销毁
     */
    virtual ~nanai_ann_nannmgr();
  protected:
    /*! 初始化所有的数据. */
    virtual void init(int max,          /*!< [in] 最大计算结点(线程),默认为1024 */
                      int now_start,    /*!< [in] 当前就要启动的计算结点数 */
                      const char *task  /*!< [in] 任务名 */
                      );
    
  public:
    
    /*! 输出结果，并调整误差
     
        使用json参数中的“input.json”格式进行训练样本与目标的输入。
        如果指定的计算结点dcalc不为空，则直接使用，随后判断是否指定了算法alg。如果指定了则通过alg
        寻找算法描述结点，如果没有找打则抛出异常“NANAI_ERROR_LOGIC_ALG_NOT_FOUND”。找到了则
        修改当前dcalc计算结点的配置为算法alg的描述结点。如果指定了ann，则应用。
        如果没有指定计算结点，首先会通过alg参数寻找当前管理器是否加载对应的算法描述插件，没有则抛出
        “NANAI_ERROR_LOGIC_ALG_NOT_FOUND”，如果找到，则调用generate产生一个新的计算结点来进行
        训练。
     */
    virtual int training(const std::string &json,                 /*!< [in] input.json格式文件 */
                         nanai_ann_nanncalc *dcalc=nullptr,       /*!< [in] 要直接使用的计算结点，可选 */
                         const char *task=nullptr,                /*!< [in] 任务的名称，可选 */
                         nanai_ann_nanncalc::ann_t *ann=nullptr,  /*!< [in] 人工神经网络，可选 */
                         const char *alg=nullptr                  /*!< [in] 可选用的算法，可选 */
                         );
    
    
    /*! 输出结果，并调整误差 
     
        如果指定的计算结点dcalc不为空，则直接使用，随后判断是否指定了算法alg。如果指定了则通过alg
        寻找算法描述结点，如果没有找打则抛出异常“NANAI_ERROR_LOGIC_ALG_NOT_FOUND”。找到了则
        修改当前dcalc计算结点的配置为算法alg的描述结点。如果指定了ann，则应用。如果指定了target
        则应用当前指定的目标向量，如果没有指定则使用默认的目标向量。
        如果没有指定计算结点，首先会通过alg参数寻找当前管理器是否加载对应的算法描述插件，没有则抛出
        “NANAI_ERROR_LOGIC_ALG_NOT_FOUND”，如果找到，则调用generate产生一个新的计算结点来进行
        训练。如果指定了target则应用当前指定的目标向量，如果没有指定则使用默认的目标向量。
     */
    virtual nanai_ann_nanncalc *training(nanmath::nanmath_vector &input,          /*!< [in] 输入样本向量 */
                                         nanmath::nanmath_vector *target,         /*!< [in] 训练目标向量，可选 */
                                         nanai_ann_nanncalc *dcalc=nullptr,       /*!< [in] 要直接使用的计算结点，可选 */
                                         const char *task=nullptr,                /*!< [in] 任务的名称，可选 */
                                         nanai_ann_nanncalc::ann_t *ann=nullptr,  /*!< [in] 人工神经网络，可选 */
                                         const char *alg=nullptr                  /*!< [in] 可选用的算法，可选 */
                                         );
    
    /*! 输出结果，不调整误差
     
        使用json参数中的“input.json”格式进行训练样本与目标的输入。
        如果指定的计算结点dcalc不为空，则直接使用，随后判断是否指定了算法alg。如果指定了则通过alg
        寻找算法描述结点，如果没有找打则抛出异常“NANAI_ERROR_LOGIC_ALG_NOT_FOUND”。找到了则
        修改当前dcalc计算结点的配置为算法alg的描述结点。如果指定了ann，则应用。
        如果没有指定计算结点，首先会通过alg参数寻找当前管理器是否加载对应的算法描述插件，没有则抛出
        “NANAI_ERROR_LOGIC_ALG_NOT_FOUND”，如果找到，则调用generate产生一个新的计算结点来进行
        训练。
     */
    virtual int training_notarget(const std::string &json,                  /*!< [in] input.json格式文件 */
                                  nanai_ann_nanncalc *dcalc=nullptr,        /*!< [in] 要直接使用的计算结点，可选 */
                                  const char *task=nullptr,                 /*!< [in] 任务的名称，可选 */
                                  nanai_ann_nanncalc::ann_t *ann=nullptr,   /*!< [in] 人工神经网络，可选 */
                                  const char *alg=nullptr                   /*!< [in] 可选用的算法，可选 */
                                  );
    
    /*! 输出结果，不调整误差
     
     如果指定的计算结点dcalc不为空，则直接使用，随后判断是否指定了算法alg。如果指定了则通过alg
     寻找算法描述结点，如果没有找打则抛出异常“NANAI_ERROR_LOGIC_ALG_NOT_FOUND”。找到了则
     修改当前dcalc计算结点的配置为算法alg的描述结点。如果指定了ann，则应用。
     如果没有指定计算结点，首先会通过alg参数寻找当前管理器是否加载对应的算法描述插件，没有则抛出
     “NANAI_ERROR_LOGIC_ALG_NOT_FOUND”，如果找到，则调用generate产生一个新的计算结点来进行
     训练。
     */
    virtual nanai_ann_nanncalc *training_notarget(nanmath::nanmath_vector &input,           /*!< [in] 输入样本向量 */
                                                  nanai_ann_nanncalc *dcalc=nullptr,        /*!< [in] 要直接使用的计算结点，可选 */
                                                  const char *task=nullptr,                 /*!< [in] 任务的名称，可选 */
                                                  nanai_ann_nanncalc::ann_t *ann=nullptr,   /*!< [in] 人工神经网络，可选 */
                                                  const char *alg=nullptr                   /*!< [in] 可选用的算法，可选 */
                                                  );
    
    /*! 不输出结果，调整误差
     
        使用json参数中的“input.json”格式进行训练样本与目标的输入。
        如果指定的计算结点dcalc不为空，则直接使用，随后判断是否指定了算法alg。如果指定了则通过alg
        寻找算法描述结点，如果没有找打则抛出异常“NANAI_ERROR_LOGIC_ALG_NOT_FOUND”。找到了则
        修改当前dcalc计算结点的配置为算法alg的描述结点。如果指定了ann，则应用。
        如果没有指定计算结点，首先会通过alg参数寻找当前管理器是否加载对应的算法描述插件，没有则抛出
        “NANAI_ERROR_LOGIC_ALG_NOT_FOUND”，如果找到，则调用generate产生一个新的计算结点来进行
        训练。
     */
    virtual int training_nooutput(std::string &json,                              /*!< [in] input.json格式文件 */
                                  nanai_ann_nanncalc *dcalc=nullptr,              /*!< [in] 要直接使用的计算结点，可选 */
                                  const char *task=nullptr,                       /*!< [in] 任务的名称，可选 */
                                  nanai_ann_nanncalc::ann_t *ann=nullptr,         /*!< [in] 人工神经网络，可选 */
                                  const char *alg=nullptr                         /*!< [in] 可选用的算法，可选 */
                                  );
    
    /*! 不输出结果，调整误差
     
     如果指定的计算结点dcalc不为空，则直接使用，随后判断是否指定了算法alg。如果指定了则通过alg
     寻找算法描述结点，如果没有找打则抛出异常“NANAI_ERROR_LOGIC_ALG_NOT_FOUND”。找到了则
     修改当前dcalc计算结点的配置为算法alg的描述结点。如果指定了ann，则应用。如果指定了target
     则应用当前指定的目标向量，如果没有指定则使用默认的目标向量。
     如果没有指定计算结点，首先会通过alg参数寻找当前管理器是否加载对应的算法描述插件，没有则抛出
     “NANAI_ERROR_LOGIC_ALG_NOT_FOUND”，如果找到，则调用generate产生一个新的计算结点来进行
     训练。如果指定了target则应用当前指定的目标向量，如果没有指定则使用默认的目标向量。
     */
    virtual nanai_ann_nanncalc *training_nooutput(nanmath::nanmath_vector &input,           /*!< [in] 输入样本向量 */
                                                  nanmath::nanmath_vector *target,          /*!< [in] 训练目标向量，可选 */
                                                  nanai_ann_nanncalc *dcalc=nullptr,        /*!< [in] 要直接使用的计算结点，可选 */
                                                  const char *task=nullptr,                 /*!< [in] 任务的名称，可选 */
                                                  nanai_ann_nanncalc::ann_t *ann=nullptr,   /*!< [in] 人工神经网络，可选 */
                                                  const char *alg=nullptr                   /*!< [in] 可选用的算法，可选 */
                                                  );
    
    /*! 读入参数nnn指定的神经网络json文件。
     
        读入后立刻创建一个计算结点，并等待训练命令。其中以文件名作为任务名进行设置。
     */
    virtual nanai_ann_nanncalc *nnn_read(const std::string &nnn           /*!< [in] 要读入的神经网络文件路径 */
                                         );
    
    /*! 写入calc当前的神经网络到nnn指定的路径。
     
        以计算结点的任务名作为文件名 + json进行输出。
     */
    virtual void nnn_write(const std::string &nnn,            /*!< [in] 要输出的路径 */
                           nanai_ann_nanncalc *calc           /*!< [in] 要输出的计算结点指针 */
                           );
    /*! 等待所有计算结点线程结束 */
    virtual void waits();
    
    /*! 设置最大计算结点数量 */
    virtual void set_max(int max                              /*!< [in] 计算结点最大数量 */
                         );
    
    /*! 获取一个任务的所有输出 */
    virtual std::vector<nanai_ann_nanncalc::task_output_t> get_all_outputs(std::string task,      /*!< [in] 任务名 */
                                                                           bool lock_c=true,      /*!< [in] 与计算结点同步运算 */
                                                                           bool not_pop=false     /*!< [in] 不弹出输出结果 */
                                                                           );
    
    /*! 获取一个任务+jid的输出 */
    virtual nanmath::nanmath_vector get_job_output(std::string task,          /*!< [in] 任务名 */
                                                   int jid,                   /*!< [in] job id */
                                                   bool lock_c=true,          /*!< [in] 与计算结点同步运算 */
                                                   bool not_pop=false         /*!< [in] 不弹出输出结果 */
                                                   );
    
    /*! 合并任务神经网络 */
    virtual void merge_ann_by_task(std::string task,                  /*!< [in] 任务名称 */
                                   nanai_ann_nanncalc::ann_t &ann     /*!< [in] 神经网络 */
                                   );
    
    /*! 获取主目录 */
    virtual std::string get_home_dir() const;
    /*! 获取库目录 */
    virtual std::string get_lib_dir() const;
    /*! 获取配置目录 */
    virtual std::string get_etc_dir() const;
    /*! 获取日志目录 */
    virtual std::string get_log_dir() const;
    /*! 获取当前默认神经网络 */
    virtual nanai_ann_nanncalc::ann_t get_ann() const;
    /*! 获取当前默认算法 */
    virtual std::string get_alg() const;
    /*! 获取目标向量 */
    virtual nanmath::nanmath_vector get_target() const;
    
  protected:
    /*! 合并神经网络 */
    virtual nanai_ann_nanncalc::ann_t merge_ann(nanai_ann_nanncalc::ann_t &a,                 /*!< 要合并的神经网络1 */
                                                nanai_ann_nanncalc::ann_t &b                  /*!< 要合并的神经网络2 */
                                                );
    
    /*! 合并矩阵 */
    virtual nanmath::nanmath_matrix merge_matrix(nanmath::nanmath_matrix &mat1,               /*!< 要合并的权值矩阵1 */
                                                 nanmath::nanmath_matrix &mat2,               /*!< 要合并的权值矩阵2 */
                                                 nanmath::nanmath_matrix &dmat1,              /*!< 要合并的偏差矩阵1 */
                                                 nanmath::nanmath_matrix &dmat2               /*!< 要合并的偏差矩阵2 */
                                                 );
    /*! 合并偏差矩阵 */
    virtual nanmath::nanmath_matrix merge_delta_matrix(nanmath::nanmath_matrix &dmat1,        /*!< 要合并的偏差矩阵1 */
                                                       nanmath::nanmath_matrix &dmat2         /*!< 要合并的偏差矩阵2 */
                                                       );
    /*! 获取jobid */
    virtual int get_jid(std::string &task     /*!< [in] 任务名 */
                        );
  public:
    /*! 已经死亡的计算结点数量 */
    virtual int dead_task();
    /*! 当前任务有多少正在计算的结点 */
    virtual int exist_task(std::string task         /*!< 任务名 */
                           );
    
  public:
    /*! 获取版本号 */
    static const char *version();
    
  protected:
    /*! 内部配置调用get_env与get_algs */
    virtual void configure();
    /*! 获取当前运行环境，获取一些主目录路径等... */
    virtual void get_env();
    /*! 获取算法 */
    virtual void get_algs(std::string &path               /*!< 要加载的算法插件配置文件路径 */
                          );
    
    /*! 获取内置默认算法描述 */
    virtual void get_def_algs();
    
    /*! 增加一个算法 */
    virtual bool add_alg(nanai_ann_nanndesc &desc         /*!< 算法描述结构 */
                         );
    
    /*! 寻找已经安装的算法，找到返回描述指针，没找到返回nullptr */
    virtual nanai_ann_nanndesc *find_alg(std::string alg  /*!< 算法名称 */
                                         );
    
    /*! 内部运行锁 */
    virtual void lock();
    /*! 内部运行解锁 */
    virtual void unlock();
    
    /*! 按照任务名产生策略 */
    static nanai_ann_nanncalc *generate_by_task(std::vector<nanai_ann_nanncalc*> &calcs,        /*!< [in] 计算结点队列 */
                                                nanai_ann_nanndesc &desc,                       /*!< [in] 算法描述指针 */
                                                const char *task,                               /*!< [in] 指定的任务名 */
                                                nanai_ann_nanncalc::ann_t *ann                  /*!< [in] 要更换的神经网络指针  */
                                                );
    /*! 按照描述产生策略 */
    static nanai_ann_nanncalc *generate_by_desc(std::vector<nanai_ann_nanncalc*> &calcs,        /*!< [in] 计算结点队列 */
                                                nanai_ann_nanndesc &desc,                       /*!< [in] 算法描述指针 */
                                                const char *task,                               /*!< [in] 指定的任务名 */
                                                nanai_ann_nanncalc::ann_t *ann                  /*!< [in] 要更换的神经网络指针  */
                                                );
    
    /*! 产生一个计算结点
     
        这里有一套策略来控制计算结点的生成，
     
     */
    virtual nanai_ann_nanncalc *generate(nanai_ann_nanndesc &desc,                /*!< [in] 算法描述 */
                                         nanai_ann_nanncalc::ann_t *ann=NULL,     /*!< [in] 神经网络指针 */
                                         const char *task=NULL                    /*!< [in] 任务名 */
                                         );
    
    /*! 产生计算结点 */
    virtual nanai_ann_nanncalc *make(nanai_ann_nanndesc &desc,                    /*!< [in] 算法描述结点 */
                                     const char *task=nullptr,                    /*!< [in] 任务名 */
                                     nanai_ann_nanncalc::ann_t *ann=nullptr       /*!< [in] 神经网络 */
                                     );
    
    /*!< 匹配任务名 */
    static bool match_task_name(const std::string &task,          /*!< [in] 匹配任务名 */
                                const std::string &calc_task      /*!< [in] 计算结点的任务名 */
                                );
    
  protected:
    /*! 当出错时触发，重载基函数 */
    void on_error(int err             /*!< [in] 发生错误时的代码 */
                  );
   
  protected:
    /*!< 策略产生函数指针 */
    typedef nanai_ann_nanncalc *(*fptr_policy_generate)
    (std::vector<nanai_ann_nanncalc*> &calcs, nanai_ann_nanndesc &desc, const char *task, nanai_ann_nanncalc::ann_t *ann);
    std::vector<fptr_policy_generate> _fptr_policy_generates;                      /*!< 产生计算结点策略函数指针 */
    
  protected:
    std::string _home_dir;            /*!< 工作主目录 */
    std::string _lib_dir;             /*!< 库目录 */
    std::string _etc_dir;             /*!< 算法配置文件目录 */
    std::string _log_dir;             /*!< 日志目录 */
    
  protected:
    std::string _alg;                 /*!< 默认算法名称 */
    nanai_ann_nanncalc::ann_t _ann;   /*!< 默认神经网络 */
    nanmath::nanmath_vector _target;  /*!< 目标向量 */
    
  protected:
    int _max_calc;                                              /*!< 最大计算结点数量 */
    int _curr_calc;                                             /*!< 当前计算结点数量 */
    std::vector<nanai_ann_nanncalc*> _calcs;                    /*!< 计算结点队列指针 */
    std::vector<nanai_ann_nanndesc> _descs;                     /*!< 算法描述结果队列 */
    std::vector<std::pair<void*, fptr_ann_alg_setup> > _algs;   /*!< 从外部加载的算法都要保存在这里 */
    std::map<std::string, int> _jobs;
    
    pthread_mutex_t _lock;                                      /*!< 全局锁，保护_calcs队列 */
  };
}

#endif /* nanai_ann_nannmgr_h */
