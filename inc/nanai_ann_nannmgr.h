#ifndef nanai_ann_nannmgr_h
#define nanai_ann_nannmgr_h

#include <string>
#include <vector>
#include <pthread.h>

#include <nanai_ann_nanncalc.h>

namespace nanai {
  //! 南南人工神经网络管理器类
  /*! 管理器类用于管理所有计算结点线程
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
                      int now_start     /*!< [in] 当前就要启动的计算结点数 */
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
    virtual nanai_ann_nanncalc *nnn_read(const std::string &nnn           /*!< 要读入的神经网络文件路径 */
                                         );
    
    /*! 写入calc当前的神经网络到nnn指定的路径。
     
        以计算结点的任务名作为文件名 + json进行输出。
     */
    virtual void nnn_write(const std::string &nnn,            /*!< [in] 要输出的路径 */
                           nanai_ann_nanncalc *calc           /*!< [in] 要输出的计算结点指针 */
                           );
    virtual void waits();
    virtual void set_max(int max);
    
    /*! 合并任务神经网络 */
    virtual void merge_ann_by_task(std::string task);
  protected:
    virtual nanai_ann_nanncalc::ann_t merge_ann(nanai_ann_nanncalc::ann_t &a,
                                                nanai_ann_nanncalc::ann_t &b);
    
    virtual nanmath::nanmath_matrix merge_matrix(nanmath::nanmath_matrix &mat1,
                                                 nanmath::nanmath_matrix &mat2,
                                                 nanmath::nanmath_matrix &dmat1,
                                                 nanmath::nanmath_matrix &dmat2);
    virtual nanmath::nanmath_matrix merge_delta_matrix(nanmath::nanmath_matrix &dmat1,
                                                       nanmath::nanmath_matrix &dmat2);
    
    
  public:
    /*
     * 针对任务的产查询
     */
    virtual int dead_task();
    virtual int exist_task(std::string task);
    
  public:
    /*
     * 静态函数
     */
    static const char *version();
    
  protected:
    virtual void configure();
    virtual void get_env();
    virtual void get_algs(std::string &path);
    virtual void get_def_algs();
    virtual bool add_alg(nanai_ann_nanndesc &desc);
    virtual nanai_ann_nanndesc *find_alg(std::string alg);
    virtual void lock();
    virtual void unlock();
    virtual nanai_ann_nanncalc *generate(nanai_ann_nanndesc &desc,
                                         nanai_ann_nanncalc::ann_t *ann=NULL,
                                         const char *task=NULL);
    virtual nanai_ann_nanncalc *make(nanai_ann_nanndesc &desc);
    
  protected:
    /*
     * 重载基类虚函数
     */
    void on_error(int err);
    
  protected:
    /*
     * 环境变量
     */
    std::string _home_dir;
    std::string _lib_dir;
    std::string _etc_dir;
    std::string _log_dir;
    
  protected:
    /* 
     * 管理器唯一性识别 
     */
    std::string _alg;
    nanai_ann_nanncalc::ann_t _ann;
    nanmath::nanmath_vector _target;
    
  protected:
    int _max_calc;
    int _curr_calc;
    std::vector<nanai_ann_nanncalc*> _calcs;
    std::vector<nanai_ann_nanndesc> _descs;
    std::vector<std::pair<void*, fptr_ann_alg_setup> > _algs;/* 从外部加载的算法都要保存在这里 */
    
    pthread_mutex_t _lock;
  };
}

#endif /* nanai_ann_nannmgr_h */
