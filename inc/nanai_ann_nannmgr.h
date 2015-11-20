#ifndef nanai_ann_nannmgr_h
#define nanai_ann_nannmgr_h

#include <string>
#include <vector>
#include <pthread.h>

#include <nanai_ann_nanncalc.h>
#include <nanai_mapreduce_ann.h>

namespace nanai {
  
#define NANNMGR_WT_PARALLEL       0
#define NANNMGR_WT_SERIAL         1
  
  
  //! 南南人工神经网络管理器类
  class nanai_ann_nannmgr : public nanai_object {
  public:
    /*! 管理器构造函数. */
    nanai_ann_nannmgr(bool auto_clear=false                 /*!< [in] 启用自定清理线程 */
                      );
    
    /*! 析构函数.
     
        会向所有正在工作的线程发出停止的命令，并等待完成，最后销毁
     */
    virtual ~nanai_ann_nannmgr();
  protected:
    /*! 初始化所有的数据. */
    virtual void init(bool auto_clear=false                 /*!< [in] 启用自定清理线程 */
                      );
    
  public:
    
    /*! 通过json文件提取训练样例与神经网络 */
    virtual void training(const std::string &task,                  /*!< [in] 任务名 */
                          const std::string &ann_json,              /*!< [in] 神经网络 */
                          const std::string &input_json,            /*!< [in] 输入样本 */
                          int wt=NANNMGR_WT_PARALLEL                /*!< [in] 工作模式，串行还是并行 */
                          );
    
    /*! 训练 */
    virtual void training(std::vector<nanmath::nanmath_vector> &inputs,           /*!< [in] 输入样本向量 */
                          std::vector<nanmath::nanmath_vector> &targets,          /*!< [in] 训练目标向量 */
                          const std::string &task,                                /*!< [in] 任务的名称 */
                          nanai_ann_nanncalc::ann_t &ann,                         /*!< [in] 人工神经网络 */
                          int wt=NANNMGR_WT_PARALLEL                              /*!< [in] 工作模式，串行还是并行 */
                          );
    
    /*! 获取map结果 */
    virtual std::vector<nanai_ann_nanncalc::result_t> get_map_result(const std::string &task    /*!< [in] 任务名 */
                                                                    );
    /*! 获取reduce结果 */
    virtual nanai_ann_nanncalc::result_t get_reduce_result(const std::string &task              /*!< [in] 任务名 */
                                                          );
    
    /*! 等待所有mapreduce结束 */
    virtual void wait(const std::string &task            /*!< [in] 任务名 */
                      );
    /*! 等待所有mapreduce结束 */
    virtual void waits();
    /*! 释放所有mapreduce结点 */
    virtual void frees();
    /*! 清除已经完成的运算结点 */
    virtual void clears();
    /*! 清除指定mapreduce结点 */
    virtual void clear_mapreduce(const std::string &task                /*!< [in] 任务名 */
                                 );
    /*! 指定mapreduce结点是否处于完成状态 */
    virtual bool mapreduce_is_done(const std::string &task                /*!< [in] 任务名 */
                                  );
    
    /*! 获取主目录 */
    virtual std::string get_home_dir() const;
    /*! 获取库目录 */
    virtual std::string get_lib_dir() const;
    /*! 获取配置目录 */
    virtual std::string get_etc_dir() const;
    /*! 获取日志目录 */
    virtual std::string get_log_dir() const;
    
  public:
    /*! 获取版本号 */
    static const char *version();
    /*! 管理线程 */
    static void *thread_nanai_ann_manager(void *arg);
    bool manager_is_stop() const;
    
  protected:
    /*! 内部配置调用get_env与get_algs */
    virtual void configure();
    /*! 获取当前运行环境，获取一些主目录路径等... */
    virtual void get_env();
    /*! 获取算法 */
    virtual void get_algs(std::string &path                             /*!< 要加载的算法插件配置文件路径 */
                          );
    
    /*! 获取内置默认算法描述 */
    virtual void get_def_algs();
    
    /*! 增加一个算法 */
    virtual bool add_alg(const nanai_ann_nanndesc &desc                 /*!< 算法描述结构 */
                         );
    
    /*! 寻找已经安装的算法，找到返回描述指针，没找到返回nullptr */
    virtual nanai_ann_nanndesc *find_alg(const std::string &alg         /*!< 算法名称 */
                                        );
    
    /*! 寻找mapreduce */
    virtual bool find_mapreduce(const std::string &task /*!< 任务名称 */
                                );
    
    /*! 内部运行锁 */
    virtual void lock();
    /*! 内部运行解锁 */
    virtual void unlock();
    
  protected:
    /*! 当出错时触发，重载基函数 */
    void on_error(int err             /*!< [in] 发生错误时的代码 */
                  );
    
  protected:
    std::string _home_dir;            /*!< 工作主目录 */
    std::string _lib_dir;             /*!< 库目录 */
    std::string _etc_dir;             /*!< 算法配置文件目录 */
    std::string _log_dir;             /*!< 日志目录 */
    
  protected:
    std::map<std::string, std::shared_ptr<nanai_mapreduce_ann> > _mapreduce;               /*!< task - mapreduce */
    pthread_mutex_t _lock;
    pthread_t _thread_manager;
    bool _run_manager;
    bool _stop;
    
  protected:
    std::vector<nanai_ann_nanndesc> _descs;                     /*!< 算法描述结果队列 */
    std::vector<std::pair<void*, fptr_ann_alg_setup> > _algs;   /*!< 从外部加载的算法都要保存在这里 */
  };
}

#endif /* nanai_ann_nannmgr_h */
