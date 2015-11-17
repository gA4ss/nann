#ifndef nanai_mapreduce_h
#define nanai_mapreduce_h

#include <string>
#include <vector>

#include <pthread.h>

#include <nanai_common.h>
#include <nanai_object.h>

namespace nanai {
  
  /*! nanai的mapreduce模型 */
  template<class input_t, class map_result_t, class reduce_result_t>
  class nanai_mapreduce : public nanai_object {
  public:
    /*! 构造函数必须指定任务名 */
    nanai_mapreduce(const std::string &task,        /*!< 任务名 */
                    input_t &input                  /*!< 输入 */
                    );
    
    /*! 只能通过释放此类内存，才可以停止管理线程 */
    virtual ~nanai_mapreduce();
    
  public:
    
    /*! 运行 */
    virtual void run();
    /*! 运行完毕 */
    virtual bool is_done();
    /*! 设置运行完毕 */
    virtual void set_done();
    /*! 获取任务名 */
    virtual std::string get_task_name() const;
    /*! 获取map结果 */
    virtual std::vector<map_result_t> get_map_result() const;
    /*! 获取reduce结果 */
    virtual reduce_result_t get_reduce_result() const;
    /*! 等待 */
    virtual void wait();
    
  public:
    /*! map操作 */
    virtual void map();
    
    /*! reduce操作，当_count减为0时调用 */
    virtual void reduce();
    
  protected:
    static void *thread_nanai_mapreduce_worker(void *arg);
    
  protected:
    pthread_t _mapreduce_thread;                      /*!< mapreduce线程 */
    pthread_mutex_t _lock;                            /*!< 线程锁 */
    bool _run;                                        /*!< 已经调用run函数，需要卸载 */
    
  protected:
    std::string _task;                                /*!< 任务名 */
    input_t _input;                                   /*!< 输入 */
    std::vector<map_result_t> _map_results;           /*!< 每个job对应的输出结果队列 */
    reduce_result_t _reduce_result;                   /*!< 最终的结果 */
    bool _done;                                       /*!< 计算是否完成 */
  };
}

#endif /* nanai_mapreduce_h */
