#ifndef nanai_mapreduce_h
#define nanai_mapreduce_h

#include <string>
#include <vector>

#include <pthread.h>

#include <nanai_common.h>
#include <nanai_object.h>
#include <iostream>
namespace nanai {
  
  /*! nanai的mapreduce模型 */
  template<typename input_t, typename map_result_t, typename reduce_result_t>
  class nanai_mapreduce : public nanai_object {
  public:
    nanai_mapreduce(){}
    
    /*! 构造函数必须指定任务名 */
    nanai_mapreduce(const std::string &task,        /*!< 任务名 */
                    input_t &input                  /*!< 输入 */
    ) : nanai_object() {
      _input = input;
      _task = task;
      _run = false;
      _done = false;
    }
    
    /*! 只能通过释放此类内存，才可以停止管理线程 */
    virtual ~nanai_mapreduce() {
      int err = 0;
      //printf("~nanai_mapreduce\n");
      if (_run) {
        
        /* 等待线程完毕 */
        while (_done == false) {
          usleep(100);
        }
        
//        void *tret = nullptr;
//        err = pthread_join(_mapreduce_thread, &tret);
//        if (err != 0) {
//          error(NAN_ERROR_RUNTIME_JOIN_THREAD);
//        }
        
        err = pthread_attr_destroy(&_mapreduce_thread_attr);
        if (err != 0) {
          error(NAN_ERROR_RUNTIME_DESTROY_ATTRIBUTE);
        }
        
        err = pthread_mutex_destroy(&_lock);
        if (err != 0) {
          error(NAN_ERROR_RUNTIME_DESTROY_MUTEX);
        }
      }
    }
    
  public:
    
    /*! 运行 */
    virtual void run() {
      /* 创建工作线程 */
      int err = pthread_mutex_init(&_lock, nullptr);
      if (err != 0) {
        error(NAN_ERROR_RUNTIME_INIT_MUTEX);
      }
      
      err = pthread_attr_init(&_mapreduce_thread_attr);
      if (err != 0) {
        error(NAN_ERROR_RUNTIME_INIT_THREAD_ATTR);
      }
      
      /* 设置为线程分离状态 */
      err = pthread_attr_setdetachstate(&_mapreduce_thread_attr, PTHREAD_CREATE_DETACHED);
      if (err != 0) {
        error(NAN_ERROR_RUNTIME_SETDETACHSTATE);
      }
      
      err = pthread_create(&_mapreduce_thread, &_mapreduce_thread_attr,
                           thread_nanai_mapreduce_worker, (void *)this);
      if (err != 0) {
        error(NAN_ERROR_RUNTIME_CREATE_THREAD);
      }
      
      _run = true;
    }
    
    /*! 运行完毕 */
    virtual bool is_done() { return _done; }
    /*! 设置运行完毕 */
    virtual void set_done() { _done = true; }
    /*! 获取任务名 */
    virtual std::string get_task_name() const { return _task; }
    /*! 获取map结果 */
    virtual std::vector<map_result_t> get_map_result() const { return _map_results; }
    /*! 获取reduce结果 */
    virtual reduce_result_t get_reduce_result() const { return _reduce_result; }
    /*! 等待 */
    virtual void wait() {
      if (_run == false) {
        return;
      }
      
      while (_done == false) {
        usleep(100);
      }
    }
    
  public:
    /*! map操作 */
    virtual void map() {};
    
    /*! reduce操作，当_count减为0时调用 */
    virtual void reduce() {};
    
  protected:
    static void *thread_nanai_mapreduce_worker(void *arg) {
      nanai_mapreduce<input_t, map_result_t, reduce_result_t> *mp =
      reinterpret_cast<nanai_mapreduce<input_t, map_result_t, reduce_result_t>*>(arg);
      
      mp->map();
      mp->reduce();
      mp->set_done();
      
      pthread_exit(0);
    }
    
  protected:
    pthread_t _mapreduce_thread;                      /*!< mapreduce线程 */
    pthread_attr_t _mapreduce_thread_attr;            /*!< mapreduce线程属性 */
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
