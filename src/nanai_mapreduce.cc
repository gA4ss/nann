#include <string>
#include <pthread.h>
#include <nanai_mapreduce.h>

namespace nanai {
  template<class input_t, class map_result_t, class reduce_result_t>
  void *nanai_mapreduce<input_t, map_result_t, reduce_result_t>::thread_nanai_mapreduce_worker(void *arg) {
    nanai_mapreduce<input_t, map_result_t, reduce_result_t> *mp =
      reinterpret_cast<nanai_mapreduce<input_t, map_result_t, reduce_result_t> >(arg);
    
    mp->map();
    mp->reduce();
    mp->set_done();

    pthread_exit(0);
  }
  
  template<class input_t, class map_result_t, class reduce_result_t>
  nanai_mapreduce<input_t, map_result_t, reduce_result_t>::nanai_mapreduce(const std::string &task,
                                                                           input_t &input) {
    
    /* 读入input参数并且开始运算 */
    _input = input;
    _task = task;
    _run = false;
    _done = false;
  }
  
  template<class input_t, class map_result_t, class reduce_result_t>
  nanai_mapreduce<input_t, map_result_t, reduce_result_t>::~nanai_mapreduce() {
    if (_run) {
      
      /* 等待线程完毕 */
      void *tret = nullptr;
      int err = pthread_join(_mapreduce_thread, &tret);
      if (err != 0) {
        error(NANAI_ERROR_RUNTIME_JOIN_THREAD);
      }
      
      err = pthread_mutex_destroy(&_lock);
      if (err != 0) {
        error(NANAI_ERROR_RUNTIME_DESTROY_MUTEX);
      }
    }
  }
  
  template<class input_t, class map_result_t, class reduce_result_t>
  void nanai_mapreduce<input_t, map_result_t, reduce_result_t>::run() {

    /* 创建工作线程 */
    int err = pthread_mutex_init(&_lock, nullptr);
    if (err != 0) {
      error(NANAI_ERROR_RUNTIME_INIT_MUTEX);
    }
    
    err = pthread_create(&_mapreduce_thread, nullptr,
                         thread_nanai_mapreduce_worker, (void *)this);
    if (err != 0) {
      error(NANAI_ERROR_RUNTIME_CREATE_THREAD);
    }
    
    _run = true;
  }
  
  template<class input_t, class map_result_t, class reduce_result_t>
  bool nanai_mapreduce<input_t, map_result_t, reduce_result_t>::is_done() {
    return _done;
  }
  
  template<class input_t, class map_result_t, class reduce_result_t>
  void nanai_mapreduce<input_t, map_result_t, reduce_result_t>::set_done() {
    _done = true;
  }

  template<class input_t, class map_result_t, class reduce_result_t>
  std::string nanai_mapreduce<input_t, map_result_t, reduce_result_t>::get_task_name() const {
    return _task;
  }
  
  template<class input_t, class map_result_t, class reduce_result_t>
  std::vector<map_result_t> nanai_mapreduce<input_t, map_result_t, reduce_result_t>::get_map_result() const {
    return _map_results;
  }
  
  template<class input_t, class map_result_t, class reduce_result_t>
  reduce_result_t nanai_mapreduce<input_t, map_result_t, reduce_result_t>::get_reduce_result() const {
    return _reduce_result;
  }
  
  template<class input_t, class map_result_t, class reduce_result_t>
  void nanai_mapreduce<input_t, map_result_t, reduce_result_t>::wait() {
    if (_run == false) {
      return;
    }
    
    while (_done == false) {
      usleep(100);
    }
  }
  
}