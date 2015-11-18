#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/types.h>
#include <sys/stat.h>
#include <dlfcn.h>
#include <unistd.h>
#include <dirent.h>
#include <pthread.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <regex>

#include <nanai_common.h>
#include <nanai_ann_nnn.h>
#include <nanai_ann_nanncalc.h>
#include <nanai_ann_nannmgr.h>
#include <nanai_ann_version.h>
#include <nanai_mapreduce.h>
#include <nanai_mapreduce_ann.h>

/*
 * 默认的算法
 */
#include <nanai_ann_alg_logistic.h>

namespace nanai {
  
  const static int s_manager_sleep_time = 1000;
  void *nanai_ann_nannmgr::thread_nanai_ann_manager(void *arg) {
    nanai_ann_nannmgr *t = reinterpret_cast<nanai_ann_nannmgr*>(arg);
    
    while (t->manager_is_stop()) {
      /* 每隔一段时间，主动清除一次mapreduce空间 */
      t->clears();
      usleep(s_manager_sleep_time);
    }
    pthread_exit(0);
  }
  
  nanai_ann_nannmgr::nanai_ann_nannmgr(bool auto_clear) {
    init(auto_clear);
  }
  
  nanai_ann_nannmgr::~nanai_ann_nannmgr() {
    
    /* 释放所有计算结点 */
    waits();
    frees();
    
    if (_run_manager) {
      _stop = true;
      void *retv = nullptr;
      if (pthread_join(_thread_manager, &retv) != 0) {
        error(NANAI_ERROR_RUNTIME_JOIN_THREAD);
      }
    }/* end if */
    
    for (auto i : _algs) {
      if (i.first) {
        dlclose(i.first);
      }
    }/* end for */
    
    if (pthread_mutex_destroy(&_lock) != 0) {
      error(NANAI_ERROR_RUNTIME_DESTROY_MUTEX);
    }
  }
  
  void nanai_ann_nannmgr::init(bool auto_clear) {
    _stop = false;
    _run_manager = auto_clear;
    configure();
    
    if (pthread_mutex_init(&_lock, NULL) != 0) {
      error(NANAI_ERROR_RUNTIME_INIT_MUTEX);
    }
    
    if (_run_manager) {
      if (pthread_create(&_thread_manager, nullptr, thread_nanai_ann_manager, reinterpret_cast<void*>(this)) != 0) {
        error(NANAI_ERROR_RUNTIME_CREATE_THREAD);
      }
    }
  }
  
  void nanai_ann_nannmgr::training(const std::string &task,
                                   const std::string &ann_json,
                                   const std::string &input_json,
                                   int wt) {
    std::vector<nanmath::nanmath_vector> inputs;
    std::vector<nanmath::nanmath_vector> targets;
    nanai_ann_nanncalc::ann_t ann;
    
    if (task.empty()) {
      error(NANAI_ERROR_LOGIC_INVALID_ARGUMENT);
    }
    
    if (input_json.empty()) {
      error(NANAI_ERROR_LOGIC_INVALID_ARGUMENT);
    }
    
    if (ann_json.empty()) {
      error(NANAI_ERROR_LOGIC_INVALID_ARGUMENT);
    }
    
    /* 分析神经网络，与输入json，提取出任务数量，训练样本等
     * 产生对应的计算结点，并且替换任务名与神经网络。
     * 将筛选出的计算结点放入到mapreduce控制中
     */
    
    /* 解析神经网络json */
    nanai_ann_nnn_read(ann_json, ann);
    
    nanai_ann_nanndesc *desc = find_alg(ann.alg);
    /* 没找到 */
    if (desc == nullptr) {
      error(NANAI_ERROR_LOGIC_ALG_NOT_FOUND);
    }
    
    /* 解析样本json */
    nanai_support_input_json(input_json, inputs, targets);
    training(inputs, targets, task, ann, wt);
  }
  
  void nanai_ann_nannmgr::training(std::vector<nanmath::nanmath_vector> &inputs,
                                   std::vector<nanmath::nanmath_vector> &targets,
                                   const std::string &task,
                                   nanai_ann_nanncalc::ann_t &ann,
                                   int wt) {
    if (task.empty()) {
      error(NANAI_ERROR_LOGIC_INVALID_ARGUMENT);
    }
    
    if (inputs.empty()) {
      error(NANAI_ERROR_LOGIC_INVALID_ARGUMENT);
    }
    
    if (targets.empty()) {
      error(NANAI_ERROR_LOGIC_INVALID_ARGUMENT);
    }
    
    if (targets.size() != inputs.size()) {
      error(NANAI_ERROR_LOGIC_INVALID_ARGUMENT);
    }
    
    nanai_ann_nanndesc *desc = find_alg(ann.alg);
    if (desc == nullptr) {
      error(NANAI_ERROR_LOGIC_ALG_NOT_FOUND);
    }
    
    if (find_mapreduce(task)) {
      error(NANAI_ERROR_LOGIC_TASK_ALREADY_EXIST);
    }
    
    nanai_mapreduce_ann::nanai_mapreduce_ann_config_t mp_config;
    mp_config.desc = *desc;
    mp_config.log_dir = _log_dir;
    mp_config.wt = wt;
    
    std::pair<std::vector<nanmath::nanmath_vector>, std::vector<nanmath::nanmath_vector> >
    sample = std::make_pair(inputs, targets);
    nanai_mapreduce_ann_input_t input = std::make_pair(sample, ann);
    //nanai_mapreduce_ann *node = reinterpret_cast<nanai_mapreduce_ann*>(new nanai_mapreduce<nanai_mapreduce_ann_input_t, nanai_ann_nanncalc::result_t,nanai_ann_nanncalc::result_t>(task, input));
    
    nanai_mapreduce_ann *node = new nanai_mapreduce_ann(task, input);
    if (node == nullptr) {
      error(NANAI_ERROR_RUNTIME_ALLOC_MEMORY);
    }
    
    node->read_config(mp_config);
    node->run();
    
    lock();
    _mapreduce[task] = node;
    unlock();
  }
  
  std::vector<nanai_ann_nanncalc::result_t> nanai_ann_nannmgr::get_map_result(const std::string &task) {
    std::vector<nanai_ann_nanncalc::result_t> result;
    lock();
    if (_mapreduce.find(task) == _mapreduce.end()) {
      error(NANAI_ERROR_LOGIC_TASK_NOT_MATCHED);
    }
    result = _mapreduce[task]->get_map_result();
    unlock();
    
    return result;
  }

  nanai_ann_nanncalc::result_t nanai_ann_nannmgr::get_reduce_result(const std::string &task) {
    nanai_ann_nanncalc::result_t result;
    lock();
    if (_mapreduce.find(task) == _mapreduce.end()) {
      error(NANAI_ERROR_LOGIC_TASK_NOT_MATCHED);
    }
    result = _mapreduce[task]->get_reduce_result();
    unlock();
    
    return result;
  }

  void nanai_ann_nannmgr::waits() {
    lock();
    for (auto i : _mapreduce) {
      i.second->wait();
    }
    unlock();
  }
  
  void nanai_ann_nannmgr::frees() {
    lock();
    for (auto i : _mapreduce) {
      if (i.second != nullptr) delete i.second;
    }
    _mapreduce.clear();
    unlock();
  }
  
  void nanai_ann_nannmgr::clears() {
    lock();
    std::vector<std::string> tasks;
    for (auto i : _mapreduce) {
      if (i.second != nullptr) {
        if (i.second->is_done()) {
          tasks.push_back(i.first);
        }
      }
    }
    
    for (auto i : tasks) {
      if (_mapreduce[i] != nullptr) delete _mapreduce[i];
      _mapreduce.erase(i);
    }
    
    unlock();
  }
  
  void nanai_ann_nannmgr::clear_mapreduce(const std::string &task) {
    lock();
    if (_mapreduce.find(task) == _mapreduce.end()) {
      return;
    }
    
    _mapreduce.erase(task);
    unlock();
  }
  
  bool nanai_ann_nannmgr::mapreduce_is_done(const std::string &task) {
    bool done = false;
    
    lock();
    if (_mapreduce.find(task) == _mapreduce.end()) {
      error(NANAI_ERROR_LOGIC_TASK_NOT_MATCHED);
    }
    
    done = _mapreduce[task]->is_done();
    unlock();
    
    return done;
  }
  
  std::string nanai_ann_nannmgr::get_home_dir() const {
    return _home_dir;
  }
  
  std::string nanai_ann_nannmgr::get_lib_dir() const {
    return _lib_dir;
  }
  
  std::string nanai_ann_nannmgr::get_etc_dir() const {
    return _etc_dir;
  }
  
  std::string nanai_ann_nannmgr::get_log_dir() const {
    return _log_dir;
  }
  
  /* static */
  const char *nanai_ann_nannmgr::version() {
    return NANAI_ANN_VERSION_STR;
  }
  
  bool nanai_ann_nannmgr::manager_is_stop() const {
    return _stop;
  }
  
  void nanai_ann_nannmgr::configure() {
    get_env();
    get_algs(_lib_dir);
  }
  
  void nanai_ann_nannmgr::get_env() {
    char *home = getenv("NANN_HOME");
    if (home) {
      _home_dir = home;
      if (_home_dir.back() != '/') _home_dir += '/';
    } else {
#if NDEBUG==0
      _home_dir = "/Users/devilogic/.nann/";
#else
      error(NANAI_ERROR_LOGIC_HOME_DIR_NOT_CONFIG);
#endif
    }
    
    char *lib_dir = getenv("NANN_LIB_DIR");
    if (lib_dir) {
      _lib_dir = lib_dir;
      if (_lib_dir.back() != '/') _lib_dir += '/';
    } else {
      _lib_dir = _home_dir + "lib/";
    }
    
    char *etc_dir = getenv("NANN_ETC_DIR");
    if (etc_dir) {
      _etc_dir = etc_dir;
      if (_etc_dir.back() != '/') _etc_dir += '/';
    } else {
      _etc_dir = _home_dir + "etc/";
    }
    
    char *log_dir = getenv("NANN_LOG_DIR");
    if (log_dir) {
      _log_dir = log_dir;
      if (_log_dir.back() != '/') _log_dir += '/';
    } else {
      _log_dir = _home_dir + "log/";
    }
    
  }
  
  void nanai_ann_nannmgr::get_algs(std::string &path) {
    get_def_algs();
    
    /* 遍历_alg_dir下的所有so文件并加载
     * 并且通过通用接口 "ann_alg_setup"
     * 如果没有此接口则不是算法控件
     */
    
    if (path.back() != '/') {
      path += '/';
    }
    
    DIR *dir;
    struct dirent *ent;;
    std::string filename, subdir;
    
    dir = opendir(path.c_str());
    
    while ((ent = readdir(dir))) {
      if (ent->d_type & DT_DIR) {
        if (strcmp(ent->d_name,".") == 0 || strcmp(ent->d_name, "..") == 0)
          continue;
        
        if (1) {
          subdir = path + ent->d_name;
          get_algs(subdir);
        }
      } else {
        /* 加载so文件 */
        filename = path + ent->d_name;
        void *h = dlopen(filename.c_str(), RTLD_NOW);
        if (h) {
          fptr_ann_alg_setup pf = (fptr_ann_alg_setup)dlsym(h, "ann_alg_setup");
          if (pf) {
            /* 安装 */
            if (add_alg(*pf(_etc_dir.c_str()))) {
              auto v = std::make_pair(h, pf);
              _algs.push_back(v);
            }
          }/* end if */
        }
      }/* end else */
    }/* end while */
    
  }
  
  void nanai_ann_nannmgr::get_def_algs() {
    /* logistic */
    add_alg(*ann_alg_logistic_setup(_etc_dir.c_str()));
  }
  
  bool nanai_ann_nannmgr::add_alg(const nanai_ann_nanndesc &desc) {
    /* 如果desc中的算法名称已经出现在列表中，则不进行加载 */
    std::string alg(desc.name);
    
    for (auto i : _descs) {
      if (i.name == alg) {
        return false;
      }
    }
    
    /* 没找到则添加 */
    desc.fptr_event_added();
    _descs.push_back(desc);
    return true;
  }
  
  nanai_ann_nanndesc *nanai_ann_nannmgr::find_alg(const std::string &alg) {
    if (alg.empty()) {
      return nullptr;
    }
    
    for (std::vector<nanai_ann_nanndesc>::iterator i = _descs.begin();
         i != _descs.end(); i++) {
      if ((*i).name == alg) {
        return &(*i);
      }
    }
    return nullptr;
  }
  
  nanai_mapreduce_ann* nanai_ann_nannmgr::find_mapreduce(const std::string &task) {
    nanai_mapreduce_ann *res = nullptr;
    
    if (task.empty()) {
      return nullptr;
    }
    
    lock();
    
    if (_mapreduce.find(task) != _mapreduce.end()) {
      res = _mapreduce[task];
    }
    
    unlock();
    
    return res;
  }
  
  void nanai_ann_nannmgr::lock() {
    if (pthread_mutex_lock(&_lock) != 0) {
      error(NANAI_ERROR_RUNTIME_LOCK_MUTEX);
    }
  }
  
  void nanai_ann_nannmgr::unlock() {
    if (pthread_mutex_unlock(&_lock) != 0) {
      error(NANAI_ERROR_RUNTIME_UNLOCK_MUTEX);
    }
  }
  
  void nanai_ann_nannmgr::on_error(int err) {
    // TODO
  }
}