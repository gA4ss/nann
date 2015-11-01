#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dlfcn.h>
#include <unistd.h>
#include <dirent.h>
#include <pthread.h>
#include <fstream>

#include <nanai_ann_nnn.h>
#include <nanai_ann_nanncalc.h>
#include <nanai_ann_nannmgr.h>
#include <nanai_ann_version.h>

/*
 * 默认的算法
 */
#include <nanai_ann_alg_logistic.h>

namespace nanai {  
  nanai_ann_nannmgr::nanai_ann_nannmgr(int max, int now_start) : _max_calc(max){
    _curr_calc = 0;
    configure();
    
    if (pthread_mutex_init(&_lock, NULL) != 0) {
      error(NANAI_ERROR_RUNTIME_INIT_MUTEX);
    }
    
    if (now_start != 0) {
      if (now_start > max) {
        error(NANAI_ERROR_LOGIC_INVALID_ARGUMENT);
      }
      
      while (--now_start) {
        make(_descs[0]);
      }
    }
  }
  
  nanai_ann_nannmgr::~nanai_ann_nannmgr() {
    
    /* 释放所有计算结点 */
    for (auto i : _calcs) {
      if (i != nullptr) delete i;
    }
    
    for (auto i : _algs) {
      if (i.first) {
        dlclose(i.first);
      }
    }/* end for */
    
    if (pthread_mutex_destroy(&_lock) != 0) {
      error(NANAI_ERROR_RUNTIME_DESTROY_MUTEX);
    }
  }
  
  nanai_ann_nanncalc *nanai_ann_nannmgr::train(nanmath::nanmath_vector &input,
                                               nanmath::nanmath_vector &target,
                                               nanai_ann_nanncalc *dcalc,
                                               const char *task,
                                               nanai_ann_nanncalc::ann_t *ann,
                                               const char *alg) {
    
    
    if (dcalc) {
      if (alg) {
        nanai_ann_nanndesc *desc = find_alg(alg);
        /* 没找到 */
        if (desc == nullptr) {
          error(NANAI_ERROR_LOGIC_ALG_NOT_FOUND);
        }
        dcalc->do_configure(*desc);
      }
      
      if (ann) {
        dcalc->do_ann_exchange(*ann);
      }
      dcalc->ann_training(input, target, task);
      return dcalc;
    }
    
    nanai_ann_nanndesc *desc = find_alg(alg);
    /* 没找到 */
    if (desc == nullptr) {
      error(NANAI_ERROR_LOGIC_ALG_NOT_FOUND);
    }
    
    nanai_ann_nanncalc *calc = generate(*desc, ann, task);
    calc->ann_training(input, target, task);
    return calc;
  }
  
  nanai_ann_nanncalc *nanai_ann_nannmgr::training_notarget(nanmath::nanmath_vector &input,
                                                           nanai_ann_nanncalc *dcalc,
                                                           const char *task,
                                                           nanai_ann_nanncalc::ann_t *ann,
                                                           const char *alg) {
    if (dcalc) {
      if (alg) {
        nanai_ann_nanndesc *desc = find_alg(alg);
        /* 没找到 */
        if (desc == nullptr) {
          error(NANAI_ERROR_LOGIC_ALG_NOT_FOUND);
        }
        dcalc->do_configure(*desc);
      }
      
      if (ann) {
        dcalc->do_ann_exchange(*ann);
      }
      dcalc->ann_training_notarget(input, task);
      return dcalc;
    }
    
    nanai_ann_nanndesc *desc = find_alg(alg);
    /* 没找到 */
    if (desc == nullptr) {
      error(NANAI_ERROR_LOGIC_ALG_NOT_FOUND);
    }
    
    nanai_ann_nanncalc *calc = generate(*desc, ann, task);
    calc->ann_training_notarget(input, task);
    return calc;
  }

  nanai_ann_nanncalc *nanai_ann_nannmgr::training_nooutput(nanmath::nanmath_vector &input,
                                                nanmath::nanmath_vector &target,
                                                nanai_ann_nanncalc *dcalc,
                                                const char *task,
                                                nanai_ann_nanncalc::ann_t *ann,
                                                const char *alg) {
    if (dcalc) {
      if (alg) {
        nanai_ann_nanndesc *desc = find_alg(alg);
        /* 没找到 */
        if (desc == nullptr) {
          error(NANAI_ERROR_LOGIC_ALG_NOT_FOUND);
        }
        dcalc->do_configure(*desc);
      }
      
      if (ann) {
        dcalc->do_ann_exchange(*ann);
      }
      dcalc->ann_training_nooutput(input, target, task);
      return dcalc;
    }
    
    nanai_ann_nanndesc *desc = find_alg(alg);
    /* 没找到 */
    if (desc == nullptr) {
      error(NANAI_ERROR_LOGIC_ALG_NOT_FOUND);
    }
    
    nanai_ann_nanncalc *calc = generate(*desc, ann, task);
    calc->ann_training_nooutput(input, target, task);
    return calc;
  }
  
  
  nanai_ann_nanncalc *nanai_ann_nannmgr::nnn_read(const std::string &nnn) {
    /* 从nnn文件中读取一个进度，判断当前是否存在算法，不存在则失败 */
    std::fstream file;
    file.open(nnn, std::ios::in | std::ios::binary);
    if (file.is_open() == false) {
      error(NANAI_ERROR_RUNTIME_OPEN_FILE);
    }
    
    nanai_ann_nnn header;
    file.read((char*)&header, sizeof(nanai_ann_nnn));
    
    /* 使用默认的算法 */
    nanai_ann_nanndesc *desc = find_alg(header.algname);
    /* 没有找到匹配算法 */
    if (desc == nullptr) {
      error(NANAI_ERROR_LOGIC_ALG_NOT_FOUND);
    }
    
    /* 映射文件 */
    file.seekg(0, std::ios::end);
    long filesize = file.tellg();
    
    unsigned char *buf = new unsigned char [filesize + 1];
    if (buf == nullptr) {
      error(NANAI_ERROR_RUNTIME_ALLOC_MEMORY);
    }
    
    file.seekg(0, std::ios::beg);
    file.read((char*)buf, filesize);
    /* 修改ann的配置 */
    nanai_ann_nanncalc::ann_t ann = nanai_ann_nnn_read(buf);
  
    delete [] buf;
    file.close();
    
    /* 启动一个计算结点，并且应用当前的神经网络 */
    return generate(*desc, &ann, header.taskname);
  }
  
  void nanai_ann_nannmgr::nnn_write(const std::string &nnn,
                                    nanai_ann_nanncalc *calc) {
    if (calc == nullptr) {
      error(NANAI_ERROR_LOGIC_INVALID_ARGUMENT);
    }
    
    if (nnn.empty()) {
      error(NANAI_ERROR_LOGIC_INVALID_ARGUMENT);
    }
    
    std::string alg = calc->get_alg_name();
    std::string task = calc->get_task_name();
    nanai_ann_nnn header = {0};
    
    strcpy(header.algname, alg.c_str());
    strcpy(header.taskname, task.c_str());
    
    nanai_ann_nanncalc::ann_t ann = calc->get_ann();
    /* FIXME:这里用的临时空间，固定大小 */
    unsigned char *tmp_buf = new unsigned char [1024 * 1024];   /* 1MB临时空间 */
    if (tmp_buf == nullptr) {
      error(NANAI_ERROR_RUNTIME_ALLOC_MEMORY);
    }
    int ret = nanai_ann_nnn_write(ann, alg, task, tmp_buf, 1024 * 1024);
    
    /* 写入到文件 */
    std::fstream file;
    file.open(nnn, std::ios::out | std::ios::binary);
    if (file.is_open() == false) {
      error(NANAI_ERROR_RUNTIME_OPEN_FILE);
    }
    
    file.write((const char*)tmp_buf, ret);
    file.close();
  }
  
  void nanai_ann_nannmgr::waits() {
    for (auto i : _calcs) {
      i->ann_wait();
    }
  }
  
  void nanai_ann_nannmgr::set_max(int max) {
    _max_calc = max;
  }
  
  const char *nanai_ann_nannmgr::version() const {
    return NANAI_ANN_VERSION_STR;
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
      _home_dir = "/Users/devilogic/.nann/";
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
  
  bool nanai_ann_nannmgr::add_alg(nanai_ann_nanndesc &desc) {
    /* 如果desc中的算法名称已经出现在列表中，则不进行加载 */
    std::string alg(desc.name);
    
    for (auto i : _descs) {
      if (i.name == alg) {
        return false;
      }
    }
    
    /* 没找到则添加 */
    desc.fptr_main();
    _descs.push_back(desc);
    return true;
  }
  
  nanai_ann_nanndesc *nanai_ann_nannmgr::find_alg(std::string alg) {
    for (std::vector<nanai_ann_nanndesc>::iterator i = _descs.begin();
         i != _descs.end(); i++) {
      if ((*i).name == alg) {
        return &(*i);
      }
    }
    return NULL;
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
  
  /* 有些C++编译器没有支持匿名函数,为排序函数提供支持 */
  static bool s_cmp1(const nanai_ann_nanncalc *a,
                  const nanai_ann_nanncalc *b) {
    return a->get_cmdlist_count() > b->get_cmdlist_count();
  }
  
  static bool s_cmp2(const nanai_ann_nanncalc *a,
                   const nanai_ann_nanncalc *b) {
    return a->get_cmdlist_count() < b->get_cmdlist_count();
  }
  
  nanai_ann_nanncalc *nanai_ann_nannmgr::generate(nanai_ann_nanndesc &desc,
                                                  nanai_ann_nanncalc::ann_t *ann,
                                                  const char *task) {
    lock();
    nanai_ann_nanncalc *calc = nullptr;
    if (_curr_calc >= _max_calc) {
      
      /* 从所有线程中找到相同任务名的所有结点 */
      std::vector<nanai_ann_nanncalc*> found_node;
      
      /* 从所有线程中找到相同任务名的所有结点 */
      for (auto i : _calcs) {
        if (i->get_task_name() == task) {
          found_node.push_back(i);
        }
      }
      
      if (found_node.empty() == false) {
        /* 找到了，则找一个命令数量最多的
         * 这里的目的是延迟计算，让同等任务名的计算优先计算完毕
         */
        /* 有些C++编译器不支持匿名函数
        std::sort(found_node.begin(), found_node.end(),
                  [](nanai_ann_nanncalc *a, nanai_ann_nanncalc *b) {
                    return a->get_cmdlist_count() > b->get_cmdlist_count();
                  });
         */
        std::sort(found_node.begin(), found_node.end(), s_cmp1);
        /* 这里可能也会造成算法更替，所以这里替换算法保障计算正确 */
        found_node[0]->ann_configure(desc);
      } else {/* 如果没有找到则进行策略2 */
        found_node.clear();
        /* 从所有线程中找到相同算法的所有结点 */
        for (auto i : _calcs) {
          if (i->get_alg_name() == desc.name) {
            found_node.push_back(i);
          }
        }
        
        /* 如果没有找到则直接选定一个命令数量最小的，进行重新配置 */
        if (found_node.empty()) {
          /* 有些C++编译器不支持匿名函数
          std::sort(_calcs.begin(), _calcs.end(),
                    [](nanai_ann_nanncalc *a, nanai_ann_nanncalc *b) {
                      return a->get_cmdlist_count() < b->get_cmdlist_count();
                    });
           */
          std::sort(_calcs.begin(), _calcs.end(), s_cmp2);
          found_node.push_back(_calcs[0]);
          found_node[0]->ann_configure(desc);
        } else {
          /* 从已经找到的结点中寻找一个命令数量最少的 */
          /* 有些C++编译器不支持匿名函数
          std::sort(found_node.begin(), found_node.end(),
                    [](nanai_ann_nanncalc *a, nanai_ann_nanncalc *b) {
                      return a->get_cmdlist_count() < b->get_cmdlist_count();
                    });
           */
          std::sort(found_node.begin(), found_node.end(), s_cmp2);
        }
      }
      
      calc = found_node[0];
    } else {
      calc = make(desc);
      _curr_calc++;
    }
    
    /* 替换神经网络 */
    if (ann) {
      calc->do_ann_exchange(*ann);
    }
    
    unlock();
    
    return calc;
  }
  
  nanai_ann_nanncalc *nanai_ann_nannmgr::make(nanai_ann_nanndesc &desc) {
    nanai_ann_nanncalc *calc = new nanai_ann_nanncalc(desc, _log_dir.c_str());
    if (calc == NULL) {
      error(NANAI_ERROR_RUNTIME_ALLOC_MEMORY);
    }
    _calcs.push_back(calc);
    return calc;
  }
  
  void nanai_ann_nannmgr::on_error(int err) {
    // TODO
  }
}