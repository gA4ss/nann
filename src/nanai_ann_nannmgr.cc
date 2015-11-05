#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dlfcn.h>
#include <unistd.h>
#include <dirent.h>
#include <pthread.h>
#include <fstream>
#include <algorithm>

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
      
      while (now_start--) {
        make(_descs[0]);
      }
    }
  }
  
  nanai_ann_nannmgr::nanai_ann_nannmgr(std::string alg,
                                       nanai_ann_nanncalc::ann_t &ann,
                                       nanmath::nanmath_vector *target,
                                       int max,
                                       int now_start) {
    _alg = alg;
    _ann = ann;
    if (target) _target = *target;
    
    nanai_ann_nannmgr(max, now_start);
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
  
  nanai_ann_nanncalc *nanai_ann_nannmgr::training(nanmath::nanmath_vector &input,
                                                  nanmath::nanmath_vector *target,
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
      
      if (target) dcalc->ann_training(input, *target, task);
      else dcalc->ann_training(input, _target, task);
      
      return dcalc;
    }
    
    nanai_ann_nanndesc *desc = find_alg(alg);
    /* 没找到 */
    if (desc == nullptr) {
      error(NANAI_ERROR_LOGIC_ALG_NOT_FOUND);
    }
    
    nanai_ann_nanncalc *calc = generate(*desc, ann, task);
    
    if (target) calc->ann_training(input, *target, task);
    else calc->ann_training(input, _target, task);
    
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
                                                nanmath::nanmath_vector *target,
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
      if (target) dcalc->ann_training_nooutput(input, *target, task);
      else dcalc->ann_training_nooutput(input, _target, task);
      
      return dcalc;
    }
    
    nanai_ann_nanndesc *desc = find_alg(alg);
    /* 没找到 */
    if (desc == nullptr) {
      error(NANAI_ERROR_LOGIC_ALG_NOT_FOUND);
    }
    
    nanai_ann_nanncalc *calc = generate(*desc, ann, task);
    if (target) calc->ann_training_nooutput(input, *target, task);
    else calc->ann_training_nooutput(input, _target, task);
    
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
    long long filesize = file.tellg();
    
    unsigned char *buf = new unsigned char [filesize + 1];
    if (buf == nullptr) {
      error(NANAI_ERROR_RUNTIME_ALLOC_MEMORY);
    }
    
    file.seekg(0, std::ios::beg);
    file.read((char*)buf, (int)filesize);
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
  
  int nanai_ann_nannmgr::exist_task(std::string task) {
    int ret = 0;
    lock();
    
    for (auto i : _calcs) {
      /* 匹配到任务名 */
      if (i->get_task_name() == task) {
        if ((i->get_state() != NANNCALC_ST_WAITING) ||
            (i->get_state() != NANNCALC_ST_STOP)){
          ret++;
        }
      }
    }
    
    unlock();
    return ret;
  }
  
  int nanai_ann_nannmgr::dead_task() {
    int ret = 0;
    lock();
    
    for (auto i : _calcs) {
      if (i->get_state() != NANNCALC_ST_STOP) {
        ret++;
      }
    }
    
    unlock();
    return ret;
  }
  
  void nanai_ann_nannmgr::set_max(int max) {
    _max_calc = max;
  }
  
  nanmath::nanmath_matrix nanai_ann_nannmgr::merge_delta_matrix(nanmath::nanmath_matrix &dmat1,
                                                                nanmath::nanmath_matrix &dmat2) {
    
    if (dmat1.row_size() == dmat2.row_size()) {
      error(NANAI_ERROR_LOGIC_INVALID_ARGUMENT);
    }
    
    if (dmat1.col_size() == dmat2.col_size()) {
      error(NANAI_ERROR_LOGIC_INVALID_ARGUMENT);
    }
    
    nanmath::nanmath_matrix c(dmat1.row_size(), dmat1.col_size());
    for (size_t i = 0; i < dmat1.row_size(); i++) {
      for (size_t j = 0; j < dmat1.col_size(); j++) {
        /* 偏差越小在融合后的矩阵中所占值越大 */
        double delta_a = 1 - (dmat1[i][j] / dmat1[i][j] + dmat2[i][j]);
        double delta_b = 1 - (dmat2[i][j] / dmat1[i][j] + dmat2[i][j]);
        
        double c_n = 1/2 * (delta_a * dmat1[i][j] + delta_b * dmat2[i][j]);
        c.set(i, j, c_n);
      }
    }
    
    return c;
  }
  
  nanmath::nanmath_matrix nanai_ann_nannmgr::merge_matrix(nanmath::nanmath_matrix &mat1,
                                                          nanmath::nanmath_matrix &mat2,
                                                          nanmath::nanmath_matrix &dmat1,
                                                          nanmath::nanmath_matrix &dmat2) {
    if ((mat1.row_size() == mat2.row_size()) &&
        (mat1.row_size() == mat2.row_size()) &&
        (mat1.row_size() == dmat1.row_size()) &&
        (mat1.row_size() == dmat2.row_size())) {
      error(NANAI_ERROR_LOGIC_INVALID_ARGUMENT);
    }
    
    if ((mat1.col_size() == mat2.col_size()) &&
        (mat1.col_size() == mat2.col_size()) &&
        (mat1.col_size() == dmat1.col_size()) &&
        (mat1.col_size() == dmat2.col_size())) {
      error(NANAI_ERROR_LOGIC_INVALID_ARGUMENT);
    }
    
    nanmath::nanmath_matrix c(mat1.row_size(), mat1.col_size());
    for (size_t i = 0; i < mat1.row_size(); i++) {
      for (size_t j = 0; j < mat1.col_size(); j++) {
        double delta_a = 1 - (dmat1[i][j] / dmat1[i][j] + dmat2[i][j]);
        double delta_b = 1 - (dmat2[i][j] / dmat1[i][j] + dmat2[i][j]);
        
        double c_n = 1/2 * (delta_a * mat1[i][j] + delta_b * mat2[i][j]);
        c.set(i, j, c_n);
      }
    }
    
    return c;
  }
  
  nanai_ann_nanncalc::ann_t nanai_ann_nannmgr::merge_ann(nanai_ann_nanncalc::ann_t &a,
                                                         nanai_ann_nanncalc::ann_t &b) {
    nanai_ann_nanncalc::ann_t c;
    if (a.weight_matrixes.size() != b.weight_matrixes.size()) {
      error(NANAI_ERROR_LOGIC_INVALID_ARGUMENT);
    }
    
    if ((a.delta_weight_matrixes.size() == 0) || (b.delta_weight_matrixes.size() == 0)) {
      error(NANAI_ERROR_LOGIC_INVALID_ARGUMENT);
    }
    
    c = a;
    std::vector<nanmath::nanmath_matrix> wm;
    std::vector<nanmath::nanmath_matrix> dwm;
    
    /* 遍历每层的权值矩阵 */
    for (size_t i = 0; i < a.weight_matrixes.size(); i++) {
      wm.push_back(merge_matrix(a.weight_matrixes[i], b.weight_matrixes[i],
                                a.delta_weight_matrixes[i], b.delta_weight_matrixes[i]));
      dwm.push_back(merge_delta_matrix(a.delta_weight_matrixes[i], b.delta_weight_matrixes[i]));
    }
    c.weight_matrixes = wm;
    c.delta_weight_matrixes = dwm;
    
    return c;
  }
  
  void nanai_ann_nannmgr::merge_ann_by_task(std::string task) {
    lock();
    std::vector<nanai_ann_nanncalc::ann_t> anns;
    for (auto i : _calcs) {
      if (i->get_task_name() == task) {
        /* 状态等于等待 */
        if (i->get_state() != NANNCALC_ST_WAITING) {
          anns.push_back(i->ann_get());
        }
      }
    }
    
    if (anns.size() <= 1) {
      return;
    }
    
    /* 进行合并 */
    nanai_ann_nanncalc::ann_t a = anns[0];
    for (size_t i = 1; i < anns.size(); i++) {
      a = merge_ann(a, anns[i]);
    }
    
    unlock();
  }
  
  /* static */
  const char *nanai_ann_nannmgr::version() {
    return NANAI_ANN_VERSION_STR;
  }
  
  void nanai_ann_nannmgr::configure() {
    get_env();
    get_algs(_lib_dir);
  }
  
  static void change_path(char *path) {
    char *plocal = realpath(path, nullptr);
    if (plocal) {
      strcpy(path, plocal);
    }
    free(plocal);
  }
  
  void nanai_ann_nannmgr::get_env() {
    char *home = getenv("NANN_HOME");
    if (home) {
      _home_dir = home;
      if (_home_dir.back() != '/') _home_dir += '/';
    } else {
      char buf[256];
      strcpy(buf, "~/.nann/");
      change_path(buf);
      _home_dir = buf;
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
        std::sort(found_node.begin(), found_node.end(),
                  [](nanai_ann_nanncalc *a, nanai_ann_nanncalc *b) {
                    return a->get_cmdlist_count() > b->get_cmdlist_count();
                  });
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
          std::sort(_calcs.begin(), _calcs.end(),
                    [](nanai_ann_nanncalc *a, nanai_ann_nanncalc *b) {
                      return a->get_cmdlist_count() < b->get_cmdlist_count();
                    });
          found_node.push_back(_calcs[0]);
          found_node[0]->ann_configure(desc);
        } else {
          /* 从已经找到的结点中寻找一个命令数量最少的 */
          std::sort(found_node.begin(), found_node.end(),
                    [](nanai_ann_nanncalc *a, nanai_ann_nanncalc *b) {
                      return a->get_cmdlist_count() < b->get_cmdlist_count();
                    });
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