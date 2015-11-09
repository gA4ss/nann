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
#include <algorithm>

#include <nanai_common.h>
#include <nanai_ann_nnn.h>
#include <nanai_ann_nanncalc.h>
#include <nanai_ann_nannmgr.h>
#include <nanai_ann_version.h>

/*
 * 默认的算法
 */
#include <nanai_ann_alg_logistic.h>

namespace nanai {  
  nanai_ann_nannmgr::nanai_ann_nannmgr(int max, int now_start) {
    init(max, now_start, nullptr);
  }
  
  nanai_ann_nannmgr::nanai_ann_nannmgr(std::string alg,
                                       nanai_ann_nanncalc::ann_t &ann,
                                       nanmath::nanmath_vector *target,
                                       const char *task,
                                       int max,
                                       int now_start) {
    _alg = alg;
    _ann = ann;
    if (target) _target = *target;
    
    init(max, now_start, task);
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
  
  void nanai_ann_nannmgr::init(int max,
                               int now_start,
                               const char *task) {
    _max_calc = max;
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
        make(_descs[0], task);
      }
    }
  }
  
  int nanai_ann_nannmgr::training(const std::string &json,
                                  nanai_ann_nanncalc *dcalc,
                                  const char *task,
                                  nanai_ann_nanncalc::ann_t *ann,
                                  const char *alg) {
    std::vector<nanmath::nanmath_vector> inputs;
    nanmath::nanmath_vector target;
    int i = 0;
    
    /* 解析json */
    nanai_support_input_json(json, inputs, &target);
    
    for (auto x : inputs) {
      if (target.size()) training(x, &target, dcalc, task, ann, alg);
      else training(x, nullptr, dcalc, task, ann, alg);
      i++;
    }
    
    return i;
  }
  
  nanai_ann_nanncalc *nanai_ann_nannmgr::training(nanmath::nanmath_vector &input,
                                                  nanmath::nanmath_vector *target,
                                                  nanai_ann_nanncalc *dcalc,
                                                  const char *task,
                                                  nanai_ann_nanncalc::ann_t *ann,
                                                  const char *alg) {
    
    /* 如果计算结点存在则使用 */
    if (dcalc) {
      /* 如果指定了算法名称 */
      if (alg) {
        /* 通过算法名，找到算法描述结点 */
        nanai_ann_nanndesc *desc = find_alg(alg);
        /* 没找到 */
        if (desc == nullptr) {
          error(NANAI_ERROR_LOGIC_ALG_NOT_FOUND);
        }
        /* 修改计算结点的配置 */
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
    
    /* 产生新结点进行运算 */
    nanai_ann_nanncalc *calc = generate(*desc, ann, task);
    
    if (target) calc->ann_training(input, *target, task);
    else calc->ann_training(input, _target, task);
    
    return calc;
  }
  
  int nanai_ann_nannmgr::training_notarget(const std::string &json,
                                           nanai_ann_nanncalc *dcalc,
                                           const char *task,
                                           nanai_ann_nanncalc::ann_t *ann,
                                           const char *alg) {
    std::vector<nanmath::nanmath_vector> inputs;
    int i = 0;
    
    /* 解析json */
    nanai_support_input_json(json, inputs, nullptr);
    for (auto x : inputs) {
      training_notarget(x, dcalc, task, ann, alg);
      i++;
    }
    
    return i;
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

  int nanai_ann_nannmgr::training_nooutput(std::string &json,
                                           nanai_ann_nanncalc *dcalc,
                                           const char *task,
                                           nanai_ann_nanncalc::ann_t *ann,
                                           const char *alg) {
    std::vector<nanmath::nanmath_vector> inputs;
    nanmath::nanmath_vector target;
    int i = 0;
    
    /* 解析json */
    nanai_support_input_json(json, inputs, &target);
    for (auto x : inputs) {
      if (target.size()) training_nooutput(x, &target, dcalc, task, ann, alg);
      else training_nooutput(x, nullptr, dcalc, task, ann, alg);
      i++;
    }
    
    return i;
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
    file.open(nnn, std::ios::in|std::ios::binary);
    if (file.is_open() == false) {
      error(NANAI_ERROR_RUNTIME_OPEN_FILE);
    }
    
    std::string alg;
    nanai_ann_nanncalc::ann_t ann;
    nanmath::nanmath_vector target;

    file.seekg(0, std::ios::end);
    size_t fs = static_cast<size_t>(file.tellg());
    char *buf = new char [fs+1];
    if (buf == nullptr) {
      error(NANAI_ERROR_RUNTIME_ALLOC_MEMORY);
    }
    memset(buf, 0, fs+1);
    file.read(buf, fs);
    std::string json_context = buf;
    delete [] buf;
    
    nanai_ann_nnn_read(json_context, alg, ann, &target);
    
    /* 使用默认的算法 */
    nanai_ann_nanndesc *desc = find_alg(alg);
    /* 没有找到匹配算法 */
    if (desc == nullptr) {
      error(NANAI_ERROR_LOGIC_ALG_NOT_FOUND);
    }
    
    /* 设置全局的目标向量，如果存在的话 */
    if (target.size() != 0) {
      _target = target;
    }
    
    file.close();
    
    /* 取出文件名作为任务名，启动一个计算结点，并且应用当前的神经网络 */
    std::string task = nanai_support_just_filename(nnn);
    return generate(*desc, &ann, task.c_str());
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
    std::string task = calc->get_task_name();     /* 作为最后的文件名 */
    nanai_ann_nanncalc::ann_t ann = calc->get_ann();
    std::string json_context;
    
    nanai_ann_nnn_write(json_context, alg, ann, &_target);
    
    /* 写入到文件 */
    std::ofstream file;
    std::string outfile = nnn;
    if (outfile.back() != '/') outfile += "/";
    outfile += task;
    outfile += ".json";
    
    file.open(outfile, std::ios::out);
    if (file.is_open() == false) {
      error(NANAI_ERROR_RUNTIME_OPEN_FILE);
    }
    
    file << json_context;
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
  
  void nanai_ann_nannmgr::merge_ann_by_task(std::string task,
                                            nanai_ann_nanncalc::ann_t &ann) {
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
      error(NANAI_ERROR_LOGIC_ANN_NUMBER_LESS_2);
    }
    
    /* 进行合并 */
    _ann = anns[0];
    for (size_t i = 1; i < anns.size(); i++) {
      _ann = merge_ann(_ann, anns[i]);
    }
    unlock();
    ann = _ann;
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
  
  nanai_ann_nanncalc::ann_t nanai_ann_nannmgr::get_ann() const {
    return _ann;
  }
  
  std::string nanai_ann_nannmgr::get_alg() const {
    return _alg;
  }
  
  nanmath::nanmath_vector nanai_ann_nannmgr::get_target() const {
    return _target;
  }
  
  /* static */
  const char *nanai_ann_nannmgr::version() {
    return NANAI_ANN_VERSION_STR;
  }
  
  void nanai_ann_nannmgr::configure() {
    get_env();
    get_algs(_lib_dir);
  }
  
  /*
  static void change_path(char *path) {
    char *plocal = realpath(path, nullptr);
    if (plocal) {
      strcpy(path, plocal);
    }
    free(plocal);
  }
  */
  
  void nanai_ann_nannmgr::get_env() {
    char *home = getenv("NANN_HOME");
    if (home) {
      _home_dir = home;
      if (_home_dir.back() != '/') _home_dir += '/';
    } else {
      //_home_dir = "/Users/devilogic/.nann/";
      error(NANAI_ERROR_LOGIC_HOME_DIR_NOT_CONFIG);
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
      calc = make(desc, task);
      _curr_calc++;
    }
    
    /* 替换神经网络 */
    if (ann) {
      calc->do_ann_exchange(*ann);
    }
    
    unlock();
    
    return calc;
  }
  
  nanai_ann_nanncalc *nanai_ann_nannmgr::make(nanai_ann_nanndesc &desc,
                                              const char *task) {
    nanai_ann_nanncalc *calc = new nanai_ann_nanncalc(desc, _log_dir.c_str(), task);
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