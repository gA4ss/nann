#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <iostream>
#include <cmath>
#include <ctime>
#include <stdexcept>
#include <regex>

#include <fcntl.h>
#include <unistd.h>

#include <nanai_common.h>
#include <nanai_ann_nanncalc.h>

namespace nanai {
  
  const static int s_cmd_sleep_time = 100;
  void *nanai_ann_nanncalc::thread_nanai_ann_worker(void *arg) {
    // assert
    
    nanai_ann_nanncalc *calc = (nanai_ann_nanncalc*)arg;
    struct nanai_ann_nanncalc::ncommand ncmd;
    int cmd = 0;
    nanmath::nanmath_vector input;
    nanmath::nanmath_vector output;
    nanmath::nanmath_vector target;
    nanai_ann_nanndesc desc;
    std::string task;
    nanai_ann_nanncalc::ann_t ann;
    result_t *result = nullptr;
    
    while (1) {
      
      calc->set_state(NANNCALC_ST_WAITING);
      
      if (calc->get_cmd(ncmd) == NANNCALC_CMD_WAITING) {
        usleep(s_cmd_sleep_time);
        continue;
      }
      
      cmd = ncmd.cmd;
      
      if (cmd == NANNCALC_CMD_TRAINING) {
        calc->set_state(NANNCALC_ST_TRAINING);
        input = ncmd.input;
        target = ncmd.target;
        result = ncmd.result;
        ann = ncmd.ann;
        task = ncmd.task;
        /* 进行训练，并且输出结果，进行权值调整 */
        calc->ann_calculate(task, input, target, output, ann);
        calc->set_result(result, output, ann);
        calc->set_state(NANNCALC_ST_TRAINED);
        calc->ann_on_trained(input, target, output, ann);
      } else if (cmd == NANNCALC_CMD_CONFIGURE) {
        calc->set_state(NANNCALC_ST_CONFIGURING);
        desc = ncmd.desc;
        calc->ann_on_alg_uninstall();     /* 算法正在被卸载，通知算法插件 */
        calc->do_configure(desc);
        calc->set_state(NANNCALC_ST_CONFIGURED);
      } else if (cmd == NANNCALC_CMD_STOP) {
        calc->do_stop();
        break;
      } else {
        // warning
        continue;
      }
    }
    
    calc->set_state(NANNCALC_ST_STOP);
    pthread_exit(0);
  }
  
  nanai_ann_nanncalc::ann_t::ann_t() {
    clear();
  }
  
  nanai_ann_nanncalc::ann_t::ann_t(const std::string &alg,
                                   std::vector<nanmath::nanmath_matrix> &wm,
                                   std::vector<nanmath::nanmath_matrix> *dwm) {
    make(alg, wm, dwm);
  }
  
  nanai_ann_nanncalc::ann_t::ann_t(const nanai_ann_nanncalc::ann_t &t) {
    set(t);
  }
  
  nanai_ann_nanncalc::ann_t::~ann_t() {
    clear();
  }
  
  void nanai_ann_nanncalc::ann_t::set(const ann_t &t) {
    alg = t.alg;
    ninput = t.ninput;
    nhidden = t.nhidden;
    noutput = t.noutput;
    nneural = t.nneural;
    weight_matrixes = t.weight_matrixes;
    delta_weight_matrixes = t.delta_weight_matrixes;
  }

  int nanai_ann_nanncalc::ann_t::make(const std::string &alg,
                                      std::vector<nanmath::nanmath_matrix> &wm,
                                      std::vector<nanmath::nanmath_matrix> *dwm) {
    weight_matrixes = wm;
    
    if (dwm) {
      delta_weight_matrixes = *dwm;
      if (wm.size() != dwm->size()) goto _error;
    }
    
    delta_weight_matrixes.resize(wm.size());
    for (size_t i = 0; i < wm.size(); i++) {
      
      if (dwm) {
        if (wm[i].row_size() != (*dwm)[i].row_size()) goto _error;
        if (wm[i].col_size() != (*dwm)[i].col_size()) goto _error;
      } else {
        delta_weight_matrixes[i].create(wm[i].row_size(), wm[i].col_size());
      }
      
      if (i < wm.size() - 1) {
        nneural.push_back(wm[i].col_size());
      }
    }
    
    nhidden = wm.size() - 1;
    ninput = wm[0].row_size();
    noutput = wm[nhidden].col_size();
    
    fill_nneural();
    
    return 0;
  _error:
    clear();
    return -1;
  }
  
  void nanai_ann_nanncalc::ann_t::fill_nneural() {
    if (weight_matrixes.empty()) {
      nneural.clear();
      return;
    }
    
    for (size_t i = 0; i < weight_matrixes.size(); i++) {
      if (i < weight_matrixes.size() - 1) {
        nneural.push_back(weight_matrixes[i].col_size());
      }
    }
  }
  
  void nanai_ann_nanncalc::ann_t::clear() {
    nhidden = 0;
    ninput = 0;
    noutput = 0;
    weight_matrixes.clear();
    delta_weight_matrixes.clear();
    nneural.clear();
  }
  
  bool nanai_ann_nanncalc::ann_t::empty() const {
    return (ninput == 0);
  }
  
  std::vector<nanmath::nanmath_vector> nanai_ann_nanncalc::ann_t::create_hidden_layers() {
    std::vector<nanmath::nanmath_vector> hs;
    
    if (nhidden == 0) {
      return hs;
    }
    
    nanmath::nanmath_vector v;
    for (auto h : nneural) {
      v.create(h);
      hs.push_back(v);
    }
    
    return hs;
  }
  
  nanai_ann_nanncalc::nanai_ann_nanncalc(const nanai_ann_nanndesc &desc,
                                         const char *lp) : nanai_object() {
    _state = NANNCALC_CMD_STOP;
    ann_destroy();
    
    int err = 0;
    _birthday = time(nullptr);
    srandom((unsigned)_birthday);
    _cid = nanai_support_nid(*(int*)this);
    
    /* 系统输出目录 */
    if (lp != nullptr) {
      std::string filepath(lp);
      if (filepath.back() != '/') filepath += '/';
      
      char filename[256] = {0};
      struct tm *tm_ptr = gmtime(&_birthday);
      
      sprintf(filename, "nanncalc.0x%x.%d%d%d_%d%d%d.%ld.log", _cid,
              tm_ptr->tm_yday, tm_ptr->tm_mon+1, tm_ptr->tm_mday,
              tm_ptr->tm_hour, tm_ptr->tm_min, tm_ptr->tm_sec, _birthday);
      
      filepath.append(filename);
      
      if ((_log_file = fopen(filepath.c_str(), "w")) == nullptr) {
        error(NAN_ERROR_RUNTIME_OPEN_FILE);
      }
    } else {
      _log_file = stdout;
    }
    
    /* 以上插件捕获不到任何信息 */
    ann_create(desc);
    ann_log_create();
    
    /* 创建工作线程 */
    err = pthread_mutex_init(&_cmdlist_lock, NULL);
    if (err != 0) {
      error(NAN_ERROR_RUNTIME_INIT_MUTEX);
    }
    
    err = pthread_create(&_thread_worker, NULL,
                         thread_nanai_ann_worker, (void *)this);
    if (err != 0) {
      error(NAN_ERROR_RUNTIME_CREATE_THREAD);
    }
    
    if (_callback_monitor_progress)
      _callback_monitor_progress(_cid, nullptr, NANNCALC_PROCESS_CREATE, reinterpret_cast<void*>(this));
    
  }
  
  nanai_ann_nanncalc::nanai_ann_nanncalc(const nanai_ann_nanncalc &t) {
    
  }
  
  nanai_ann_nanncalc& nanai_ann_nanncalc::operator=(const nanai_ann_nanncalc &t) {
    return *this;
  }
  
  nanai_ann_nanncalc::~nanai_ann_nanncalc() {
    int err = 0;
    
    ann_stop();
    ann_destroy();
    
    err = pthread_mutex_destroy(&_cmdlist_lock);
    if (err != 0) {
      error(NAN_ERROR_RUNTIME_DESTROY_MUTEX);
    }
    
    fflush(_log_file);
    fclose(_log_file);
    
    if (_callback_monitor_progress)
      _callback_monitor_progress(_cid, nullptr, NANNCALC_PROCESS_DESTROY, reinterpret_cast<void*>(this));
  }
  
  void nanai_ann_nanncalc::ann_training(const std::string &task,
                                        const nanmath::nanmath_vector &input,
                                        const nanmath::nanmath_vector &target,
                                        const nanai_ann_nanncalc::ann_t &ann,
                                        result_t *result) {
    struct ncommand ncmd;
    ncmd.cmd = NANNCALC_CMD_TRAINING;
    ncmd.task = task;
    ncmd.input = input;
    ncmd.target = target;
    ncmd.result = result;
    ncmd.ann = ann;
    set_cmd(ncmd);
  }
  
  void nanai_ann_nanncalc::ann_configure(const nanai_ann_nanndesc &desc) {
    struct ncommand ncmd;
    ncmd.cmd = NANNCALC_CMD_CONFIGURE;
    ncmd.desc = desc;
    set_cmd(ncmd);
  }
  
  void nanai_ann_nanncalc::ann_stop() {
    if (_state == NANNCALC_ST_STOP) {
      return;
    }
    
    struct nanai_ann_nanncalc::ncommand ncmd;
    ncmd.cmd = NANNCALC_CMD_STOP;
    
    set_cmd(ncmd);
    
    void *tret = nullptr;
    int err = pthread_join(_thread_worker, &tret);
    if (err != 0) {
      error(NAN_ERROR_RUNTIME_JOIN_THREAD);
    }
    
    // assert CURR ST == ST_STOP
  }
  
  void nanai_ann_nanncalc::ann_destroy() {
    _task.clear();
    _alg.clear();
    
    _fptr_input_filter = nullptr;
    _fptr_result = nullptr;
    _fptr_output_error = nullptr;
    _fptr_calculate = nullptr;
    
    _fptr_hidden_calcs = nullptr;
    _fptr_hidden_errors = nullptr;
    _fptr_hidden_adjust_weights = nullptr;
    
    _callback_monitor_except = nullptr;
    _callback_monitor_trained = nullptr;
    _callback_monitor_progress = nullptr;
    _callback_monitor_alg_uninstall = nullptr;
  }
  
  void nanai_ann_nanncalc::ann_wait(int st,
                                    int slt) {
    while (_state != st) {
      usleep(slt);
    }
  }
  
  std::string nanai_ann_nanncalc::get_alg_name() const {
    return _alg;
  }
  
  size_t nanai_ann_nanncalc::get_cmdlist_count()  const {
    return _cmdlist.size();
  }
  
  int nanai_ann_nanncalc::get_state() const {    
    return _state;
  }
  
  void nanai_ann_nanncalc::set_cmd(struct ncommand &ncmd, bool lock) {
    int err = 0;
    
    if (lock) {
      err = pthread_mutex_lock(&_cmdlist_lock);
      if (err != 0) {
        error(NAN_ERROR_RUNTIME_LOCK_MUTEX);
      }
    }
    
    _cmdlist.push(ncmd);
    
    if (lock) {
      err = pthread_mutex_unlock(&_cmdlist_lock);
      if (err != 0) {
        error(NAN_ERROR_RUNTIME_UNLOCK_MUTEX);
      }
    }
  }
  
  int nanai_ann_nanncalc::get_cmd(struct ncommand &ncmd, bool lock) {
    int err = 0, cmd = 0;
    err = pthread_mutex_lock(&_cmdlist_lock);
    if (err != 0) {
      error(NAN_ERROR_RUNTIME_LOCK_MUTEX);
    }
    
    if (_cmdlist.empty()) {
      cmd = NANNCALC_CMD_WAITING;
      goto _end;
    }
    
    ncmd = _cmdlist.front();
    _cmdlist.pop();
    cmd = ncmd.cmd;
    
  _end:
    err = pthread_mutex_unlock(&_cmdlist_lock);
    if (err != 0) {
      error(NAN_ERROR_RUNTIME_UNLOCK_MUTEX);
    }
    
    return cmd;
  }
  
  void nanai_ann_nanncalc::set_state(int st) {
    _state = st;
  }
  
  void nanai_ann_nanncalc::do_configure(nanai_ann_nanndesc &desc) {
    ann_create(desc);
    ann_log("configure successful");
  }
  
  void nanai_ann_nanncalc::set_result(nanai_ann_nanncalc::result_t *result,
                                      const nanmath::nanmath_vector &output,
                                      const nanai_ann_nanncalc::ann_t &ann) {
    result->first = output;
    result->second = ann;
  }
  
  void nanai_ann_nanncalc::do_stop() {
    ann_log("nanncalc will be stop");
  }
  
  /* 重写基类的虚函数 */
  void nanai_ann_nanncalc::on_error(int err) {
    ann_on_except(err);
  }
  
  void nanai_ann_nanncalc::ann_create(const nanai_ann_nanndesc &desc) {
    ann_destroy();
    
    _alg = desc.name;
    
    _fptr_input_filter = desc.fptr_input_filter;
    _fptr_result = desc.fptr_result;
    _fptr_output_error = desc.fptr_output_error;
    _fptr_calculate = desc.fptr_calculate;

    _fptr_hidden_calcs = desc.fptr_hidden_calcs;
    _fptr_hidden_errors = desc.fptr_hidden_errors;
    _fptr_hidden_adjust_weights = desc.fptr_hidden_adjust_weights;

    _callback_monitor_except = desc.callback_monitor_except;
    _callback_monitor_trained = desc.callback_monitor_trained;
    _callback_monitor_progress = desc.callback_monitor_progress;
  }
  
  void nanai_ann_nanncalc::ann_on_except(int err) {
    if (_callback_monitor_except) {
      _callback_monitor_except(_cid, _task.c_str(), err, this);
    }
    
    printf("[-]<error>: nanai_ann_nanncalc on 0x%x\n", err);
  }
  
  void nanai_ann_nanncalc::ann_on_trained(nanmath::nanmath_vector &input,
                                          nanmath::nanmath_vector &target,
                                          nanmath::nanmath_vector &output,
                                          nanai::nanai_ann_nanncalc::ann_t &ann) {
    if (_callback_monitor_trained) {
      _callback_monitor_trained(_cid,
                                _task.c_str(),
                                reinterpret_cast<void*>(&input),
                                reinterpret_cast<void*>(&target),
                                reinterpret_cast<void*>(&output),
                                reinterpret_cast<void*>(&ann));
    }
    
    ann_log("training completed");
  }
  
  void nanai_ann_nanncalc::ann_on_alg_uninstall() {
    if (_callback_monitor_alg_uninstall) {
      _callback_monitor_alg_uninstall(_cid);
    }
    
    ann_log("algorithm uninstall completed");
  }
  
  void nanai_ann_nanncalc::ann_process(int process, void *arg) {
    if (_callback_monitor_progress) {
      _callback_monitor_progress(_cid, _task.c_str(), process, arg);
    }
  }
  
  void nanai_ann_nanncalc::ann_log(const char *fmt, ...) {
    char log[256] = {0};
    
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(log, 256, fmt, ap);
    va_end(ap);
    
    const char *tn = nullptr;
    if (_task.empty() == false)
      tn = _task.c_str();
    
    /* 写入到系统日志 */
    if (tn) {
      fprintf(_log_file, "<%d>[%s]:%s\n", _cid, tn, log);
    } else {
      fprintf(_log_file, "<%d>:%s\n", _cid, log);
    }
    fflush(_log_file);
    
    /* 写入到用户自定义日志 */
    if (_callback_monitor_progress) {
      _callback_monitor_progress(_cid, tn, NANNCALC_PROCESS_LOG, log);
    }
  }
  
  void nanai_ann_nanncalc::ann_log_create() {
    /* 创建神经网络 */
    ann_log("--------------------------------------------------");
    ann_log("ann start creating");
    ann_log("algorithm = %s", _alg.c_str());
    
    ann_log("ann create successful");
  }
  
  void nanai_ann_nanncalc::ann_calculate(const std::string &task,
                                         nanmath::nanmath_vector &input,
                                         nanmath::nanmath_vector &target,
                                         nanmath::nanmath_vector &output,
                                         nanai_ann_nanncalc::ann_t &ann) {
    _task = task;
    std::vector<nanmath::nanmath_vector> hs = ann.create_hidden_layers();
    output.clear();
    
    if (_fptr_calculate) {
      if (_fptr_calculate(reinterpret_cast<void*>(&_task),
                          reinterpret_cast<void*>(&input),
                          reinterpret_cast<void*>(&target),
                          reinterpret_cast<void*>(&output),
                          reinterpret_cast<void*>(&ann)) == NANAI_ANN_DESC_RETURN) {
        return;
      }
    }
    
    nanmath::nanmath_vector i, o, d_o;
    
    /* 插件输入函数过滤 */
    if (_fptr_input_filter) {
      _fptr_input_filter(&input, &i);
    } else {
      i = input;
    }
    
    /* 设定输出的向量数目 */
    o.create(ann.noutput);
    
    /* 前馈计算 */
    ann_forward(i, ann, hs, o);
    
    /* 进行权值调整 */
    if (target.empty() == false) {
      d_o = ann_output_error(target, o);                  /* 计算误差 */
      ann_hiddens_error(ann, hs, i, d_o);                 /* 调整每个权值矩阵 */
    }
    
    /* 输出结果 */
    if (_fptr_result) {
      _fptr_result(&o, &output);
    } else {
      output = o;
    }
  }
  
  void nanai_ann_nanncalc::ann_forward(nanmath::nanmath_vector &input,
                                       nanai_ann_nanncalc::ann_t &ann,
                                       std::vector<nanmath::nanmath_vector> &hiddens,
                                       nanmath::nanmath_vector &output) {
    nanmath::nanmath_vector l1;
    nanmath::nanmath_vector l2;
    nanmath::nanmath_matrix wm;
    
    /* 运算的次数(权值矩阵的个数)要比隐藏层数多1 */
    for (int i = 0; i < ann.nhidden+1; i++) {
      if (i == 0) {
        l1 = input;
        l2 = hiddens[i];
        wm = ann.weight_matrixes[i];
      } else if (i == ann.nhidden) {
        l1 = hiddens[i-1];
        l2 = output;
        wm = ann.weight_matrixes[i];
      } else {
        l1 = hiddens[i-1];
        l2 = hiddens[i];
        wm = ann.weight_matrixes[i];
      }
      
      if (_fptr_hidden_calcs) {
        ann_layer_forward(i, l1, l2, wm, _fptr_hidden_calcs);
      } else ann_layer_forward(i, l1, l2, wm, nullptr);
      
    }
    
    output = l2;
    
  }
  
  /* sigmoid */
  static double s_sigmoid(double x) {
    return (1.0 / (1.0 + exp(-x)));
  }
  
  void nanai_ann_nanncalc::ann_layer_forward(int h,
                                             nanmath::nanmath_vector &l1,
                                             nanmath::nanmath_vector &l2,
                                             nanmath::nanmath_matrix &wm,
                                             fptr_ann_alg_hidden_calc calc) {
    nanmath::nanmath_vector result;
    result = wm.left_mul(l1);
    
    if (result.size() != l2.size()) {
      error(NANAI_ERROR_LOGIC_ANN_INVALID_VECTOR_DEGREE);
    }
    
    double c = 0.0;
    l2.clear();
      
    for (int i = 0; i < result.size(); i++) {
      if (calc)
        c = calc(h, result[i]);
      else
        c = s_sigmoid(result[i]);
      
      l2.push(c);
    }
  }
  
  static double s_output_error(double target, double output) {
    double delta = output * (1.0 - output) * (target - output);
    return delta;
  }
  
  nanmath::nanmath_vector nanai_ann_nanncalc::ann_output_error(nanmath::nanmath_vector &target,
                                                               nanmath::nanmath_vector &output) {
    nanmath::nanmath_vector res;
    
    if (target.size() != output.size()) {
      error(NANAI_ERROR_LOGIC_ANN_INVALID_VECTOR_DEGREE);
    }
    
    for (int i = 0; i < target.size(); i++) {
      if (_fptr_output_error) {
        res.push(_fptr_output_error(target[i], output[i]));
      } else {
        res.push(s_output_error(target[i], output[i]));
      }
    }
    
    return res;
  }
  
  static nanmath::nanmath_vector s_ann_calc_hidden_delta(nanmath::nanmath_vector &delta_k,
                                                         nanmath::nanmath_matrix &w_kh,
                                                         nanmath::nanmath_vector &o_h) {
    nanmath::nanmath_vector delta_h(o_h.size());
    nanmath::nanmath_vector delta_sum = w_kh.right_mul(delta_k);
    
    for (size_t i = 0; i < o_h.size(); i++) {
      delta_h.set(i, o_h[i] * (1 - o_h[i]) * delta_sum[i]);
    }
    
    return delta_h;
  }
  
  static void s_ann_adjust_weight(nanmath::nanmath_matrix &wm,
                                  nanmath::nanmath_matrix &prev_dwm,
                                  nanmath::nanmath_vector &layer,
                                  nanmath::nanmath_vector &delta) {
    static const double s_eta = 0.05;/* 学习速率 */
    static const double momentum = 0.03;/* 冲量项 */
    
    /* 这里是遍历列向量 */
    for (size_t i = 0; i < delta.size(); i++) {          /* 矩阵的列 */
      for (size_t j = 0; j < layer.size(); j++) {        /* 矩阵的行 */
        /* 让上一层的每个输入向量都乘以当前的偏差值
         * 然后在修订这个偏差值的权向量
         */
        double new_dw = (s_eta * delta[i] * layer[j]) + (momentum * prev_dwm[j][i]);
        wm[j][i] += new_dw;
        prev_dwm[j][i] = new_dw;
      }
    }/* end for */
  }
  
  void nanai_ann_nanncalc::ann_hiddens_error(nanai_ann_nanncalc::ann_t &ann,
                                             std::vector<nanmath::nanmath_vector> &hiddens,
                                             nanmath::nanmath_vector &input,
                                             nanmath::nanmath_vector &output_delta) {
    /* 遍历所有隐藏权值矩阵 */
    nanmath::nanmath_vector delta_k, delta_k_next = output_delta;
    nanmath::nanmath_vector i;
    
    /* 隐藏层比权值矩阵少一个 */
    for (long h = static_cast<long>(ann.nhidden); h >= 0; h--) {
      delta_k = delta_k_next;
      /* 所需重要的变量
       * h 处于第几个权值矩阵运算
       * delta_k 上一层的误差向量
       * w_kh h->k的权值矩阵
       * o_h 当前隐藏单元的h的值
       *
       * 目的是求出delta_h，当前隐藏层的相对于上一次的偏差
       */
      
      if (h == 0) {
        /* 上一层就是输入层 */
        i = input;
      } else {
        i = hiddens[h-1];
      }
      
      if (_fptr_hidden_errors) {
        nanmath::nanmath_vector res;        /*!< 临时变量缓存结果 */
        _fptr_hidden_errors(static_cast<int>(h), &delta_k, &ann.weight_matrixes[h], &i, &res);
        delta_k_next = res;
      } else {
        delta_k_next = s_ann_calc_hidden_delta(delta_k, ann.weight_matrixes[h], i);
      }
      
      /* 调整当前的权值矩阵 */
      if (_fptr_hidden_adjust_weights) {
        _fptr_hidden_adjust_weights(static_cast<int>(h), &i, &delta_k,
                                    &ann.weight_matrixes[h], &ann.delta_weight_matrixes[h]);
      } else {
        s_ann_adjust_weight(ann.weight_matrixes[h], ann.delta_weight_matrixes[h], i, delta_k);
      }
    }
  }
}