#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <stdexcept>

#include <nanai_common.h>
#include <nanai_ann_nanncalc.h>

namespace nanai {
  
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
    
    while (1) {
      
      calc->set_state(NANNCALC_ST_WAITING);
      
      if (calc->get_cmd(ncmd) == NANNCALC_ST_WAITING) {
        sleep(100);
        continue;
      }
      
      cmd = ncmd.cmd;
      
      if (cmd == NANNCALC_CMD_TRAINING) {
        calc->set_state(NANNCALC_ST_TRAINING);
        input = ncmd.input;
        target = ncmd.target;
        task = ncmd.task;
        
        /* 进行训练，并且输出结果，进行权值调整 */
        calc->ann_calculate(task.c_str(), input, &target, &output);
        calc->set_output(output);
        calc->set_state(NANNCALC_ST_TRAINED);
        calc->ann_on_trained();
      } else if (cmd == NANNCALC_CMD_CALCULATE) {
        calc->set_state(NANNCALC_ST_TRAINING);
        input = ncmd.input;
        task = ncmd.task;
        
        /* 进行训练，输出结果，不进行权值调整 */
        calc->ann_calculate(task.c_str(), input, nullptr, &output);
        calc->set_output(output);
        calc->set_state(NANNCALC_ST_TRAINED);
        calc->ann_on_calculated();
      } else if (cmd == NANNCALC_CMD_TRAINING_NOOUTPUT) {
        calc->set_state(NANNCALC_ST_TRAINING);
        input = ncmd.input;
        target = ncmd.target;
        task = ncmd.task;

        /* 单训练，不输出结果，进行权值调整 */
        calc->ann_calculate(task.c_str(), input, &target, nullptr);
        calc->set_state(NANNCALC_ST_TRAINED);
        calc->ann_on_trained_nooutput();
      } else if (cmd == NANNCALC_CMD_CONFIGURE) {
        calc->set_state(NANNCALC_ST_CONFIGURING);
        desc = ncmd.desc;
        calc->ann_on_alg_uninstall();     /* 算法正在被卸载，通知算法插件 */
        calc->do_configure(desc);
        calc->set_state(NANNCALC_ST_CONFIGURED);
      } else if (cmd == NANNCALC_CMD_ANN_EXCHANGE) {
        calc->set_state(NANNCALC_ST_ANN_EXCHANGING);
        ann = ncmd.ann;
        if (ann.weight_matrix.empty() == false) {
          calc->do_ann_exchange(ann);
        }
        calc->set_state(NANNCALC_ST_ANN_EXCHANGED);
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
  
  nanai_ann_nanncalc::nanai_ann_nanncalc(nanai_ann_nanndesc &desc, const char *lp) : _state(NANNCALC_CMD_STOP) {
    
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
        error(NANAI_ERROR_RUNTIME_OPEN_FILE);
      }
    } else {
      _log_file = stdout;
    }
    
    /* 以上插件捕获不到任何信息 */
    ann_create(desc);
    
    err = pthread_mutex_init(&_outputs_lock, NULL);
    if (err != 0) {
      error(NANAI_ERROR_RUNTIME_INIT_MUTEX);
    }
    
    err = pthread_mutex_init(&_state_lock, NULL);
    if (err != 0) {
      error(NANAI_ERROR_RUNTIME_INIT_MUTEX);
    }
    
    err = pthread_mutex_init(&_ann_lock, NULL);
    if (err != 0) {
      error(NANAI_ERROR_RUNTIME_INIT_MUTEX);
    }
    
    /* 创建工作线程 */
    err = pthread_mutex_init(&_cmdlist_lock, NULL);
    if (err != 0) {
      error(NANAI_ERROR_RUNTIME_INIT_MUTEX);
    }
    
    err = pthread_create(&_thread_worker, NULL,
                         thread_nanai_ann_worker, (void *)this);
    if (err != 0) {
      error(NANAI_ERROR_RUNTIME_CREATE_THREAD);
    }
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
    
    err = pthread_mutex_destroy(&_outputs_lock);
    if (err != 0) {
      error(NANAI_ERROR_RUNTIME_DESTROY_MUTEX);
    }
    
    err = pthread_mutex_destroy(&_state_lock);
    if (err != 0) {
      error(NANAI_ERROR_RUNTIME_DESTROY_MUTEX);
    }
    
    err = pthread_mutex_destroy(&_ann_lock);
    if (err != 0) {
      error(NANAI_ERROR_RUNTIME_DESTROY_MUTEX);
    }
    
    err = pthread_mutex_destroy(&_cmdlist_lock);
    if (err != 0) {
      error(NANAI_ERROR_RUNTIME_DESTROY_MUTEX);
    }
    
    fflush(_log_file);
    fclose(_log_file);
  }
  
  void nanai_ann_nanncalc::ann_default_create(int ninput,
                                              int nhidden,
                                              int output,
                                              std::vector<int> &nneure) {
    nanai_ann_nanndesc desc = {0};
    strcpy(desc.name, "NDBpA");
    strcpy(desc.description, "default");
    desc.nhidden = nhidden;
    desc.ninput = ninput;
    desc.noutput = output;
    
    for (int i = 0; i < nhidden; i++) {
      desc.nneure[i] = nneure[i];
    }
    
    ann_create(desc);
  }
  
  void nanai_ann_nanncalc::ann_training(nanmath::nanmath_vector &input,
                                        nanmath::nanmath_vector &target,
                                        const char *task) {
    struct ncommand ncmd;
    ncmd.cmd = NANNCALC_CMD_TRAINING;
    ncmd.input = input;
    ncmd.target = target;
    if (task) ncmd.task = task;
    set_cmd(ncmd);
  }
  
  void nanai_ann_nanncalc::ann_training_notarget(nanmath::nanmath_vector &input,
                                                 const char *task) {
    struct ncommand ncmd;
    ncmd.cmd = NANNCALC_CMD_TRAINING_NOTARGET;
    ncmd.input = input;
    if (task) ncmd.task = task;
    set_cmd(ncmd);
  }
  
  void nanai_ann_nanncalc::ann_training_nooutput(nanmath::nanmath_vector &input,
                                                 nanmath::nanmath_vector &target,
                                                 const char *task) {
    struct ncommand ncmd;
    ncmd.cmd = NANNCALC_CMD_TRAINING_NOOUTPUT;
    ncmd.input = input;
    ncmd.target = target;
    if (task) ncmd.task = task;
    set_cmd(ncmd);
  }
  
  void nanai_ann_nanncalc::ann_configure(nanai_ann_nanndesc &desc) {
    struct ncommand ncmd;
    ncmd.cmd = NANNCALC_CMD_CONFIGURE;
    ncmd.desc = desc;
    set_cmd(ncmd);
  }
  
  void nanai_ann_nanncalc::ann_exchange(const nanai_ann_nanncalc::ann_t &ann) {
    struct ncommand ncmd;
    ncmd.cmd = NANNCALC_CMD_ANN_EXCHANGE;
    ncmd.ann = ann;
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
      error(NANAI_ERROR_RUNTIME_JOIN_THREAD);
    }
    
    // assert CURR ST == ST_STOP
  }
  
  void nanai_ann_nanncalc::ann_destroy() {
    _task.clear();
    _output_error.destroy();
    _alg.clear();
    
    _ann.ninput = 0;
    _ann.nhidden = 0;
    _ann.noutput = 0;
    _ann.nneural.clear();
    _ann.weight_matrix.clear();
    _ann.delta_weight_matrix.clear();
    
    _hiddens.clear();
    _outputs.clear();
    
    _fptr_input_filter = nullptr;
    _fptr_result = nullptr;
    _fptr_output_error = nullptr;
    _fptr_calculate = nullptr;
    
    _fptr_hidden_inits = nullptr;
    _fptr_hidden_calcs = nullptr;
    _fptr_hidden_errors = nullptr;
    _fptr_hidden_adjust_weights = nullptr;
    
    _callback_monitor_except = nullptr;
    _callback_monitor_trained = nullptr;
    _callback_monitor_trained_nooutput = nullptr;
    _callback_monitor_calculated = nullptr;
    _callback_monitor_progress = nullptr;
    _callback_monitor_alg_uninstall = nullptr;
  }
  
  void nanai_ann_nanncalc::ann_wait(int st, int slt) {
    if ((st < NANNCALC_ST_STOP) || (st > NANNCALC_ST_CONFIGURED)) {
      error(NANAI_ERROR_LOGIC_INVALID_ARGUMENT);
    }
    
    while (_state != st) {
      sleep(slt);
    }
  }
  
  std::string nanai_ann_nanncalc::get_task_name() const {
    return _task;
  }
  
  std::string nanai_ann_nanncalc::get_alg_name()  const {
    return _alg;
  }
  
  size_t nanai_ann_nanncalc::get_cmdlist_count()  const {
    return _cmdlist.size();
  }
  
  void nanai_ann_nanncalc::set_cmd(struct ncommand &ncmd, bool lock) {
    int err = 0;
    
    if (lock) {
      err = pthread_mutex_lock(&_cmdlist_lock);
      if (err != 0) {
        error(NANAI_ERROR_RUNTIME_LOCK_MUTEX);
      }
    }
    
    _cmdlist.push(ncmd);
    
    if (lock) {
      err = pthread_mutex_unlock(&_cmdlist_lock);
      if (err != 0) {
        error(NANAI_ERROR_RUNTIME_UNLOCK_MUTEX);
      }
    }
  }
  
  int nanai_ann_nanncalc::get_cmd(struct ncommand &ncmd, bool lock) {
    int err = 0, cmd = 0;
    err = pthread_mutex_lock(&_cmdlist_lock);
    if (err != 0) {
      error(NANAI_ERROR_RUNTIME_LOCK_MUTEX);
    }
    
    if (_cmdlist.empty()) {
      cmd = NANNCALC_ST_WAITING;
      goto _end;
    }
    
    ncmd = _cmdlist.front();
    _cmdlist.pop();
    cmd = ncmd.cmd;
    
  _end:
    err = pthread_mutex_unlock(&_cmdlist_lock);
    if (err != 0) {
      error(NANAI_ERROR_RUNTIME_UNLOCK_MUTEX);
    }
    
    return cmd;
  }
  
  void nanai_ann_nanncalc::set_state(int st, bool lock) {
    
    if (lock) {
      if (pthread_mutex_lock(&_state_lock) != 0) {
        error(NANAI_ERROR_RUNTIME_LOCK_MUTEX);
      }
    }
    
    _state = st;
    
    if (lock) {
      if (pthread_mutex_unlock(&_state_lock) != 0) {
        error(NANAI_ERROR_RUNTIME_UNLOCK_MUTEX);
      }
    }
  }
  
  int nanai_ann_nanncalc::get_state(bool lock) {
    
    if (lock) {
      if (pthread_mutex_lock(&_state_lock) != 0) {
        error(NANAI_ERROR_RUNTIME_LOCK_MUTEX);
      }
    }
    
    int ret = _state;
    
    if (lock) {
      if (pthread_mutex_unlock(&_state_lock) != 0) {
        error(NANAI_ERROR_RUNTIME_UNLOCK_MUTEX);
      }
    }
    
    return ret;
  }
  
  void nanai_ann_nanncalc::set_output(nanmath::nanmath_vector &output, bool lock) {
    if (&output == &nanmath::nv_null) {
      return;
    }
    
    if (lock) {
      if (pthread_mutex_lock(&_outputs_lock) != 0) {
        error(NANAI_ERROR_RUNTIME_LOCK_MUTEX);
      }
    }
    
    _outputs.push_back(output);
    
    if (lock) {
      if (pthread_mutex_unlock(&_outputs_lock) != 0) {
        error(NANAI_ERROR_RUNTIME_UNLOCK_MUTEX);
      }
    }
  }
  
  size_t nanai_ann_nanncalc::get_output(nanmath::nanmath_vector &output, bool lock) {
    size_t ret = 0;
    
    if (lock) {
      if (pthread_mutex_lock(&_outputs_lock) != 0) {
        error(NANAI_ERROR_RUNTIME_LOCK_MUTEX);
      }
    }
    
    if (_outputs.empty()) {
      output.destroy();
      return 0;
    }
    
    output = *(_outputs.end());
    _outputs.pop_back();
    ret = _outputs.size();
    
    if (lock) {
      if (pthread_mutex_unlock(&_outputs_lock) != 0) {
        error(NANAI_ERROR_RUNTIME_UNLOCK_MUTEX);
      }
    }
    
    return ret;
  }
  
  void nanai_ann_nanncalc::do_configure(nanai_ann_nanndesc &desc) {
    ann_create(desc);
    ann_log("configure successful");
  }
  
  void nanai_ann_nanncalc::do_ann_exchange(const nanai_ann_nanncalc::ann_t &ann) {
    _ann.ninput = ann.ninput;
    _ann.nhidden = ann.nhidden;
    _ann.noutput = ann.noutput;
    _ann.nneural = ann.nneural;
    _ann.weight_matrix = ann.weight_matrix;
    _ann.delta_weight_matrix = ann.delta_weight_matrix;
    ann_log("ann exchanged");
  }
  
  nanai_ann_nanncalc::ann_t nanai_ann_nanncalc::get_ann(bool lock) {
    if (lock) {
      if (pthread_mutex_lock(&_ann_lock) != 0) {
        error(NANAI_ERROR_RUNTIME_LOCK_MUTEX);
      }
    }
    
    nanai_ann_nanncalc::ann_t ret = _ann;
    
    if (lock) {
      if (pthread_mutex_unlock(&_ann_lock) != 0) {
        error(NANAI_ERROR_RUNTIME_UNLOCK_MUTEX);
      }
    }
    
    return ret;
  }
  
  void nanai_ann_nanncalc::do_stop() {
    ann_log("nanncalc will be stop");
  }
  
  /* 重写基类的虚函数 */
  void nanai_ann_nanncalc::on_error(int err) {
    ann_on_except(err);
  }
  
  void nanai_ann_nanncalc::ann_create(nanai_ann_nanndesc &desc) {
    ann_destroy();
    
    _alg = desc.name;
    _ann.ninput = desc.ninput;
    _ann.nhidden = (desc.nhidden > MAX_HIDDEN_NUMBER) ? MAX_HIDDEN_NUMBER : desc.nhidden;
    _ann.noutput = desc.noutput;
    _output_error.create(_ann.noutput);
    
    _fptr_input_filter = desc.fptr_input_filter;
    _fptr_result = desc.fptr_result;
    _fptr_output_error = desc.fptr_output_error;
    _fptr_calculate = desc.fptr_calculate;
    
    _fptr_hidden_inits = desc.fptr_hidden_inits;
    _fptr_hidden_calcs = desc.fptr_hidden_calcs;
    _fptr_hidden_errors = desc.fptr_hidden_errors;
    _fptr_hidden_adjust_weights = desc.fptr_hidden_adjust_weights;

    _callback_monitor_except = desc.callback_monitor_except;
    _callback_monitor_trained = desc.callback_monitor_trained;
    _callback_monitor_trained_nooutput = desc.callback_monitor_trained_nooutput;
    _callback_monitor_calculated = desc.callback_monitor_calculated;
    _callback_monitor_progress = desc.callback_monitor_progress;
    
    /* 创建神经网络 */
    ann_log("--------------------------------------------------");
    ann_log("ann start creating");
    ann_log("algorithm = %s", _alg.c_str());
    ann_log("ninput = %d", _ann.ninput);
    ann_log("nhidden = %d", _ann.nhidden);
    ann_log("noutput = %d", _ann.noutput);
    ann_log("total of weight matrix = %d", _ann.nhidden+1);
    
    _hiddens.resize(_ann.nhidden);
    _ann.weight_matrix.resize(_ann.nhidden + 1);
    _ann.delta_weight_matrix.resize(_ann.nhidden + 1);
    for (int i = 0; i < _ann.nhidden + 1; i++) {
      /* 隐藏层的个数要比需要的权重矩阵个数少1 */
      if (i < _ann.nhidden) {
        _ann.nneural.push_back(desc.nneure[i]);/* 是否不需要nneural */
        _hiddens[i].create(_ann.nneural[i]);
        ann_log("%dth hidden has %d neurals", i+1, _ann.nneural[i]);
      }
      
      /*
       * 根据隐藏层所处的位置不同，进行不同的权值矩阵的构造
       */
      if (i == 0) {
        /* 处理第一层，第一层是由输入向量 与 第一个隐藏层之间的权值矩阵 */
        _ann.weight_matrix[i].create(_ann.ninput, _ann.nneural[i]);
      } else if (i == _ann.nhidden) {
        /* 处理最后一层，最后一层是最后一个隐藏层 与 输出向量之间的权值矩阵 
         * 因为权值矩阵的个数要比隐藏层的个数多一个，所以到达这个时候要获取
         * 最后一个隐藏层必须退回一个索引
         */
        _ann.weight_matrix[i].create(_ann.nneural[i-1], _ann.noutput);
      } else {
        /* 其余是隐藏层 与 隐藏层之间的权值矩阵 */
        _ann.weight_matrix[i].create(_ann.nneural[i-1], _ann.nneural[i]);
      }
      
      /* 初始化当前层的权值矩阵 */
      _ann.delta_weight_matrix[i].zero();
      if (_fptr_hidden_inits) {
        _fptr_hidden_inits(i, &_ann.weight_matrix[i]);
      } else {
        _ann.weight_matrix[i].random();
      }
    }
    
    ann_log("ann create successful");
  }
  
  void nanai_ann_nanncalc::ann_on_except(int err) {
    if (_callback_monitor_except) {
      _callback_monitor_except(_cid, _task.c_str(), err, this);
    }
    
    /* FIXME */
    printf("!!! what fuck~~~, except happened, with errcode:%d", err);
  }
  
  void nanai_ann_nanncalc::ann_on_trained() {
    if (_callback_monitor_trained) {
      _callback_monitor_trained(_cid, _task.c_str(), this);
    }
    
    ann_log("training completed");
  }
  
  void nanai_ann_nanncalc::ann_on_trained_nooutput() {
    if (_callback_monitor_trained_nooutput) {
      _callback_monitor_trained_nooutput(_cid, _task.c_str(), this);
    }
    
    ann_log("training without output completed");
  }
  
  void nanai_ann_nanncalc::ann_on_calculated() {
    if (_callback_monitor_calculated) {
      _callback_monitor_calculated(_cid, _task.c_str(), this);
    }
    
    ann_log("calculate completed");
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
    
    /* 写入到系统日志 */
    fprintf(_log_file, "<%d>[%s]:%s\n", _cid, _task.c_str(), log);
    fflush(_log_file);
    
    /* 写入到用户自定义日志 */
    if (_callback_monitor_progress) {
      const char *tn = nullptr;
      if (_task.empty() == false)
        tn = _task.c_str();
      
      _callback_monitor_progress(_cid, tn, NANNCALC_PROCESS_LOG, log);
    }
  }
  
  void nanai_ann_nanncalc::ann_calculate(const char *task,
                                         nanmath::nanmath_vector &input,
                                         nanmath::nanmath_vector *target,
                                         nanmath::nanmath_vector *output) {
    if ((task) && (_task.empty())) {
      _task = task;
    } else if ((task == nullptr) && (_task.empty())) {
      error(NANAI_ERROR_LOGIC_TASK_NOT_MATCHED);
    }
    
    if (_fptr_calculate) {
      _fptr_calculate((void*)_task.c_str(), &input, target, output, this);
      return;
    }
    
    nanmath::nanmath_vector i, o, d_o;
    
    /* 插件输入函数过滤 */
    if (_fptr_input_filter) {
      _fptr_input_filter(&input, &i);
    } else {
      i = input;
    }
    
    /* 前馈计算 */
    ann_forward(i, o);
    
    /* 进行权值调整 */
    if (target) {
      d_o = ann_output_error(*target, o);     /* 计算误差 */
      ann_hiddens_error(i, d_o);              /* 调整每个权值矩阵 */
    }
    
    /* 输出结果 */
    if (output) {
      if (_fptr_result) {
        _fptr_result(&o, output);
      } else {
        *output = o;
      }
    }
  }
  
  void nanai_ann_nanncalc::ann_forward(nanmath::nanmath_vector &input, nanmath::nanmath_vector &output) {
    nanmath::nanmath_vector l1;
    nanmath::nanmath_vector l2;
    nanmath::nanmath_matrix wm;
    
    /* 运算的次数(权值矩阵的个数)要比隐藏层数多1 */
    for (int i = 0; i < _ann.nhidden+1; i++) {
      if (i == 0) {
        l1 = input;
        l2 = _hiddens[i];
        wm = _ann.weight_matrix[i];
      } else if (i == _ann.nhidden) {
        l1 = _hiddens[i-1];
        l2 = output;
        wm = _ann.weight_matrix[i];
      } else {
        l1 = _hiddens[i-1];
        l2 = _hiddens[i];
        wm = _ann.weight_matrix[i];
      }
      
      if (_fptr_hidden_calcs) {
        ann_layer_forward(i, l1, l2, wm, _fptr_hidden_calcs);
      }
      
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
                                             fptr_ann_hidden_calc calc) {
    nanmath::nanmath_vector result;
    result = wm.left_mul(l1);
    
    if (result.size() != l2.size()) {
      error(NANAI_ERROR_LOGIC_INVALID_ARGUMENT);
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
      error(NANAI_ERROR_LOGIC_INVALID_ARGUMENT);
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
  
  nanmath::nanmath_vector nanai_ann_nanncalc::ann_calc_hidden_delta(size_t h,
                                                                    nanmath::nanmath_vector &delta_k,
                                                                    nanmath::nanmath_matrix &w_kh,
                                                                    nanmath::nanmath_vector &o_h) {
    nanmath::nanmath_vector delta_h(o_h.size());
    nanmath::nanmath_vector delta_sum = w_kh.right_mul(delta_k);
    
    for (int i = 0; i < o_h.size(); i++) {
      delta_h.set(i, o_h[i] * (1 - o_h[i]) * delta_sum[i]);
    }
    
    return delta_h;
  }
  
  void nanai_ann_nanncalc::ann_hiddens_error(nanmath::nanmath_vector &input, nanmath::nanmath_vector &output_delta) {
    /* 遍历所有隐藏权值矩阵 */
    nanmath::nanmath_vector delta_k = output_delta;
    nanmath::nanmath_vector i;
    
    /* 隐藏层比权值矩阵少一个 */
    for (size_t h = _ann.weight_matrix.size() - 1; h > 0; h--) {
      /* 所需重要的变量
       * h 处于第几个权值矩阵运算
       * delta_k 上一层的误差向量
       * w_kh h->k的权值矩阵
       * o_h 当前隐藏单元的h的值
       *
       * 目的是求出delta_h，当前隐藏层的相对于上一次的偏差
       */
      
      if (h == 1) {
        /* 上一层就是输入层 */
        i = input;
      } else {
        i = _hiddens[h-1];
      }
      
      if (_fptr_hidden_errors) {
        _fptr_hidden_errors((int)h, &delta_k, &_ann.weight_matrix[h], &i, &delta_k);
      } else {
        delta_k = ann_calc_hidden_delta(h, delta_k, _ann.weight_matrix[h], i);
      }
      
      /* 调整当前的权值矩阵 */
      if (_fptr_hidden_adjust_weights) {
        _fptr_hidden_adjust_weights((int)h, &i, &delta_k,
                                       &_ann.weight_matrix[h], &_ann.delta_weight_matrix[h]);
      } else {
        ann_adjust_weight(h, i, delta_k);
      }
    }
  }
  
  void nanai_ann_nanncalc::ann_adjust_weight(size_t h,
                                             nanmath::nanmath_vector &layer,
                                             nanmath::nanmath_vector &delta) {
    static const double s_eta = 0.05;/* 学习速率 */
    static const double momentum = 0.03;/* 冲量项 */
    nanmath::nanmath_matrix &wm = _ann.weight_matrix[h];
    nanmath::nanmath_matrix &prev_dwm = _ann.delta_weight_matrix[h];
    
    /* 这里是遍历列向量 */
    for (int i = 0; i < delta.size(); i++) {          /* 矩阵的列 */
      for (int j = 0; j < layer.size(); j++) {        /* 矩阵的行 */
        /* 让上一层的每个输入向量都乘以当前的偏差值
         * 然后在修订这个偏差值的权向量
         */
        double new_dw = (s_eta * delta[i] * layer[j]) + (momentum * prev_dwm[j][i]);
        wm[j][i] += new_dw;
        prev_dwm[j][i] = new_dw;
      }
    }/* end for */
  }
  
}