#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>

#include <nanai_common.h>
#include <nanai_ann_nanncalc.h>

namespace nanai {
  
  void *nanai_ann_nanncalc::thread_nanai_ann_worker(void *arg) {
    // assert
    
    nanai_ann_nanncalc *calc = (nanai_ann_nanncalc*)arg;
    struct nanai_ann_nanncalc::ncommand ncmd;
    int cmd = 0, err = 0;
    nanmath::nanmath_vector input;
    nanmath::nanmath_vector output;
    nanmath::nanmath_vector target;
    
    while (1) {
      
      calc->set_state(NANNCALC_ST_WAITING);
      
      if (calc->get_cmd(ncmd) == 0) {
        sleep(100);
        continue;
      }
      
      cmd = ncmd.cmd;
      
      if (cmd == NANNCALC_CMD_TRAINING) {
        calc->set_state(NANNCALC_ST_TRAINING);
        
        input = ncmd.input;
        target = ncmd.target;
        err = s_ann_calculate(input, target, output);
        if (err != 0) {
          // error
        }
        
        calc->set_output(output);
        
        calc->set_state(NANNCALC_ST_TRAINED);
      } else if (cmd == NANNCALC_CMD_TRAINING_NOTARGET) {
        calc->set_state(NANNCALC_ST_TRAINING);
        
        input = ncmd.input;
        err = s_ann_calculate(input, nanmath::nv_null, output);
        if (err != 0) {
          // error
        }
        
        calc->set_output(output);
        
        calc->set_state(NANNCALC_ST_TRAINED);
      } else if (cmd == NANNCALC_CMD_TRAINING_NOOUTPUT) {
        calc->set_state(NANNCALC_ST_TRAINING);
        
        input = ncmd.input;
        err = s_ann_calculate(input, target, nanmath::nv_null);
        if (err != 0) {
          // error
        }
        
        calc->set_output(output);
        
        calc->set_state(NANNCALC_ST_TRAINED);
      } else if (cmd == NANNCALC_CMD_CONFIGURE) {
        calc->set_state(NANNCALC_ST_CONFIGURING);
        
        /* 当前版本暂时 不支持*/
        //calc->ann_destroy();
        
        calc->set_state(NANNCALC_ST_CONFIGURED);
      } else if (cmd == NANNCALC_CMD_STOP) {
        break;
      } else {
        // warning
        continue;
      }
    }
    
    calc->set_state(NANNCALC_ST_STOP);
    pthread_exit(NULL);
  }
  
  nanai_ann_nanncalc::nanai_ann_nanncalc(const char *lp) : _state(NANNCALC_CMD_STOP) {
    int err = 0;
    _birthday = time(NULL);
    srandom((unsigned)_birthday);
    _cid = nanai_support_nid(*(int*)this);
    
    if (lp) {
      _log_dir = lp;
    } else {
      _log_dir = "./";
    }
    
    /* 创建工作线程 */
    err = pthread_mutex_init(_cmdlist_lock, NULL);
    if (err != 0) {
      // error
    }
    
    err = pthread_create(&_thread_worker, NULL,
                         thread_nanai_ann_worker, (void *)this);
    if (err != 0) {
      // error
    }
  }
  
  nanai_ann_nanncalc::~nanai_ann_nanncalc() {
    int err = 0;
    
    ann_stop();
    ann_destroy();
    
    err = pthread_mutex_destroy(_cmdlist_lock);
    if (err != 0) {
      // error
    }
    
    fflush(_log_file);
    fclose(_log_file);
  }
  
  int nanai_ann_nanncalc::ann_training(nanmath::nanmath_vector &input, nanmath::nanmath_vector &target,
                                       const char *task) {
    struct ncommand ncmd;
    ncmd.cmd = NANNCALC_CMD_TRAINING;
    ncmd.input = input;
    ncmd.target = target;
    ncmd.task = task;
    set_cmd(ncmd);
    return 0;
  }
  
  int nanai_ann_nanncalc::ann_training_notarget(nanmath::nanmath_vector &input,
                                                const char *task) {
    struct ncommand ncmd;
    ncmd.cmd = NANNCALC_CMD_TRAINING_NOTARGET;
    ncmd.input = input;
    ncmd.task = task;
    set_cmd(ncmd);
    return 0;
  }
  
  int nanai_ann_nanncalc::ann_configure(nanai_ann_nanndesc &desc) {
    struct ncommand ncmd;
    ncmd.cmd = NANNCALC_CMD_CONFIGURE;
    ncmd.desc = desc;
    set_cmd(ncmd);
    return 0;
  }
  
  int nanai_ann_nanncalc::ann_output(nanmath::nanmath_vector &output) {
    if (_outputs.empty()) {
      output.clear();
      return 0;
    }
    output = *(_outputs.end());
    _outputs.pop_back();
    return 0;
  }
  
  int nanai_ann_nanncalc::ann_stop() {
    if (_state == NANNCALC_ST_STOP) {
      return 0;
    }
    
    struct nanai_ann_nanncalc::ncommand ncmd;
    ncmd.cmd = NANNCALC_CMD_STOP;
    
    set_cmd(ncmd);
    
    void *tret = NULL;
    int err = pthread_join(_thread_worker, &tret);
    if (err != 0) {
      // error
    }
    
    // assert CURR ST == ST_STOP
    
    return 0;
  }
  
  int nanai_ann_nanncalc::ann_destroy() {
    _alg.clear();
    _nneural.clear();
    _task.clear();
    _input.clear();
    
    _ninput = 0;
    _nhidden = 0;
    _noutput = 0;
    
    std::vector<nanmath::nanmath_matrix>::iterator ann_iter;
    for (ann_iter = _ann.begin(); ann_iter < _ann.end(); ann_iter++) {
      (*ann_iter).clear();
    }
    _ann.clear();

    std::vector<nanmath::nanmath_vector>::iterator outputs_iter;
    for (outputs_iter = _outputs.begin(); outputs_iter < _outputs.end(); outputs_iter++) {
      (*outputs_iter).clear();
    }
    _outputs.clear();
    
    _hidden_init.clear();
    _hidden_calc.clear();
    _hidden_error.clear();
    
    return 0;
  }
  
  int nanai_ann_nanncalc::ann_wait(int st, int slt) {
    if ((st < NANNCALC_ST_STOP) || (st > NANNCALC_ST_CONFIGURED)) {
      // error
    }
    
    while (_state != st) {
      sleep(slt);
    }
    
    return 0;
    
  }
  
  int nanai_ann_nanncalc::ann_create(nanai_ann_nanndesc &desc) {
    
    ann_destroy();
    
    _alg = desc.name;
    _ninput = desc.ninput;
    _nhidden = (desc.nhidden > MAX_HIDDEN_NUMBER) ? MAX_HIDDEN_NUMBER : desc.nhidden;
    _noutput = desc.noutput;
    
    _taret_outfilter = desc.taret_outfilter;
    _callback_monitor_except = desc.callback_monitor_except;
    _callback_monitor_trained = desc.callback_monitor_trained;
    _callback_monitor_calculated = desc.callback_monitor_calculated;
    _callback_monitor_progress = desc.callback_monitor_progress;
    
    nanmath::nanmath_matrix a_weights;
    for (int i = 0; i < _nhidden + 1; i++) {
      /* 隐藏层的个数要比需要的权重矩阵个数少1 */
      if (i < _nhidden) {
        _nneural.push_back(desc.nneure[i]);
      }
      
      _hidden_init.push_back(desc.hidden_init[i]);
      _hidden_calc.push_back(desc.hidden_calc[i]);
      _hidden_error.push_back(desc.hidden_error[i]);
      
      /*
       * 根据隐藏层所处的位置不同，进行不同的权值矩阵的构造
       */
      if (i == 0) {
        /* 处理第一层，第一层是由输入向量 与 第一个隐藏层之间的权值矩阵 */
        a_weights.create(_ninput, _nneural[i]);
      } else if (i == _nhidden) {
        /* 处理最后一层，最后一层是最后一个隐藏层 与 输出向量之间的权值矩阵 
         * 因为权值矩阵的个数要比隐藏层的个数多一个，所以到达这个时候要获取
         * 最后一个隐藏层必须退回一个索引
         */
        a_weights.create(_nneural[i-1], _noutput);
      } else {
        /* 其余是隐藏层 与 隐藏层之间的权值矩阵 */
        a_weights.create(_nneural[i-1], _nneural[i]);
      }
      
      if (_hidden_init[i]) {
        _hidden_init[i](i, a_weights);
      }
      _ann.push_back(a_weights);
      a_weights.clear();
    }
    
    return 0;
  }
  
  void nanai_ann_nanncalc::ann_except(int err) {
    if (_callback_monitor_except) {
      _callback_monitor_except(_cid, _task.c_str(), err);
    }
  }
  
  void nanai_ann_nanncalc::ann_trained() {
    if (_callback_monitor_trained) {
      _callback_monitor_trained(_cid, _task.c_str(), _ann);
    }
  }
  
  void nanai_ann_nanncalc::ann_calculated() {
    if (_callback_monitor_calculated) {
      _callback_monitor_calculated(_cid, _task.c_str(), *(_outputs.end()));
    }
  }
  
  void nanai_ann_nanncalc::ann_process(int process, void *arg) {
    if (_callback_monitor_progress) {
      _callback_monitor_progress(_cid, _task.c_str(), process, arg);
    }
  }
  
  void nanai_ann_nanncalc::ann_log(const char *fmt, ...) {
    std::string log;
    if (_callback_monitor_progress) {
      _callback_monitor_progress(_cid, _task.c_str(), NANNCALC_PROCESS_LOG,
                                 (void*)log.c_str());
    }
  }
  
  int nanai_ann_nanncalc::read(void *nnn) {
    return 0;
  }
  
  int nanai_ann_nanncalc::write(void *nnn, int len) {
    return 0;
  }
  
  void nanai_ann_nanncalc::set_cmd(struct ncommand &ncmd) {
    int err = 0;
    err = pthread_mutex_lock(_cmdlist_lock);
    if (err != 0) {
      // error
    }
    
    _cmdlist.push(ncmd);
    
    err = pthread_mutex_unlock(_cmdlist_lock);
    if (err != 0) {
      // error
    }
  }
  
  int nanai_ann_nanncalc::get_cmd(struct ncommand &ncmd) {
    int err = 0, cmd = 0;
    err = pthread_mutex_lock(_cmdlist_lock);
    if (err != 0) {
      // error
    }
    
    if (_cmdlist.empty()) {
      cmd = NANNCALC_ST_WAITING;
      goto _end;
    }
    
    ncmd = _cmdlist.front();
    _cmdlist.pop();
    
  _end:
    err = pthread_mutex_unlock(_cmdlist_lock);
    if (err != 0) {
      // error
    }
    
    return cmd;
  }
  
  void nanai_ann_nanncalc::set_state(int st) {
    _state = st;
  }
  
  int nanai_ann_nanncalc::get_state() {
    return _state;
  }
  
  void nanai_ann_nanncalc::set_output(nanmath::nanmath_vector &output) {
    if (&output == &nanmath::nv_null) {
      return;
    }
    
    nanmath::nanmath_vector result;
    
    if (_taret_outfilter) {
      if (_taret_outfilter(output, result) != 0) {
        // error
      }
    }
    
    _outputs.push_back(result);
  }
  
  int nanai_ann_nanncalc::ann_create(int ninput, int nnetlayer, int noutput) {
    _ninput = ninput;
    _nnetlayer = nnetlayer;
    
    newnet->input_units = alloc_1d_dbl(n_in + 1);
    newnet->hidden_units = alloc_1d_dbl(n_hidden + 1);
    newnet->output_units = alloc_1d_dbl(n_out + 1);
    
    newnet->hidden_delta = alloc_1d_dbl(n_hidden + 1);
    newnet->output_delta = alloc_1d_dbl(n_out + 1);
    newnet->target = alloc_1d_dbl(n_out + 1);
    
    newnet->input_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
    newnet->hidden_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);
    
    newnet->input_prev_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
    newnet->hidden_prev_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);
    
    return newnet;
  }
  
  /* 所有层的向前运算 */
  int nanai_ann_nanncalc::s_ann_forward() {
    
    ann_layerforward(net->input_units, net->hidden_units,
                     net->input_weights, in, hid);
    ann_layerforward(net->hidden_units, net->output_units,
                      net->hidden_weights, hid, out);
  }
  
  void nanai_ann_nanncalc::ann_training(double *input, int ninput, double target) {
    
    /* 如果输入向量的个数大于类内定义的，则按类中来算 
     * 如果小于类内定义的，则末位补0
     */
    
    double out_err, hid_err;
    
    hid = net->hidden_n;
    out = net->output_n;
    
    /* 计算输出 */
    ann_forward();

    
    /* 计算目标与输出之间的误差 */
    bpnn_output_error(net->output_delta, net->target, net->output_units,
                      out, &out_err);
    
    /* 计算隐藏层与输出层的误差 */
    bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out,
                      net->hidden_weights, net->hidden_units, &hid_err);
    *eo = out_err;
    *eh = hid_err;
    
    /* 重新审核权重 */
    bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid,
                        net->hidden_weights, net->hidden_prev_weights, eta, momentum);
    bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in,
                        net->input_weights, net->input_prev_weights, eta, momentum);
  }
  
  
  /* 输出误差计算
   * delta:输出差距的向量
   * target:目标向量
   * output:输出向量
   * nj:输出向量的个数
   * err:误差的调整
   */
  static
  void bpnn_output_error(double *delta, double *target,
                         double *output, int nj, double *err) {
    int j;
    double o, t, errsum;
    
    errsum = 0.0;
    /* j=1表示跳过阀值 */
    for (j = 1; j <= nj; j++) {
      o = output[j];          /* 实际输出 */
      t = target[j];          /* 目标输出 */
      /* 计算偏差 */
      delta[j] = o * (1.0 - o) * (t - o);
      errsum += ABS(delta[j]);
    }
    /* 误差 */
    *err = errsum;
  }
  
  /* 隐藏层误差
   * delta_h:隐藏层偏差
   * nh:隐藏层个数
   * delta_o:输出层偏差
   * no:输出层个数
   * who:隐藏层权重矩阵
   * hidden:隐藏层
   * err:隐藏层误差
   */
  static
  void bpnn_hidden_error(double *delta_h, int nh, double *delta_o,
                         int no, double **who, double *hidden, double *err) {
    int j, k;
    double h, sum, errsum;
    
    errsum = 0.0;
    /* 隐藏层 */
    for (j = 1; j <= nh; j++) {
      h = hidden[j];
      sum = 0.0;
      /* 遍历输出向量，计算权和 */
      for (k = 1; k <= no; k++) {
        sum += delta_o[k] * who[j][k];
      }
      /* 计算隐藏层的偏差 */
      delta_h[j] = h * (1.0 - h) * sum;
      errsum += ABS(delta_h[j]);
    }
    *err = errsum;
  }
  
  /* 修订权重 */
  static
  void bpnn_adjust_weights(double *delta, int ndelta, double *ly, int nly,
                           double **w, double **oldw, double eta, double momentum) {
    double new_dw;
    int k, j;
    
    ly[0] = 1.0;
    for (j = 1; j <= ndelta; j++) {
      for (k = 0; k <= nly; k++) {
        new_dw = ((eta * delta[j] * ly[k]) + (momentum * oldw[k][j]));
        w[k][j] += new_dw;
        oldw[k][j] = new_dw;
      }
    }
  }
  
  /* 网络反馈 */
  void bpnn_feedforward(BPNN *net) {
    int in, hid, out;
    
    in = net->input_n;
    hid = net->hidden_n;
    out = net->output_n;
    
    bpnn_layerforward(net->input_units, net->hidden_units,
                      net->input_weights, in, hid);
    bpnn_layerforward(net->hidden_units, net->output_units,
                      net->hidden_weights, hid, out);
  }
  
  void nanai_ann_nanncalc::ann_layerforward(double *l1, double *l2, double **conn,
                                            int n1, int n2) {
    double sum;
    int j, k;
    
    /* 设置阀值单元 */
    l1[0] = 1.0;
    
    /* 每个第二层的单元,j=1,跳过阀值的权重 */
    for (j = 1; j <= n2; j++) {
      
      /* 计算输入的权和 */
      sum = 0.0;
      for (k = 0; k <= n1; k++) {
        sum += conn[k][j] * l1[k];
      }
      l2[j] = sigmoid(sum);
    }
  }
  
  void nanai_ann_nanncalc::ann_read(unsigned char *nnn) {
    
  }
  
  void nanai_ann_nanncalc::ann_write(unsigned char *nnn, int len) {
    
  }
}