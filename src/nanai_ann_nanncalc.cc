#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>

#include "nanai_common.h"
#include "nanai_ann_nanncalc.h"

namespace nanai {
  
  void *nanai_ann_thread_worker(void *arg) {
    // assert
    int state;
    nanai_ann_nanncalc *calc = (nanai_ann_nanncalc*)arg;
    
    while ((state = calc->get_state()) != NANNCALC_ST_STOP) {
      
      if (state == NANNCALC_ST_WAITING) {
        sleep(100);
      }
      
      calc->lock();
      
      if (state == NANNCALC_ST_TRAINING) {
        
      } else if (state == NANNCALC_ST_TRAINING) {
        
      } else if (state == NANNCALC_ST_TRAINING_NOTARGET) {
        
      } else if (state == NANNCALC_ST_CONFIGURE) {
        calc->ann_free();
        
      } else {
        // error
      }
      
      calc->unlock();
    }
    
    return NULL;
  }
  
  /*
   * 获取创建时间
   * 创建唯一ID
   * 创建日志文件
   */
  nanai_ann_nanncalc::nanai_ann_nanncalc(const char *lp) {
    _birthday = time(NULL);
    srandom((unsigned)_birthday);
    _cid = nanai_support_nid(*(int*)this);
    
    if (lp) {
      _log_dir = lp;
    } else {
      _log_dir = "./";
    }
    
    _state = NANNCALC_ST_WAITING;
    
    /* 创建工作线程 */
    int err = pthread_create(&_thread_worker, &_thread_attr_worker,
                             nanai_ann_thread_worker, (void *)this);
    if (err != 0) {
      // error
    }
  }
  
  nanai_ann_nanncalc::~nanai_ann_nanncalc() {
    
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
  
  void nanai_ann_nanncalc::ann_free() {
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