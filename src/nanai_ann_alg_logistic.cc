#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string>
#include <fstream>

#include <nanai_ann_nanndesc.h>
#include <nanai_ann_alg_logistic.h>
#include <nanai_ann_nanncalc.h>

#include "cJSON.h"

nanai::nanai_ann_nanndesc nanai_ann_alg_logistic_desc;

void ann_input_filter(void *input, void *input_filted) {
  /* TODO : 做归一化处理 */
}

void ann_result(nanmath::nanmath_vector *output, nanmath::nanmath_vector* result) {
}

void ann_hidden_init(int h, nanmath::nanmath_matrix *wmat) {
}

/* sigmoid */
static double s_sigmoid(double x) {
  return (1.0 / (1.0 + exp(-x)));
}

double ann_hidden_calc(int h, double input) {
  return s_sigmoid(input);
}

void ann_hidden_error(int h,
                      nanmath::nanmath_vector *delta_k,
                      nanmath::nanmath_matrix *w_kh,
                      nanmath::nanmath_vector *o_h,
                      nanmath::nanmath_vector *delta_h) {
  delta_h->create(o_h->size());
  nanmath::nanmath_vector delta_sum = w_kh->right_mul(*delta_k);
  
  for (size_t i = 0; i < o_h->size(); i++) {
    delta_h->set(i, o_h->at(i) * (1 - o_h->at(i)) * delta_sum[i]);
  }
}

double ann_output_error(double target, double output) {
  double delta = output * (1.0 - output) * (target - output);
  return delta;
}

void ann_hidden_adjust_weight(int h,
                              nanmath::nanmath_vector *layer,
                              nanmath::nanmath_vector *delta,
                              nanmath::nanmath_matrix *wm,
                              nanmath::nanmath_matrix *prev_dwm) {
  static const double s_eta = 0.05;/* 学习速率 */
  static const double momentum = 0.03;/* 冲量项 */
  
  /* 这里是遍历列向量 */
  for (size_t i = 0; i < delta->size(); i++) {          /* 矩阵的列 */
    for (size_t j = 0; j < layer->size(); j++) {        /* 矩阵的行 */
      /* 让上一层的每个输入向量都乘以当前的偏差值
       * 然后在修订这个偏差值的权向量
       */
      double new_dw = (s_eta * delta->at(i) * layer->at(j)) + (momentum * prev_dwm->at(j, i));
      double t = wm->at(j, i) + new_dw;
      wm->set(j, i, t);
      prev_dwm->set(j, i, new_dw);
    }
  }
}

void ann_monitor_except(int cid, const char *task, int errcode, nanai::nanai_ann_nanncalc *arg) {
  printf("ann_alg_logistic - <%d>[%s]:except with errcode - %d\n", cid, task, errcode);
}

void ann_monitor_trained(int cid, const char *task, nanai::nanai_ann_nanncalc *arg) {
  printf("ann_alg_logistic - <%d>[%s]:trained\n", cid, task);
}

void ann_monitor_trained_nooutput(int cid, const char *task, nanai::nanai_ann_nanncalc *arg) {
  printf("ann_alg_logistic - <%d>[%s]:trained\n", cid, task);
}

void ann_monitor_calculated(int cid, const char *task, nanai::nanai_ann_nanncalc *arg) {
  printf("ann_alg_logistic - <%d>[%s]:calculated\n", cid, task);
}

void ann_monitor_progress(int cid, const char *task, int progress, void *arg) {
  /* 日志进度 */
  if (progress == NANNCALC_PROCESS_LOG) {
    printf("ann_alg_logistic - <%d>[%s]:%s\n", cid, task, (char*)arg);
  }
}

void ann_monitor_alg_uninstall(int cid) {
  printf("ann_alg_logistic - <%d>:uninstalled\n", cid);
}

void ann_calculate(std::string *task,
                   nanmath::nanmath_vector *input,
                   nanmath::nanmath_vector *target,
                   nanmath::nanmath_vector *output,
                   nanai::nanai_ann_nanncalc *arg) {
  if (input == nullptr) {
    // error
  }
}

/* 算法主函数，在调用时调用 */
void ann_alg_logistic_main() {
  printf("ann_alg_logistic installed\n");
}

nanai::nanai_ann_nanndesc *ann_alg_setup(const char *conf_dir) {
  return NULL;
}

static cJSON *parse_conf(char *text) {
  char *out;cJSON *json;
  
  json=cJSON_Parse(text);
  if (!json) {
    printf("Error before: [%s]\n",cJSON_GetErrorPtr());
    return nullptr;
  }
  if (json) {
    out=cJSON_Print(json);
    printf("%s\n",out);
    free(out);
  }
  
  return json;
}

static cJSON *parse_conf_file(const std::string &filename) {
  std::ifstream file;
  file.open(filename, std::ios::in|std::ios::binary);
  if (file.is_open() == false) {
    nanai::error(NANAI_ERROR_RUNTIME_OPEN_FILE);
  }
  
  /* 映射文件 */
  file.seekg(0, std::ios::end);
  long long filesize = file.tellg();
  
  char *buf = new char [filesize + 1];
  if (buf == nullptr) {
    nanai::error(NANAI_ERROR_RUNTIME_ALLOC_MEMORY);
  }
  
  file.seekg(0, std::ios::beg);
  file.read((char*)buf, filesize);
  cJSON *json = parse_conf(buf);
  file.close();
  delete [] buf;
  
  return json;
}

const char *alg_name = "ann_alg_logistic";
nanai::nanai_ann_nanndesc *ann_alg_logistic_setup(const char *conf_dir) {
  
  std::string conf_file = conf_dir;
  
  strcpy(nanai_ann_alg_logistic_desc.name, alg_name);
  strcpy(nanai_ann_alg_logistic_desc.description, "use logistic function");
  
  nanai_ann_alg_logistic_desc.fptr_input_filter = ann_input_filter;
  nanai_ann_alg_logistic_desc.fptr_result = (nanai::fptr_ann_result)ann_result;
  nanai_ann_alg_logistic_desc.fptr_output_error = ann_output_error;
  nanai_ann_alg_logistic_desc.fptr_calculate = nullptr;
  
  nanai_ann_alg_logistic_desc.fptr_hidden_inits = (nanai::fptr_ann_hidden_init)ann_hidden_init;
  nanai_ann_alg_logistic_desc.fptr_hidden_calcs = ann_hidden_calc;
  nanai_ann_alg_logistic_desc.fptr_hidden_errors = (nanai::fptr_ann_hidden_error)ann_hidden_error;
  nanai_ann_alg_logistic_desc.fptr_hidden_adjust_weights = (nanai::fptr_ann_hidden_adjust_weight)ann_hidden_adjust_weight;
  
  nanai_ann_alg_logistic_desc.callback_monitor_except = (nanai::fptr_ann_monitor_except)ann_monitor_except;
  nanai_ann_alg_logistic_desc.callback_monitor_trained = (nanai::fptr_ann_monitor_trained)ann_monitor_trained;
  nanai_ann_alg_logistic_desc.callback_monitor_trained_nooutput = (nanai::fptr_ann_monitor_trained2)ann_monitor_trained_nooutput;
  nanai_ann_alg_logistic_desc.callback_monitor_calculated = (nanai::fptr_ann_monitor_calculated)ann_monitor_calculated;
  nanai_ann_alg_logistic_desc.callback_monitor_progress = ann_monitor_progress;
  nanai_ann_alg_logistic_desc.callback_monitor_alg_uninstall = ann_monitor_alg_uninstall;
  nanai_ann_alg_logistic_desc.fptr_main = ann_alg_logistic_main;
  
  if (conf_dir == nullptr) {
    /* 使用默认配置 */
    nanai_ann_alg_logistic_desc.ninput = 5;
    nanai_ann_alg_logistic_desc.nhidden = 2;
    nanai_ann_alg_logistic_desc.noutput = 3;
    
    nanai_ann_alg_logistic_desc.nneure[0] = 6;
    nanai_ann_alg_logistic_desc.nneure[1] = 4;
  } else {
    if (conf_file.back() != '/') {
      conf_file += '/';
    }
    conf_file += alg_name;
    conf_file += ".json";
    
    /* 读取json配置文件 */
    cJSON *json = parse_conf_file(conf_file);
    if (json == nullptr) {
      // error();
    }
    cJSON *json_child = json->child;
    if (json_child == nullptr) {
      // error();
    }
    
    if (strcmp(json_child->string, "ann") != 0) {
      // error();
    }
    
    if (json->child) {
      cJSON *json_next = json->child->child; /* ann第一个子结点 */
      while (json_next) {
        if (strcmp(json_next->string, "ninput") == 0 ) {
          nanai_ann_alg_logistic_desc.ninput = json_next->valueint;
        } else if (strcmp(json_next->string, "noutput") == 0 ) {
          nanai_ann_alg_logistic_desc.noutput = json_next->valueint;
        } else if (strcmp(json_next->string, "nhidden") == 0 ) {
          nanai_ann_alg_logistic_desc.nhidden = json_next->valueint;
        } else if (strcmp(json_next->string, "nneure") == 0 ) {
          cJSON *tmp = json_next->child;
          for (int i = 0; i < nanai_ann_alg_logistic_desc.nhidden; i++) {
            nanai_ann_alg_logistic_desc.nneure[i] = tmp->valueint;
            tmp = tmp->next;
          }
        } else if (strcmp(json_next->string, "eta") == 0 ) {
        } else if (strcmp(json_next->string, "momentum") == 0 ) {
        } else {
          // continue
        }
        
        json_next = json_next->next;
      }
    } else {
      // error('ann no context');
    }
    
    if (json) {
      cJSON_Delete(json);
    }
  }
  
  return &nanai_ann_alg_logistic_desc;
}

