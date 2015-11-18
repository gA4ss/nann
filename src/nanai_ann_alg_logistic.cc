#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <fstream>
#include <vector>

#include <nanai_ann_nanndesc.h>
#include <nanai_ann_alg_logistic.h>
#include <nanai_mapreduce.h>
#include <nanai_mapreduce_ann.h>
#include <nanai_ann_nanncalc.h>

#include "cJSON.h"

nanai::nanai_ann_nanndesc nanai_ann_alg_logistic_desc;
static double g_eta = 0.05;
static double g_momentum = 0.03;
//std::ofstream g_logfile;

void ann_input_filter(nanmath::nanmath_vector *input,
                      nanmath::nanmath_vector *input_filted) {
  *input_filted = *input;
}

void ann_result(nanmath::nanmath_vector *output, nanmath::nanmath_vector *result) {
  *result = *output;
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

void ann_hidden_adjust_weight(int h,                                    /*!< 第几个隐藏层 */
                              nanmath::nanmath_vector *layer,           /*!< 输入层向量 */
                              nanmath::nanmath_vector *delta,           /*!< 误差向量 */
                              nanmath::nanmath_matrix *wm,              /*!< 要调节的权值矩阵 */
                              nanmath::nanmath_matrix *prev_dwm         /*!< 保存偏差矩阵 */
                              ) {
  if (wm->col_size() != delta->size() ||
      (wm->row_size() != layer->size())) {
    nanai::error(NANAI_ERROR_LOGIC_ANN_INVALID_MATRIX_DEGREE);
  }
                                
  /* 这里是遍历列向量 */
  for (size_t i = 0; i < delta->size(); i++) {          /* 矩阵的列 */
    for (size_t j = 0; j < layer->size(); j++) {        /* 矩阵的行 */
      /* 让上一层的每个输入向量都乘以当前的偏差值
       * 然后在修订这个偏差值的权向量
       */
      double new_dw = (g_eta * delta->at(i) * layer->at(j)) + (g_momentum * prev_dwm->at(j, i));
      double t = wm->at(j, i) + new_dw;
      wm->set(j, i, t);
      prev_dwm->set(j, i, new_dw);
    }
  }
}

void ann_monitor_except(int cid,
                        const char *task,
                        int errcode,
                        nanai::nanai_ann_nanncalc *arg) {
}

void ann_monitor_trained(int cid,
                         const char *task,
                         nanmath::nanmath_vector *input,
                         nanmath::nanmath_vector *target,
                         nanmath::nanmath_vector *output,
                         nanai::nanai_ann_nanncalc::ann_t *ann) {
}

void ann_monitor_progress(int cid,
                          const char *task,
                          int progress,
                          void *arg) {
  /* 日志进度 */
  if (progress == NANNCALC_PROCESS_LOG) {
    if (task)
      printf("ann_alg_logistic - <%d>[%s]:%s\n", cid, task, (char*)arg);
    else
      printf("ann_alg_logistic - <%d>:%s\n", cid, (char*)arg);
  }
}

void ann_monitor_alg_uninstall(int cid) {
}

int ann_calculate(const std::string *task,
                  const nanmath::nanmath_vector *input,
                  const nanmath::nanmath_vector *target,
                  nanmath::nanmath_vector *output,
                  nanai::nanai_ann_nanncalc *ann) {
  if ((task == nullptr) || (input == nullptr) || (target == nullptr)) {
    nanai::error(NANAI_ERROR_LOGIC_INVALID_ARGUMENT);
  }
  
  return NANAI_ANN_DESC_RETURN;
}

/* 算法主函数，在调用时调用 */
void ann_alg_logistic_added() {
}

void ann_alg_logistic_close() {
}

static nanai::nanai_ann_nanncalc *s_make(const nanai::nanai_ann_nanndesc &desc,
                                         const std::string &log_dir) {
  nanai::nanai_ann_nanncalc *calc = new nanai::nanai_ann_nanncalc(desc, log_dir.c_str());
  if (calc == NULL) {
    nanai::error(NANAI_ERROR_RUNTIME_ALLOC_MEMORY);
  }
  
  return calc;
}

int ann_alg_logistic_map(const std::string *task,
                         const nanai::nanai_mapreduce_ann::nanai_mapreduce_ann_config_t *config,
                         const std::vector<nanmath::nanmath_vector> *inputs,
                         const std::vector<nanmath::nanmath_vector> *targets,
                         const nanai::nanai_ann_nanncalc::ann_t *ann,
                         std::vector<nanai::nanai_ann_nanncalc::result_t> *map_results) {
  
  size_t count = inputs->size();
  nanai::nanai_ann_nanncalc::ann_t ann_ = *ann;
  
  if (config->wt == 0) {
    
    std::vector<std::shared_ptr<nanai::nanai_ann_nanncalc> > calcs;
    for (size_t i = 0; i < count; i++) {
      std::shared_ptr<nanai::nanai_ann_nanncalc>calc(s_make(config->desc, config->log_dir));
      calc->ann_training(*task, (*inputs)[i], (*targets)[i], ann_, &((*map_results)[i]));
      calcs.push_back(calc);
    }
    
    /* 等待所有计算结果执行完毕 */
    for (size_t i = 0; i < count; i++) {
      while ((*map_results)[i].first.size() == 0) {
        usleep(100);
      }
    }
    
    /* 停止所有计算结点 */
    for (auto i : calcs) {
      i->ann_stop();
    }
  } else {
    nanmath::nanmath_vector input;
    nanmath::nanmath_vector target;
    nanai::nanai_ann_nanncalc::result_t result;
    std::shared_ptr<nanai::nanai_ann_nanncalc>calc(s_make(config->desc, config->log_dir));
    for (size_t i = 0; i < count; i++) {
      input = (*inputs)[i];
      target = (*targets)[i];
      
      calc->ann_training(*task, input, target, ann_, &result);
      /* 直到训练完毕 */
      while (result.first.size() == 0) {
        usleep(100);
      }
      /* 更新神经网络 */
      ann_ = result.second;
      (*map_results)[i] = result;
      
      /* 清空临时变量 */
      result.first.clear();
      result.second.clear();
    }
    calc->ann_stop();
  }
  
  return NANAI_ANN_DESC_RETURN;
}

static nanmath::nanmath_matrix s_merge_delta_matrix(nanmath::nanmath_matrix &dmat1,
                                                    nanmath::nanmath_matrix &dmat2) {
  
  if ((dmat1.row_size() != dmat2.row_size()) ||
      (dmat1.col_size() != dmat2.col_size())) {
    nanai::error(NANAI_ERROR_LOGIC_ANN_INVALID_MATRIX_DEGREE);
  }
  
  nanmath::nanmath_matrix c(dmat1.row_size(), dmat1.col_size());
  for (size_t i = 0; i < dmat1.row_size(); i++) {
    for (size_t j = 0; j < dmat1.col_size(); j++) {
      /* 偏差越小在融合后的矩阵中所占值越大 */
      double delta_a = 1 - (dmat1[i][j] / dmat1[i][j] + dmat2[i][j]);
      double delta_b = 1 - (dmat2[i][j] / dmat1[i][j] + dmat2[i][j]);
      
      double c_n = delta_a * dmat1[i][j] + delta_b * dmat2[i][j];
      c.set(i, j, c_n);
    }
  }
  
  return c;
}

static nanmath::nanmath_matrix s_merge_matrix(nanmath::nanmath_matrix &mat1,
                                              nanmath::nanmath_matrix &mat2,
                                              nanmath::nanmath_matrix &dmat1,
                                              nanmath::nanmath_matrix &dmat2) {
  if ((mat1.row_size() != mat2.row_size()) ||
      (mat1.col_size() != mat2.col_size()) ||
      (mat1.row_size() != dmat1.row_size()) ||
      (mat1.col_size() != dmat2.col_size())) {
    nanai::error(NANAI_ERROR_LOGIC_ANN_INVALID_MATRIX_DEGREE);
  }
  
  nanmath::nanmath_matrix c(mat1.row_size(), mat1.col_size());
  for (size_t i = 0; i < mat1.row_size(); i++) {
    for (size_t j = 0; j < mat1.col_size(); j++) {
      double delta_a = 1 - (dmat1[i][j] / dmat1[i][j] + dmat2[i][j]);
      double delta_b = 1 - (dmat2[i][j] / dmat1[i][j] + dmat2[i][j]);
      
      double c_n = delta_a * mat1[i][j] + delta_b * mat2[i][j];
      c.set(i, j, c_n);
    }
  }
  
  return c;
}

static nanai::nanai_ann_nanncalc::ann_t s_merge_ann(nanai::nanai_ann_nanncalc::ann_t &a,
                                                    nanai::nanai_ann_nanncalc::ann_t &b) {
  nanai::nanai_ann_nanncalc::ann_t c;
  if (a.weight_matrixes.size() != b.weight_matrixes.size()) {
    nanai::error(NANAI_ERROR_LOGIC_ANN_INVALID_MATRIX_DEGREE);
  }
  
  if ((a.delta_weight_matrixes.size() == 0) ||
      (b.delta_weight_matrixes.size() == 0)) {
    nanai::error(NANAI_ERROR_LOGIC_ANN_INVALID_MATRIX_DEGREE);
  }
  
  c = a;
  std::vector<nanmath::nanmath_matrix> wm;
  std::vector<nanmath::nanmath_matrix> dwm;
  
  /* 遍历每层的权值矩阵 */
  for (size_t i = 0; i < a.weight_matrixes.size(); i++) {
    wm.push_back(s_merge_matrix(a.weight_matrixes[i], b.weight_matrixes[i],
                                a.delta_weight_matrixes[i], b.delta_weight_matrixes[i]));
    dwm.push_back(s_merge_delta_matrix(a.delta_weight_matrixes[i], b.delta_weight_matrixes[i]));
  }
  c.weight_matrixes = wm;
  c.delta_weight_matrixes = dwm;
  
  return c;
}

void s_merge_anns(std::vector<nanai::nanai_ann_nanncalc::ann_t> &anns,
                  nanai::nanai_ann_nanncalc::ann_t &ann) {
  
  if (anns.size() <= 1) {
    nanai::error(NANAI_ERROR_LOGIC_ANN_MERGE_NUMBER_LESS_2);
  }
  
  /* 进行合并 */
  ann = anns[0];
  for (size_t i = 1; i < anns.size(); i++) {
    ann = s_merge_ann(ann, anns[i]);
  }
}

nanmath::nanmath_vector s_merge_outputs(std::vector<nanmath::nanmath_vector> &outputs,
                                        nanmath::nanmath_vector &output) {
  output = outputs[0];
  for (size_t i = 1; i < outputs.size(); i++) {
    output = output.add(outputs[i]);
  }
  /* 取平均值 */
  output = output.mul(1 / outputs.size());
  return output;
}

int ann_alg_logistic_reduce(int wt,
                            const std::string *task,
                            std::vector<nanai::nanai_ann_nanncalc::result_t> *map_results,
                            nanai::nanai_ann_nanncalc::result_t *reduce_result) {
  
  if (wt == 1) {
    *reduce_result = map_results->back();
  } else {
    std::vector<nanai::nanai_ann_nanncalc::ann_t> anns;
    nanai::nanai_ann_nanncalc::ann_t ann;
    std::vector<nanmath::nanmath_vector> outputs;
    nanmath::nanmath_vector output;
    
    for (auto r : *map_results) {
      outputs.push_back(r.first);
      anns.push_back(r.second);
    }
    
    /* 等于1个不合并 */
    if (anns.size() < 2) {
      reduce_result->first = outputs.back();
      reduce_result->second = anns.back();
      return NANAI_ANN_DESC_RETURN;
    }
    
    /*
     * 结果合并
     */
    s_merge_outputs(outputs, output);
    s_merge_anns(anns, ann);
    reduce_result->first = output;
    reduce_result->second = ann;
  }
  
  return NANAI_ANN_DESC_RETURN;
}

nanai::nanai_ann_nanndesc *ann_alg_setup(const char *conf_dir) {
  return NULL;
}

static cJSON *parse_conf(char *text) {
  cJSON *json=cJSON_Parse(text);
  if (!json) {
    return nullptr;
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
  
  nanai_ann_alg_logistic_desc.fptr_input_filter = reinterpret_cast<nanai::fptr_ann_alg_input_filter>(ann_input_filter);
  nanai_ann_alg_logistic_desc.fptr_result = reinterpret_cast<nanai::fptr_ann_alg_result>(ann_result);
  nanai_ann_alg_logistic_desc.fptr_output_error = ann_output_error;
  nanai_ann_alg_logistic_desc.fptr_calculate = nullptr;
  
  nanai_ann_alg_logistic_desc.fptr_hidden_calcs = ann_hidden_calc;
  nanai_ann_alg_logistic_desc.fptr_hidden_errors = reinterpret_cast<nanai::fptr_ann_alg_hidden_error>(ann_hidden_error);
  nanai_ann_alg_logistic_desc.fptr_hidden_adjust_weights =
    reinterpret_cast<nanai::fptr_ann_alg_hidden_adjust_weight>(ann_hidden_adjust_weight);
  
  nanai_ann_alg_logistic_desc.callback_monitor_except = reinterpret_cast<nanai::fptr_ann_monitor_except>(ann_monitor_except);
  nanai_ann_alg_logistic_desc.callback_monitor_trained = reinterpret_cast<nanai::fptr_ann_monitor_trained>(ann_monitor_trained);
  nanai_ann_alg_logistic_desc.callback_monitor_progress = ann_monitor_progress;
  nanai_ann_alg_logistic_desc.callback_monitor_alg_uninstall = ann_monitor_alg_uninstall;
  
  nanai_ann_alg_logistic_desc.fptr_event_added = ann_alg_logistic_added;
  nanai_ann_alg_logistic_desc.fptr_event_close = ann_alg_logistic_close;
  
  nanai_ann_alg_logistic_desc.fptr_map =
    reinterpret_cast<nanai::fptr_ann_mapreduce_map>(ann_alg_logistic_map);
  nanai_ann_alg_logistic_desc.fptr_reduce =
    reinterpret_cast<nanai::fptr_ann_mapreduce_reduce>(ann_alg_logistic_reduce);
  
  if (conf_dir == nullptr) {
  } else {
    if (conf_file.back() != '/') {
      conf_file += '/';
    }
    conf_file += alg_name;
    conf_file += ".json";
    
    /* 读取json配置文件 */
    cJSON *json = parse_conf_file(conf_file);
    if (json == nullptr) {
      nanai::error(NANAI_ERROR_LOGIC_INVALID_CONFIG);
    }
    cJSON *json_child = json->child;
    if (json_child == nullptr) {
      nanai::error(NANAI_ERROR_LOGIC_INVALID_CONFIG);
    }
    
    if (strcmp(json_child->string, "ann") != 0) {
      nanai::error(NANAI_ERROR_LOGIC_INVALID_CONFIG);
    }
    
    if (json->child) {
      cJSON *json_next = json->child->child; /* ann第一个子结点 */
      while (json_next) {
        if (strcmp(json_next->string, "eta") == 0 ) {
          g_eta = json_next->valuedouble;
        } else if (strcmp(json_next->string, "momentum") == 0 ) {
          g_momentum = json_next->valuedouble;
        } else {
          // continue
        }
        
        json_next = json_next->next;
      }
    } else {
      nanai::error(NANAI_ERROR_LOGIC_INVALID_CONFIG);
    }
    
    if (json) {
      cJSON_Delete(json);
    }
    
  }
  
  return &nanai_ann_alg_logistic_desc;
}

