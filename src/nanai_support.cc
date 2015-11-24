#include <cstdio>
#include <fstream>
#include <nanai_common.h>
#include <nanai_ann_nanncalc.h>

#include "cJSON.h"

namespace nanai {
  int nanai_support_nid(int adr) {
    int r = rand();
    return r ^ adr;
  }
  
  int nanai_support_tid() {
    return 0;
  }
  
  static nanmath::nanmath_vector s_read_vector(cJSON *json) {
    nanmath::nanmath_vector res;
    while (json) {
      if (json->valuedouble) res.push(json->valuedouble);
      else res.push(static_cast<double>(json->valueint));
      json = json->next;
    }
    return res;
  }
  
  static void parse_samples(cJSON *json,
                            std::vector<nanmath::nanmath_vector> &samples,
                            std::vector<nanmath::nanmath_vector> &targets) {
    nanmath::nanmath_vector input, target;
    bool inputed = false, targeted = false;
    int count = 0;
    cJSON *curr = json->child;
    if (curr == nullptr) error(NANAI_ERROR_LOGIC_INVALID_CONFIG);
    int idx = 0;
    std::pair<int, std::pair<nanmath::nanmath_vector, nanmath::nanmath_vector> > idx_sample_target;
    std::vector<std::pair<int, std::pair<nanmath::nanmath_vector, nanmath::nanmath_vector> > > idx_samples_targets;
    
    while (curr) {
      if (strcmp(curr->string, "samples") == 0) {
        cJSON *vec = curr->child;
        if (vec == nullptr) error(NANAI_ERROR_LOGIC_INVALID_CONFIG);
        
        while (vec) {
          cJSON *val = vec->child;
          idx = atoi(vec->string);
          inputed = false;
          targeted = false;
          count = 0;
          
          while (val) {
            /* 如果,没有成对出现，则默认填充一个空的 */
            if (strcmp(val->string, "input") == 0) {
              input = s_read_vector(val->child);
              inputed = true;
              count++;
            } else if (strcmp(val->string, "target") == 0) {
              target = s_read_vector(val->child);
              targeted = true;
              count++;
            }
            val = val->next;
          }
          
          /* 一个sample项目中最多只能有两项 */
          if (count > 2) {
            error(NANAI_ERROR_LOGIC_INVALID_CONFIG);
          }
          
          /* 输入向量必须存在 */
          if (inputed == false) {
            error(NANAI_ERROR_LOGIC_INVALID_CONFIG);
          }
          
          if (targeted == false) {
            target.clear();
          }
          
          /* 保存临时值 */
          idx_sample_target.first = idx;
          idx_sample_target.second.first = input;
          idx_sample_target.second.second = target;
          idx_samples_targets.push_back(idx_sample_target);
          
          vec = vec->next;
        }/* end while */
      }
      curr = curr->next;
    }/* end while */
    
    /* 排序 */
    std::sort(idx_samples_targets.begin(),
              idx_samples_targets.end(),
              [](std::pair<int, std::pair<nanmath::nanmath_vector, nanmath::nanmath_vector> > &a,
                 std::pair<int, std::pair<nanmath::nanmath_vector, nanmath::nanmath_vector> > &b) {
      return a.first < b.first;
    });
    for (auto s : idx_samples_targets) {
      samples.push_back(s.second.first);
      targets.push_back(s.second.second);
    }
    
    if (samples.size() != targets.size()) {
      error(NANAI_ERROR_LOGIC_INVALID_CONFIG);
    }
  }
  
  void nanai_support_input_json(const std::string &json_context,
                                std::vector<nanmath::nanmath_vector> &inputs,
                                std::vector<nanmath::nanmath_vector> &targets) {
    if (json_context.empty()) {
      error(NAN_ERROR_LOGIC_INVALID_ARGUMENT);
    }
    
    cJSON *json = cJSON_Parse(json_context.c_str());
    if (json == nullptr) {
      error(NANAI_ERROR_LOGIC_INVALID_CONFIG);
    }
    parse_samples(json, inputs, targets);
    
    if (json) {
      cJSON_Delete(json);
    }
  }
  
  std::string nanai_support_just_filename(const std::string &path) {
    std::string jfilename;
    size_t f = path.rfind("/");
    if (f != std::string::npos) {
      jfilename = path.substr(f+1);
      jfilename = jfilename.substr(0, jfilename.rfind("."));
    }
    return jfilename;
  }
  
  size_t nanai_support_get_file_size(const std::string &path) {
    std::fstream file;
    
    file.open(path, std::ios::in);
    if (file.is_open() == false) {
      error(NAN_ERROR_RUNTIME_OPEN_FILE);
    }
    file.seekg(0, std::ios::end);
    size_t r = static_cast<size_t>(file.tellg());
    file.close();
    return r;
  }
  
}
