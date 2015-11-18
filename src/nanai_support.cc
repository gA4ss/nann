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
    
    while (curr) {
      if (strcmp(curr->string, "samples") == 0) {
        cJSON *vec = curr->child;
        if (vec == nullptr) error(NANAI_ERROR_LOGIC_INVALID_CONFIG);
        while (vec) {
          cJSON *val = vec->child;
          inputed = false;
          targeted = false;
          count = 0;
          
          while (val) {
            input.clear();
            target.clear();
            
            /* 如果,没有成对出现，则默认填充一个空的 */
            if (strcmp(val->string, "input") == 0) {
              input = s_read_vector(val->child);
              samples.push_back(input);
              inputed = true;
              count++;
            } else if (strcmp(val->string, "target") == 0) {
              target = s_read_vector(val->child);
              targets.push_back(target);
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
            targets.push_back(target);
          }
          
          vec = vec->next;
        }/* end while */
      }
      curr = curr->next;
    }/* end while */
    
    if (samples.size() != targets.size()) {
      error(NANAI_ERROR_LOGIC_INVALID_CONFIG);
    }
    
  }
  
  void nanai_support_input_json(const std::string &json_context,
                                std::vector<nanmath::nanmath_vector> &inputs,
                                std::vector<nanmath::nanmath_vector> &targets) {
    if (json_context.empty()) {
      error(NANAI_ERROR_LOGIC_INVALID_ARGUMENT);
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
      error(NANAI_ERROR_RUNTIME_OPEN_FILE);
    }
    file.seekg(0, std::ios::end);
    size_t r = static_cast<size_t>(file.tellg());
    file.close();
    return r;
  }
  
}
