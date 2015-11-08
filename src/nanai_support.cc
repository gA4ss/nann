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
  
  static void parse_samples(cJSON *json,
                            std::vector<nanmath::nanmath_vector> &samples,
                            nanmath::nanmath_vector *target) {
    nanmath::nanmath_vector input;
    int ncol = 0, nprev = 0;
    cJSON *curr = json->child;
    if (curr == nullptr) error(NANAI_ERROR_LOGIC_INVALID_CONFIG);
    
    while (curr) {
      if (strcmp(curr->string, "samples") == 0) {
        cJSON *vec = curr->child;
        if (vec == nullptr) error(NANAI_ERROR_LOGIC_INVALID_CONFIG);
        while (vec) {
          cJSON *val = vec->child;
          
          ncol = 0;
          input.clear();
          while (val) {
            if (val->valuedouble) input.push(val->valuedouble);
            else input.push(val->valueint);
            ncol++;
            val = val->next;
          }
          
          /* 每个输入的向量个数必须一样 */
          if (nprev) {
            if (nprev != ncol) error(NANAI_ERROR_LOGIC_INVALID_CONFIG);
          }
          nprev = ncol;
          samples.push_back(input);
          vec = vec->next;
        }
        
      } else if (strcmp(curr->string, "target") == 0) {
        
        if (target == nullptr) {
          continue;
        }
        
        cJSON *vec = curr->child;
        if (vec == nullptr) error(NANAI_ERROR_LOGIC_INVALID_CONFIG);
        
        /* 读入一条向量 */
        while (vec) {
          if (vec->valuedouble) target->push(vec->valuedouble);
          else target->push(vec->valueint);
          vec = vec->next;
        }
        
      }
      
      curr = curr->next;
    }
  }
  
  void nanai_support_input_json(const std::string &json_context,
                                std::vector<nanmath::nanmath_vector> &inputs,
                                nanmath::nanmath_vector *target) {
    if (json_context.empty()) {
      error(NANAI_ERROR_LOGIC_INVALID_ARGUMENT);
    }
    
    cJSON *json = cJSON_Parse(json_context.c_str());
    if (json == nullptr) {
      error(NANAI_ERROR_LOGIC_INVALID_CONFIG);
    }
    parse_samples(json, inputs, target);
    
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
