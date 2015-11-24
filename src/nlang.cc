#include <nlang.h>
#include "cJSON.h"

namespace nlang {
  
  
  
  nlang_var::nlang_var() {
    type = NLANG_TYPE_NULL;
    bool_v = false;
    int_v = 0;
    double_v = 0.0;
  }
  
  nlang_var::~nlang_var() {
    clear();
  }
  
  void nlang_var::clear() {
    type = NLANG_TYPE_NULL;
    bool_v = false;
    int_v = 0;
    double_v = 0.0;
    string_v.clear();
  }

  /*******************************************************************************************************************/
  
  nlang::nlang() {
    
  }
  
  nlang::~nlang() {
    
  }
  
  void nlang::read(nlang_source_object source) {
    cJSON *json = static_cast<cJSON*>(source);
    cJSON *json_child = json->child;
    if (json_child == nullptr) {
    }
    
    cJSON *json_next = json_child;
    while (json_next) {
      std::string name = json_next->string;
      if (name.empty()) {
        json_next = json_next->next;
        continue;
      }
      
      if (json_next->type == cJSON_String) {
        set(name, json_next->string);
      } else if (json_next->type == cJSON_Number) {
        set(name, json_next->valuedouble);
      } else if (json_next->type == cJSON_False) {
        set(name, false);
      } else if (json_next->type == cJSON_True) {
        set(name, true);
      } else if (json_next->type == cJSON_NULL) {
        set_null(name);
      } else if (json_next->type == cJSON_Array) {
      } else if (json_next->type == cJSON_Object) {
      }
      
      json_next = json_next->next;
    }
  }

  void nlang::set_null(const std::string &name) {
    nlang_var tmp;
    if (_symbols.find(name) == _symbols.end()) {
      _symbols[name] = tmp;
    } else {
      _symbols[name].clear();
    }
  }
  
  void nlang::set(const std::string &name,
                  const bool v) {
    nlang_var tmp;
    
    tmp.type = NLANG_TYPE_BOOL;
    tmp.bool_v = v;
    
    _symbols[name] = tmp;
  }
  
  void nlang::set(const std::string &name,
                  const double v) {
    nlang_var tmp;
    
    tmp.type = NLANG_TYPE_NUMBER;
    tmp.double_v = v;
    tmp.int_v = static_cast<int>(v);
    
    _symbols[name] = tmp;
  }
  
  void nlang::set(const std::string &name,
                  const std::string &v) {
    nlang_var tmp;
    
    tmp.type = NLANG_TYPE_STRING;
    tmp.string_v = v;
    
    _symbols[name] = tmp;
  }
  
  int nlang::get(const std::string &name,
                 bool &v) {
    if (_symbols.find(name) == _symbols.end()) {
      return -1;
    }
    
    if (_symbols[name].type != NLANG_TYPE_NUMBER) {
      return -2;
    }
    
    v = _symbols[name].bool_v;
    
    return 0;
  }
  
  int nlang::get(const std::string &name,
                 int &v) {
    if (_symbols.find(name) == _symbols.end()) {
      return -1;
    }
    
    if (_symbols[name].type != NLANG_TYPE_NUMBER) {
      return -2;
    }
    
    v = _symbols[name].int_v;
    
    return 0;
  }
  
  int nlang::get(const std::string &name,
                 double &v) {
    if (_symbols.find(name) == _symbols.end()) {
      return -1;
    }
    
    if (_symbols[name].type != NLANG_TYPE_NUMBER) {
      return -2;
    }
    
    v = _symbols[name].double_v;
    
    return 0;
  }
  
  int nlang::get(const std::string &name,
                 std::string &v) {
    if (_symbols.find(name) == _symbols.end()) {
      return -1;
    }
    
    if (_symbols[name].type != NLANG_TYPE_STRING) {
      return -2;
    }
    
    v = _symbols[name].string_v;
    
    return 0;
  }
  
  int nlang::parse(std::string &source) {
    return 0;
  }
  
  int nlang::parse_object(std::string &source) {
    return 0;
  }

}