#include <nanai_object.h>

namespace nanai {
    
  nanai_object::nanai_object() : nanan::nan_object(){
    register_error(NANAI_ERROR_LOGIC_INVALID_CONFIG, "invalid config");
    register_error(NANAI_ERROR_LOGIC_ALG_NOT_FOUND, "algorithm not found");
    register_error(NANAI_ERROR_LOGIC_TASK_NOT_MATCHED, "task not matched");
    register_error(NANAI_ERROR_LOGIC_TASK_ALREADY_EXIST, "task already exist");
    register_error(NANAI_ERROR_LOGIC_HOME_DIR_NOT_CONFIG, "home dir not config");
    register_error(NANAI_ERROR_LOGIC_DESC_FUNCTION_NOT_FOUND, "desc function not found");
    
    register_error(NANAI_ERROR_LOGIC_ANN_MERGE_NUMBER_LESS_2, "ann merge number < 2");
    register_error(NANAI_ERROR_LOGIC_ANN_INVALID_VECTOR_DEGREE, "ann invalid vector degree");
    register_error(NANAI_ERROR_LOGIC_ANN_INVALID_MATRIX_DEGREE, "ann invalid matrix degree");
  }
  
  nanai_object::~nanai_object() {
    
  }
  
  void error(size_t err) {
    nanai_object obj;
    obj.error(err);
  }
}