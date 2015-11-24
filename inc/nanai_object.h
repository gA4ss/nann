#ifndef nanai_object_h
#define nanai_object_h

#include <nan_object.h>

namespace nanai {

#define NANAI_ERROR_LOGIC                               0x82100000
#define NANAI_ERROR_LOGIC_INVALID_CONFIG                0x82100001
#define NANAI_ERROR_LOGIC_ALG_NOT_FOUND                 0x82100002
#define NANAI_ERROR_LOGIC_TASK_NOT_MATCHED              0x82100003
#define NANAI_ERROR_LOGIC_TASK_ALREADY_EXIST            0x82100004
#define NANAI_ERROR_LOGIC_HOME_DIR_NOT_CONFIG           0x82100005
#define NANAI_ERROR_LOGIC_DESC_FUNCTION_NOT_FOUND       0x82100006
  
  /*! 人工神经网络的错误 */
#define NANAI_ERROR_LOGIC_ANN_MERGE_NUMBER_LESS_2       0x82110001
#define NANAI_ERROR_LOGIC_ANN_INVALID_VECTOR_DEGREE     0x82110002
#define NANAI_ERROR_LOGIC_ANN_INVALID_MATRIX_DEGREE     0x82110003
  
  class nanai_object : public nanan::nan_object {
  public:
    nanai_object();
    virtual ~nanai_object();
  };
  
  void error(size_t err);
}

#endif /* nanai_object_h */
