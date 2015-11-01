#ifndef nanai_object_h
#define nanai_object_h

#include <stdexcept>

namespace nanai {
  
  /* error code */
#define NANAI_ERROR_SUCCESS                             0
#define NANAI_ERROR_UNKNOW_ERROR                        0x80000000

#define NANAI_ERROR_LOGIC                               0x80100000
#define NANAI_ERROR_LOGIC_INVALID_ARGUMENT              0x80100001
#define NANAI_ERROR_LOGIC_INVALID_CONFIG                0x80100002
#define NANAI_ERROR_LOGIC_ALG_NOT_FOUND                 0x80110001
#define NANAI_ERROR_LOGIC_TASK_NOT_MATCHED              0x80110002
  
#define NANAI_ERROR_RUNTIME                             0x80200000
#define NANAI_ERROR_RUNTIME_CREATE_THREAD               0x80200001
#define NANAI_ERROR_RUNTIME_INIT_MUTEX                  0x80200002
#define NANAI_ERROR_RUNTIME_DESTROY_MUTEX               0x80200003
#define NANAI_ERROR_RUNTIME_LOCK_MUTEX                  0x80200004
#define NANAI_ERROR_RUNTIME_UNLOCK_MUTEX                0x80200005
#define NANAI_ERROR_RUNTIME_JOIN_THREAD                 0x80200006
#define NANAI_ERROR_RUNTIME_OPEN_FILE                   0x80200007
#define NANAI_ERROR_RUNTIME_ALLOC_MEMORY                0x8020000A
  
  
  class nanai_error_unknow_error : public std::exception {
  public:
    explicit nanai_error_unknow_error();
    virtual const char* what() const _NOEXCEPT;
  public:
    int _errcode;
  };
  
  class nanai_error_logic_invalid_argument : public std::invalid_argument {
  public:
    explicit nanai_error_logic_invalid_argument();
  public:
    int _errcode;
  };
  
  class nanai_error_logic_invalid_config : public std::logic_error {
  public:
    explicit nanai_error_logic_invalid_config();
  public:
    int _errcode;
  };
  
  class nanai_error_logic_alg_not_found : public std::logic_error {
  public:
    explicit nanai_error_logic_alg_not_found();
  public:
    int _errcode;
  };
  
  class nanai_error_logic_task_not_matched : public std::logic_error {
  public:
    explicit nanai_error_logic_task_not_matched();
  public:
    int _errcode;
  };
  
  class nanai_error_runtime_create_thread : public std::runtime_error {
  public:
    explicit nanai_error_runtime_create_thread();
  public:
    int _errcode;
  };
  
  class nanai_error_runtime_init_mutex : public std::runtime_error {
  public:
    explicit nanai_error_runtime_init_mutex();
  public:
    int _errcode;
  };
  
  class nanai_error_runtime_destroy_mutex : public std::runtime_error {
  public:
    explicit nanai_error_runtime_destroy_mutex();
  public:
    int _errcode;
  };
  
  class nanai_error_runtime_lock_mutex : public std::runtime_error {
  public:
    explicit nanai_error_runtime_lock_mutex();
  public:
    int _errcode;
  };
  
  class nanai_error_runtime_unlock_mutex : public std::runtime_error {
  public:
    explicit nanai_error_runtime_unlock_mutex();
  public:
    int _errcode;
  };
  
  class nanai_error_runtime_join_thread : public std::runtime_error {
  public:
    explicit nanai_error_runtime_join_thread();
  public:
    int _errcode;
  };
  
  class nanai_error_runtime_open_file : public std::runtime_error {
  public:
    explicit nanai_error_runtime_open_file();
  public:
    int _errcode;
  };
  
  class nanai_error_runtime_alloc_memory : public std::runtime_error {
  public:
    explicit nanai_error_runtime_alloc_memory();
  public:
    int _errcode;
  };
  
  class nanai_object {
  public:
    nanai_object();
    nanai_object(const nanai_object &t);
    virtual ~nanai_object();
    
  protected:
    virtual void on_error(int err);
    
  public:
    int get_last_error() const;
    void error(int err);
    
  protected:
    int _last_error;
  };
  
  void error(int err);
  
}


#endif /* nanai_object_h */
