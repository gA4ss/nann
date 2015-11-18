#include <nanai_object.h>
#include <errno.h>

namespace nanai {

  nanai_error_unknow_error::nanai_error_unknow_error() {
    _errcode = NANAI_ERROR_UNKNOW_ERROR;
  }
  const char* nanai_error_unknow_error::what() const _NOEXCEPT { return "unknow error"; }
  
  nanai_error_logic_invalid_argument::nanai_error_logic_invalid_argument() : std::invalid_argument("invalid argument") {
    _errcode = NANAI_ERROR_LOGIC_INVALID_ARGUMENT;
  }
  
  nanai_error_logic_invalid_config::nanai_error_logic_invalid_config() : std::logic_error("invalid config") {
    _errcode = NANAI_ERROR_LOGIC_INVALID_CONFIG;
  }
  
  nanai_error_logic_alg_not_found::nanai_error_logic_alg_not_found() : std::logic_error("algorithm not found") {
    _errcode = NANAI_ERROR_LOGIC_ALG_NOT_FOUND;
  }
  
  nanai_error_logic_task_not_matched::nanai_error_logic_task_not_matched() : std::logic_error("task not matched") {
    _errcode = NANAI_ERROR_LOGIC_TASK_NOT_MATCHED;
  }
  
  nanai_error_logic_task_already_exist::nanai_error_logic_task_already_exist() : std::logic_error("task already exist") {
    _errcode = NANAI_ERROR_LOGIC_TASK_ALREADY_EXIST;
  }
  
  nanai_error_logic_home_dir_not_config::nanai_error_logic_home_dir_not_config() : std::logic_error("home dir not config") {
    _errcode = NANAI_ERROR_LOGIC_HOME_DIR_NOT_CONFIG;
  }
  
  nanai_error_logic_desc_function_not_found::nanai_error_logic_desc_function_not_found()
  : std::logic_error("desc function not found") {
    _errcode = NANAI_ERROR_LOGIC_DESC_FUNCTION_NOT_FOUND;
  }
  
  nanai_error_logic_ann_merge_number_less_2::nanai_error_logic_ann_merge_number_less_2()
  : std::logic_error("number of ann less 2") {
    _errcode = NANAI_ERROR_LOGIC_ANN_MERGE_NUMBER_LESS_2;
  }
  
  nanai_error_logic_ann_invalid_vector_degree::nanai_error_logic_ann_invalid_vector_degree()
  : std::logic_error("invalid vector degree") {
    _errcode = NANAI_ERROR_LOGIC_ANN_INVALID_VECTOR_DEGREE;
  }
  
  nanai_error_logic_ann_invalid_matrix_degree::nanai_error_logic_ann_invalid_matrix_degree()
  : std::logic_error("invalid matrix degree") {
    _errcode = NANAI_ERROR_LOGIC_ANN_INVALID_MATRIX_DEGREE;
  }
  
  nanai_error_runtime_create_thread::nanai_error_runtime_create_thread() : std::runtime_error("create thread failed") {
    _errcode = NANAI_ERROR_RUNTIME_CREATE_THREAD;
  }
  
  nanai_error_runtime_init_mutex::nanai_error_runtime_init_mutex() : std::runtime_error("init mutex failed") {
    _errcode = NANAI_ERROR_RUNTIME_INIT_MUTEX;
  }
  
  nanai_error_runtime_destroy_mutex::nanai_error_runtime_destroy_mutex() : std::runtime_error("destroy mutex failed") {
    _errcode = NANAI_ERROR_RUNTIME_DESTROY_MUTEX;
  }
  
  nanai_error_runtime_lock_mutex::nanai_error_runtime_lock_mutex() : std::runtime_error("lock mutex failed") {
    _errcode = NANAI_ERROR_RUNTIME_LOCK_MUTEX;
  }
  
  nanai_error_runtime_unlock_mutex::nanai_error_runtime_unlock_mutex() : std::runtime_error("unlock mutex failed") {
    _errcode = NANAI_ERROR_RUNTIME_UNLOCK_MUTEX;
  }
  
  nanai_error_runtime_join_thread::nanai_error_runtime_join_thread() : std::runtime_error("join thread failed") {
    _errcode = NANAI_ERROR_RUNTIME_JOIN_THREAD;
  }
  
  nanai_error_runtime_open_file::nanai_error_runtime_open_file() : std::runtime_error("open file failed") {
    _errcode = NANAI_ERROR_RUNTIME_OPEN_FILE;
  }
  
  nanai_error_runtime_alloc_memory::nanai_error_runtime_alloc_memory() : std::runtime_error("alloc memory failed") {
    _errcode = NANAI_ERROR_RUNTIME_ALLOC_MEMORY;
  }
  
  /* -------------------------------------------------------------------------------- */
  
  
  static nanai_error_logic_invalid_argument logic_invalid_argument;
  static nanai_error_logic_invalid_config logic_invalid_config;
  static nanai_error_logic_alg_not_found logic_alg_not_found;
  static nanai_error_logic_task_not_matched logic_task_not_matched;
  static nanai_error_logic_task_already_exist logic_task_already_exist;
  static nanai_error_logic_home_dir_not_config logic_home_dir_not_config;
  static nanai_error_logic_desc_function_not_found logic_desc_function_not_found;
  static nanai_error_logic_ann_merge_number_less_2 logic_ann_merge_number_less_2;
  static nanai_error_logic_ann_invalid_vector_degree logic_ann_invalid_vector_degree;
  static nanai_error_logic_ann_invalid_matrix_degree logic_ann_invalid_matrix_degree;
  static nanai_error_runtime_create_thread runtime_create_thread;
  static nanai_error_runtime_init_mutex runtime_init_mutex;
  static nanai_error_runtime_destroy_mutex runtime_destroy_mutex;
  static nanai_error_runtime_lock_mutex runtime_lock_mutex;
  static nanai_error_runtime_unlock_mutex runtime_unlock_mutex;
  static nanai_error_runtime_join_thread runtime_join_thread;
  static nanai_error_runtime_open_file runtime_open_file;
  static nanai_error_runtime_alloc_memory runtime_alloc_memory;
  static struct {
    int errcode;
    std::exception &except;
  } s_nanai_excepts [] = {
    {static_cast<int>(NANAI_ERROR_LOGIC_INVALID_ARGUMENT),  logic_invalid_argument},
    {static_cast<int>(NANAI_ERROR_LOGIC_INVALID_CONFIG), logic_invalid_config},
    {static_cast<int>(NANAI_ERROR_LOGIC_ALG_NOT_FOUND), logic_alg_not_found},
    {static_cast<int>(NANAI_ERROR_LOGIC_TASK_NOT_MATCHED), logic_task_not_matched},
    {static_cast<int>(NANAI_ERROR_LOGIC_TASK_ALREADY_EXIST), logic_task_already_exist},
    {static_cast<int>(NANAI_ERROR_LOGIC_HOME_DIR_NOT_CONFIG), logic_home_dir_not_config},
    {static_cast<int>(NANAI_ERROR_LOGIC_DESC_FUNCTION_NOT_FOUND), logic_desc_function_not_found},
    {static_cast<int>(NANAI_ERROR_LOGIC_ANN_MERGE_NUMBER_LESS_2), logic_ann_merge_number_less_2},
    {static_cast<int>(NANAI_ERROR_LOGIC_ANN_INVALID_VECTOR_DEGREE), logic_ann_invalid_vector_degree},
    {static_cast<int>(NANAI_ERROR_LOGIC_ANN_INVALID_MATRIX_DEGREE), logic_ann_invalid_matrix_degree},
    {static_cast<int>(NANAI_ERROR_RUNTIME_CREATE_THREAD), runtime_create_thread},
    {static_cast<int>(NANAI_ERROR_RUNTIME_INIT_MUTEX), runtime_init_mutex},
    {static_cast<int>(NANAI_ERROR_RUNTIME_DESTROY_MUTEX), runtime_destroy_mutex},
    {static_cast<int>(NANAI_ERROR_RUNTIME_LOCK_MUTEX), runtime_lock_mutex},
    {static_cast<int>(NANAI_ERROR_RUNTIME_UNLOCK_MUTEX), runtime_unlock_mutex},
    {static_cast<int>(NANAI_ERROR_RUNTIME_JOIN_THREAD), runtime_join_thread},
    {static_cast<int>(NANAI_ERROR_RUNTIME_OPEN_FILE), runtime_open_file},
    {static_cast<int>(NANAI_ERROR_RUNTIME_ALLOC_MEMORY), runtime_alloc_memory},
  };
  
  nanai_object::nanai_object() {
    _last_error = 0;
  }
  
  nanai_object::nanai_object(const nanai_object &t) {
    _last_error = t._last_error;
  }
  
  nanai_object::~nanai_object() {
  }
  
  void nanai_object::on_error(int err) {
    
  }
  
  int nanai_object::get_last_error() const {
    return _last_error;
  }
  
  void nanai_object::error(int err) {
    _last_error = err;
    
    on_error(err);
    
    int i = 0;
    while (s_nanai_excepts[i++].errcode != 0) {
      
      if (s_nanai_excepts[i++].errcode == err) {
        throw s_nanai_excepts[i++].except;
      }
    }
    
    // 其余情况抛出未知异常
    throw nanai_error_unknow_error();
  }
  
  void error(int err) {
    int i = 0;
    while (s_nanai_excepts[i++].errcode != 0) {
      
      if (s_nanai_excepts[i++].errcode == err) {
        throw s_nanai_excepts[i++].except;
      }
    }
    throw nanai_error_unknow_error();
  }
}
