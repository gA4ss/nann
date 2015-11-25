#include <nan_object.h>
#include <errno.h>

namespace nanan {

  nan_error::nan_error(size_t errcode,
                       const std::string &errstr) {
    _errcode = errcode;
    _errstr = errstr;
  }
  
  nan_error::~nan_error() {
    
  }

  const char* nan_error::what() const _NOEXCEPT {
    return _errstr.c_str();
  }
  
  const size_t nan_error::errcode() const _NOEXCEPT {
    return _errcode;
  }
  
  nan_object::nan_object() {
    _last_error = 0;
    
    register_error(NAN_ERROR_RUNTIME_CREATE_THREAD, "create thread error");
    register_error(NAN_ERROR_RUNTIME_INIT_THREAD_ATTR, "init thread attribute error");
    register_error(NAN_ERROR_RUNTIME_SETDETACHSTATE, "setdetachstate error");
    register_error(NAN_ERROR_RUNTIME_DESTROY_ATTRIBUTE, "destroy thread attribute error");
    register_error(NAN_ERROR_RUNTIME_INIT_MUTEX, "init mutex error");
    register_error(NAN_ERROR_RUNTIME_DESTROY_MUTEX, "destroy mutex error");
    register_error(NAN_ERROR_RUNTIME_LOCK_MUTEX, "lock mutex error");
    register_error(NAN_ERROR_RUNTIME_UNLOCK_MUTEX, "unlock mutex error");
    register_error(NAN_ERROR_RUNTIME_JOIN_THREAD, "join thread error");
    register_error(NAN_ERROR_RUNTIME_OPEN_FILE, "open file error");
    register_error(NAN_ERROR_RUNTIME_ALLOC_MEMORY, "alloc memory error");
    register_error(NAN_ERROR_LOGIC_INVALID_ARGUMENT, "invaild argument");
  }
  
  nan_object::~nan_object() {
  }
  
  void nan_object::on_error(size_t err) {
    
  }
  
  size_t nan_object::get_last_error() const {
    return _last_error;
  }
  
  void nan_object::error(size_t err) {
    if (err == NAN_ERROR_SUCCESS) {
      return;
    }
    
    _last_error = err;
    on_error(err);
    
    if (_errors.find(err) != _errors.end()) {
      throw nan_error(err, _errors[err]);
    }
    
    throw nan_error(NAN_ERROR_FAILED, "unknow error");
  }
  
  void nan_object::register_error(size_t errcode,
                                  const std::string &errstr) {
    if (errcode == NAN_ERROR_SUCCESS) {
      return;
    }
    
    if (_errors.find(errcode) == _errors.end()) {
      _errors[errcode] = errstr;
    }
  }
  
  std::string nan_object::errstr(size_t errcode) {
    if (errcode == NAN_ERROR_SUCCESS) {
      return "success";
    }
    
    if (_errors.find(errcode) != _errors.end()) {
      return _errors[errcode];
    }
    
    return "none";
    
  }
  
  void error(size_t err) {
    nan_object obj;
    obj.error(err);
  }
}
