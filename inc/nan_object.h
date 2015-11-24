#ifndef nan_object_h
#define nan_object_h

#include <map>
#include <string>

#ifndef _NOEXCEPT
#define _NOEXCEPT   noexcept
#endif

namespace nanan {
  
/* error code */
#define NAN_ERROR_SUCCESS                               0
#define NAN_ERROR_FAILED                                0x80000000
  
#define NAN_ERROR_RUNTIME                               0x81000000
#define NAN_ERROR_RUNTIME_CREATE_THREAD                 0x81000001
#define NAN_ERROR_RUNTIME_INIT_MUTEX                    0x81000002
#define NAN_ERROR_RUNTIME_DESTROY_MUTEX                 0x81000003
#define NAN_ERROR_RUNTIME_LOCK_MUTEX                    0x81000004
#define NAN_ERROR_RUNTIME_UNLOCK_MUTEX                  0x81000005
#define NAN_ERROR_RUNTIME_JOIN_THREAD                   0x81000006
#define NAN_ERROR_RUNTIME_OPEN_FILE                     0x81000007
#define NAN_ERROR_RUNTIME_ALLOC_MEMORY                  0x8100000A
  
#define NAN_ERROR_LOGIC                                 0x82000000
#define NAN_ERROR_LOGIC_INVALID_ARGUMENT                0x82000001
  
  class nan_error {
  public:
    nan_error(size_t errcode,
              const std::string &errstr);
    virtual ~nan_error();
  public:
    virtual const char* what() const _NOEXCEPT;
    virtual const size_t errcode() const _NOEXCEPT;
    
  private:
    std::string _errstr;
    size_t _errcode;
  };
  
  class nan_object {
  public:
    nan_object();
    virtual ~nan_object();
    
  protected:
    virtual void on_error(size_t err);
    
  public:
    size_t get_last_error() const;
    void error(size_t err);
    void register_error(size_t errcode,
                        const std::string &errstr);
    
  private:
    size_t _last_error;
    std::map<size_t, std::string> _errors;
  };
  
  void error(size_t err);
  
}


#endif /* nan_object_h */
