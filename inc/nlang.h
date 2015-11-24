#ifndef nlang_h
#define nlang_h

#include <nan_object.h>
#include <string>
#include <vector>
#include <map>

namespace nlang {
  
#define NLANG_TYPE_NULL             0
#define NLANG_TYPE_BOOL             1
#define NLANG_TYPE_NUMBER           2
#define NLANG_TYPE_STRING           3
#define NLANG_TYPE_ARRAY            4
#define NLANG_TYPE_OBJECT           5
  
  class nlang_object : public nanan::nan_object {
  public:
    nlang_object();
    virtual ~nlang_object();
  };
  
  class nlang_var : public nlang_object {
  public:
    nlang_var();
    virtual ~nlang_var();
    void clear();
    
  public:
    int type;
    bool bool_v;
    int int_v;
    double double_v;
    std::string string_v;
    
    typedef struct {
      std::shared_ptr<nlang_var> child;
      std::shared_ptr<nlang_var> next;
    } nlang_var_object;
    
    std::vector<nlang_var> array;
  };
  
  typedef void* nlang_source_object;
  
  class nlang : public nlang_object {
  public:
    nlang();
    virtual ~nlang();
    
  public:
    
    void read(nlang_source_object source);
    
    void set_null(const std::string &name);
    
    void set(const std::string &name,
             const bool v);
    
    void set(const std::string &name,
             const double v);
    
    void set(const std::string &name,
             const std::string &v);
    
    int get(const std::string &name,
            bool &v);
    
    int get(const std::string &name,
            int &v);
    
    int get(const std::string &name,
            double &v);
    
    int get(const std::string &name,
            std::string &v);
    
  protected:
    int parse(std::string &source);
    int parse_object(std::string &source);
    
  private:
    std::map<std::string, nlang_var> _symbols;
  };

}

#endif /* nlang_h */
