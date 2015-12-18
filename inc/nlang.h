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
#define NLANG_TYPE_PROCEDURE        6
#define NLANG_TYPE_PROCEDURE_PLUGIN 61
#define NLANG_TYPE_PROCEDURE_LISP   62
#define NLANG_TYPE_MAP              7
  
  
#define NLANG_ERROR_LOGIC                           0x82300000
#define NLANG_ERROR_LOGIC_PARSE_ERROR               0x82300001
#define NLANG_ERROR_LOGIC_SYMBOL_NOT_FOUND          0x82300002
#define NLANG_ERROR_LOGIC_SYMBOL_TYPE_NOT_MATCHED   0x82300003

  class nlang_object;
  
#if 0
  class nlang_procedure : public nanan::nan_object {
  public:
    nlang_procedure();
    virtual ~nlang_procedure();
    
  public:
    int type;
    std::string name;
    size_t argc;
    std::shared_ptr<nlang_object> parents;
  };
  
  class nlang_graph : public nanan::nan_object {
  public:
    nlang_graph();
    virtual ~nlang_graph();
    
  public:
    std::string name;
    std::string context;
    std::shared_ptr<nlang_object> parents;
    std::vector<std::shared_ptr<nlang_object> > map;
  };
#endif
  
  class nlang_object : public nanan::nan_object {
  public:
    nlang_object();
    virtual ~nlang_object();
  };
  
  class nlang_symbol : public nlang_object {
  public:
    nlang_symbol();
    virtual ~nlang_symbol();
    void clear();
    
  public:
    int type;
    std::string name;
    
    typedef struct _nlang_symbol_value_t {
      bool bool_v;
      double double_v;
      std::string string_v;
#if 0
      nlang_procedure procedure_v;
      nlang_graph graph_v;
#endif
    } nlang_symbol_value_t;
    
    nlang_symbol_value_t value;
    
    std::shared_ptr<nlang_symbol> child;
    std::shared_ptr<nlang_symbol> prev;
    std::shared_ptr<nlang_symbol> next;
  };
  
  typedef std::shared_ptr<nlang_symbol> nlang_symbol_ptr;
  
  class nlang : public nlang_object {
  public:
    nlang();
    nlang(const std::string &str);
    virtual ~nlang();
    
  public:
    void read(const std::string &str);
    void read(const char *source);
    
  public:
    void set_null(const std::string &name);
    
    void set(const std::string &name,
             const bool v);
    
    void set(const std::string &name,
             const double v);
    
    void set(const std::string &name,
             const std::string &v);
    
    void set(const std::string &name,
             nlang_symbol_ptr sym);
    
    int get(const std::string &name,
            bool &v);
    
    int get(const std::string &name,
            double &v);
    
    int get(const std::string &name,
            std::string &v);
    
    nlang_symbol_ptr get(const std::string &name);
    nlang_symbol_ptr root() const;
    
  protected:
    void parse(const char *source);
    const char *parse_value(nlang_symbol_ptr sym, const char *source);
    const char *parse_string(nlang_symbol_ptr sym, const char *str);
    const char *parse_number(nlang_symbol_ptr sym, const char *num);
    const char *parse_array(nlang_symbol_ptr sym, const char *value);
    const char *parse_object(nlang_symbol_ptr sym, const char *value);
    const char *parse_procedure(nlang_symbol_ptr sym, const char *procedure);
    const char *parse_graph(nlang_symbol_ptr sym, const char *graph);
    
  private:
    std::map<std::string, nlang_symbol_ptr> _symbols;
    nlang_symbol_ptr _root;
    char *_aptr;
  };
  
}

#endif /* nlang_h */
