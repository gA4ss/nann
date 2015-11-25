#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include <nlang.h>

namespace nlang {
  
  nlang_object::nlang_object() : nanan::nan_object() {
    register_error(NLANG_ERROR_LOGIC_PARSE_ERROR, "parse nlang source error");
    register_error(NLANG_ERROR_LOGIC_SYMBOL_NOT_FOUND, "symbol not found");
    register_error(NLANG_ERROR_LOGIC_SYMBOL_TYPE_NOT_MATCHED, "symbol type not matched");
  }
  
  nlang_object::~nlang_object() {
    
  }
  
  /*******************************************************************************************************************/
  
  nlang_symbol::nlang_symbol() : nlang_object() {
    type = NLANG_TYPE_NULL;
    clear();
  }
  
  nlang_symbol::~nlang_symbol() {
    clear();
  }
  
  void nlang_symbol::clear() {
    type = NLANG_TYPE_NULL;
    value.bool_v = false;
    value.double_v = 0.0;
    value.string_v.clear();
    child = next = prev = nullptr;
  }
  
  static nlang_symbol_ptr s_new_symbol() {
    std::shared_ptr<nlang_symbol> res = std::shared_ptr<nlang_symbol>(new nlang_symbol());
    if (res == nullptr) {
      nanan::error(NAN_ERROR_RUNTIME_ALLOC_MEMORY);
    }
    return res;
  }

  /*******************************************************************************************************************/
  
  nlang::nlang() : nlang_object() {
    _aptr = nullptr;
  }
  
  nlang::nlang(const std::string &str) : nlang_object() {
    _aptr = nullptr;
    read(str);
  }
  
  nlang::~nlang() {
    
  }
  
  void nlang::read(const std::string &str) {
    read(str.c_str());
  }
  
  void nlang::read(const char *source) {
    parse(source);
  }

  void nlang::set_null(const std::string &name) {
    nlang_symbol_ptr tmp = s_new_symbol();
    if (_symbols.find(name) == _symbols.end()) {
      _symbols[name] = tmp;
    } else {
      _symbols[name]->clear();
    }
  }
  
  void nlang::set(const std::string &name,
                  const bool v) {
    nlang_symbol_ptr tmp = s_new_symbol();
    
    tmp->type = NLANG_TYPE_BOOL;
    tmp->value.bool_v = v;
    
    _symbols[name] = tmp;
  }
  
  void nlang::set(const std::string &name,
                  const double v) {
    nlang_symbol_ptr tmp = s_new_symbol();
    
    tmp->type = NLANG_TYPE_NUMBER;
    tmp->value.double_v = v;
    
    _symbols[name] = tmp;
  }
  
  void nlang::set(const std::string &name,
                  const std::string &v) {
    nlang_symbol_ptr tmp = s_new_symbol();
    
    tmp->type = NLANG_TYPE_STRING;
    tmp->value.string_v = v;
    
    _symbols[name] = tmp;
  }
  
  void nlang::set(const std::string &name,
                  nlang_symbol_ptr sym) {
    _symbols[name] = sym;
  }
  
  int nlang::get(const std::string &name,
                 bool &v) {
    if (_symbols.find(name) == _symbols.end()) {
      error(NLANG_ERROR_LOGIC_SYMBOL_NOT_FOUND);
    }
    
    if (_symbols[name]->type != NLANG_TYPE_BOOL) {
      error(NLANG_ERROR_LOGIC_SYMBOL_TYPE_NOT_MATCHED);
    }
    
    v = _symbols[name]->value.bool_v;
    
    return 0;
  }
  
  int nlang::get(const std::string &name,
                 double &v) {
    if (_symbols.find(name) == _symbols.end()) {
      error(NLANG_ERROR_LOGIC_SYMBOL_NOT_FOUND);
    }
    
    if (_symbols[name]->type != NLANG_TYPE_NUMBER) {
      error(NLANG_ERROR_LOGIC_SYMBOL_TYPE_NOT_MATCHED);
    }
    
    v = _symbols[name]->value.double_v;
    
    return 0;
  }
  
  int nlang::get(const std::string &name,
                 std::string &v) {
    if (_symbols.find(name) == _symbols.end()) {
      error(NLANG_ERROR_LOGIC_SYMBOL_NOT_FOUND);
    }
    
    if (_symbols[name]->type != NLANG_TYPE_STRING) {
      error(NLANG_ERROR_LOGIC_SYMBOL_TYPE_NOT_MATCHED);
    }
    
    v = _symbols[name]->value.string_v;
    
    return 0;
  }
  
  nlang_symbol_ptr nlang::get(const std::string &name) {
    if (_symbols.find(name) == _symbols.end()) return nullptr;
    return _symbols[name];
  }
  
  nlang_symbol_ptr nlang::root() const {
    return _root;
  }
  
  /*! 跳过空格与换行等不打印字符 */
  static const char *s_skip(const char *in) {while (in && *in && (unsigned char)*in<=32) in++; return in;}
  void nlang::parse(const char *source) {
    const char *end = nullptr;
    
    _root = s_new_symbol();
    if (_root == nullptr) {
      error(NAN_ERROR_RUNTIME_ALLOC_MEMORY);
    }
    
    _aptr = nullptr;
    
    end = parse_value(_root, s_skip(source));
    if (end == nullptr) {
      error(NLANG_ERROR_LOGIC_PARSE_ERROR);
    }
  }
  
  const char *nlang::parse_value(nlang_symbol_ptr sym, const char *source) {
    if (!source) return nullptr;
    if (!strncmp(source,"null",4))	{ sym->clear(); sym->type = NLANG_TYPE_NULL; return source+4; }
    if (!strncmp(source,"false",5))	{ sym->type = NLANG_TYPE_BOOL; sym->value.bool_v = false; return source+5; }
    if (!strncmp(source,"true",4))	{ sym->type = NLANG_TYPE_BOOL; sym->value.bool_v = true; return source+4; }
    if (*source == '\"' || *source == '\'')	{ return parse_string(sym, source); }
    if (*source == '-' || (*source >= '0' && *source <= '9'))	{ return parse_number(sym, source); }
    if (*source == '[')	{ return parse_array(sym, source); }
    if (*source == '{')	{ return parse_object(sym, source); }
    
    _aptr = const_cast<char *>(source);
    return nullptr;
  }
  
  /* 分析16进制字符 */
  static unsigned s_parse_hex4(const char *str) {
    unsigned h = 0;
    
    if (*str >= '0' && *str <= '9') h += (*str) - '0';
    else if (*str >= 'A' && *str <= 'F') h += 10 + (*str) - 'A';
    else if (*str >= 'a' && *str <= 'f') h += 10 + (*str) - 'a';
    else return 0;
    
    h = h << 4;
    str++;
    
    if (*str >= '0' && *str <= '9') h += (*str) - '0';
    else if (*str >= 'A' && *str <= 'F') h += 10 + (*str) - 'A';
    else if (*str >= 'a' && *str <= 'f') h += 10 + (*str) - 'a';
    else return 0;
    
    h = h << 4;
    str++;
    
    if (*str >= '0' && *str <= '9') h += (*str) - '0';
    else if (*str >= 'A' && *str <= 'F') h += 10 + (*str) - 'A';
    else if (*str >= 'a' && *str <= 'f') h += 10 + (*str) - 'a';
    else return 0;
    
    h = h << 4;
    str++;
    
    if (*str >= '0' && *str <= '9') h += (*str) - '0';
    else if (*str >= 'A' && *str <= 'F') h += 10 + (*str) - 'A';
    else if (*str >= 'a' && *str <= 'f') h += 10 + (*str) - 'a';
    else return 0;
    
    return h;
  }
  
  const char *nlang::parse_string(nlang_symbol_ptr sym, const char *str) {
    const char *ptr = str + 1;
    const unsigned char first_byte_mark[7] = { 0x00, 0x00, 0xC0, 0xE0, 0xF0, 0xF8, 0xFC };
    
    if ((*str != '\"') && (*str != '\'')) { _aptr = const_cast<char*>(str); return nullptr; }
    
    char head = '\"';
    if (*str == '\'') head = '\'';          /* 设置字符串标志 */
    
    /* 计算字符串长度 */
    size_t len = 0;
    while (*ptr != head && *ptr && ++len) if (*ptr++ == '\\') ptr++;	/* 跳过转移字符 */
    
    char *out = new char [len + 1];
    if (!out) {
      error(NAN_ERROR_RUNTIME_ALLOC_MEMORY);
    }
    
    ptr = str + 1;
    char *ptr2 = out;
    
    /* 读取字符串 */
    while (*ptr != head && *ptr) {
      /* 跳过转移字符 */
      if (*ptr != '\\') *ptr2++ = *ptr++;
      else {
        ptr++;
        switch (*ptr) {
          case 'b': *ptr2++ = '\b';	break;
          case 'f': *ptr2++ = '\f';	break;
          case 'n': *ptr2++ = '\n';	break;
          case 'r': *ptr2++ = '\r';	break;
          case 't': *ptr2++ = '\t';	break;
          case 'u': {/* 转换utf16到utf8. */
            unsigned uc = s_parse_hex4(ptr + 1);
            ptr += 4;	/* 获取unicode字符 */
            
            /* 检查是否合法	*/
            if ((uc >= 0xDC00 && uc <= 0xDFFF) || uc == 0) break;
            
            /* UTF16*/
            if (uc >= 0xD800 && uc <= 0xDBFF) {
              if (ptr[1] != '\\' || ptr[2] != 'u') break;	/* 丢失半个对 */
              unsigned uc2 = s_parse_hex4(ptr + 3);
              ptr += 6;
              if (uc2 < 0xDC00 || uc2 > 0xDFFF) break;	/* 无效的配对 */
              uc = 0x10000 + (((uc & 0x3FF) << 10) | (uc2 & 0x3FF));
            }
            
            len = 4;
            if (uc < 0x80) len = 1;
            else if (uc < 0x800) len=2;
            else if (uc < 0x10000) len=3;
            ptr2 += len;
            
            switch (len) {
              case 4: *--ptr2 =((uc | 0x80) & 0xBF); uc >>= 6;
              case 3: *--ptr2 =((uc | 0x80) & 0xBF); uc >>= 6;
              case 2: *--ptr2 =((uc | 0x80) & 0xBF); uc >>= 6;
              case 1: *--ptr2 =(uc | first_byte_mark[len]);
            }
            ptr2 += len;
          } break;
          default: *ptr2++ = *ptr; break;
        }
        ptr++;
      }
    }
    
    *ptr2 = '\0';
    if (*ptr == head) ptr++;
    sym->value.string_v = out;
    sym->type = NLANG_TYPE_STRING;
    if (out) delete [] out;
    
    return ptr;
  }
  
  const char *nlang::parse_number(nlang_symbol_ptr sym, const char *num) {
    double n = 0, sign = 1, scale = 0;
    int subscale = 0, signsubscale = 1;
    
    if (*num == '-') sign = -1, num++;  /* 是否有符号? */
    if (*num == '0') num++;             /* 是否是零 */
    if (*num >= '1' && *num <= '9') do n = (n * 10.0) + (*num++ -'0'); while (*num >= '0' && *num <= '9');	/* 数字? */
    /* 浮点部分 */
    if (*num == '.' && num[1] >= '0' && num[1] <= '9') {
      num++; do n = (n * 10.0) + (*num++ - '0'), scale--;
      while (*num >= '0' && *num <= '9');
    }
    /* 指数部分 */
    if (*num == 'e' || *num == 'E') {
      num++;
      if (*num == '+') num++;
      else if (*num == '-') signsubscale = -1, num++;     /* 带符号 */
      while (*num >= '0' && *num <= '9') subscale = (subscale * 10) + (*num++ - '0'); /* 数字? */
    }
    
    n = sign * n * pow(10.0, (scale + subscale * signsubscale));	/* number = +/- number.fraction * 10^+/- exponent */
    
    sym->value.double_v = n;
    sym->type = NLANG_TYPE_NUMBER;
    
    return num;
  }
  
  const char *nlang::parse_array(nlang_symbol_ptr sym, const char *value) {
    if (*value != '[')	{ _aptr = const_cast<char*>(value); return nullptr; }
    
    sym->type = NLANG_TYPE_ARRAY;
    value = s_skip(value + 1);
    if (*value == ']') return value + 1;
    
    nlang_symbol_ptr child;
    sym->child = child = s_new_symbol();
    if (!sym->child) return nullptr;
    value = s_skip(parse_value(child, s_skip(value)));
    if (!value) return nullptr;
    
    while (*value == ',') {
      nlang_symbol_ptr new_item;
      if (!(new_item = s_new_symbol())) return nullptr;
      child->next = new_item; new_item->prev = child; child = new_item;
      value = s_skip(parse_value(child, s_skip(value + 1)));
      if (!value) return nullptr;
    }
    
    if (*value == ']') return value + 1;
    _aptr = const_cast<char*>(value);
    return nullptr;
  }
  
  const char *nlang::parse_object(nlang_symbol_ptr sym, const char *value) {
    if (*value != '{') { _aptr = const_cast<char*>(value); return nullptr; }
    
    sym->type = NLANG_TYPE_OBJECT;
    value = s_skip(value + 1);
    if (*value == '}') return value + 1;
    
    nlang_symbol_ptr child;
    sym->child = child = s_new_symbol();
    if (!sym->child) return nullptr;
    value = s_skip(parse_string(child, s_skip(value)));
    if (!value) return nullptr;
    child->name = child->value.string_v; child->value.string_v.clear();
    if (*value != ':') { _aptr = const_cast<char*>(value); return nullptr; }
    value = s_skip(parse_value(child, s_skip(value + 1)));
    if (!value) return nullptr;
    
    /* 添加到符号 */
    set(child->name, child);
    
    while (*value == ',') {
      nlang_symbol_ptr new_item;
      if (!(new_item = s_new_symbol()))	return nullptr;
      child->next = new_item; new_item->prev = child; child = new_item;
      value = s_skip(parse_string(child, s_skip(value + 1)));
      if (!value) return nullptr;
      child->name = child->value.string_v; child->value.string_v.clear();
      if (*value != ':') { _aptr = const_cast<char*>(value); return nullptr; }
      value = s_skip(parse_value(child, s_skip(value + 1)));
      if (!value) return nullptr;
      
      /* 添加到符号 */
      set(child->name, child);
    }
    
    if (*value=='}') return value + 1;
    _aptr = const_cast<char*>(value); return nullptr;
  }

}