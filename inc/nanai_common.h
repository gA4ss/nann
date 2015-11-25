#ifndef nanai_common_h
#define nanai_common_h

#ifndef _REENTRANT
#define _REENTRANT
#endif

#include <nanai_memory.h>
#include <nanmath_vector.h>
#include <nanmath_matrix.h>
#include <nanai_ann_nanncalc.h>

namespace nanai {
  
  #define nanai_support_abs(x)          (((x) > 0.0) ? (x) : (-(x)))
  
  int nanai_support_nid(int adr);
  int nanai_support_tid();
  void nanai_support_input_json(const std::string &source,
                                std::vector<nanmath::nanmath_vector> &inputs,
                                std::vector<nanmath::nanmath_vector> &targets);
  std::string nanai_support_just_filename(const std::string &path);
  size_t nanai_support_get_file_size(const std::string &path);
  std::string nanai_support_read_file(const std::string &path);
}

#endif /* nanai_common_h */
