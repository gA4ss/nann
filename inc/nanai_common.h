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
  void nanai_support_input_json(const std::string &json_context,
                                std::vector<nanmath::nanmath_vector> &inputs,
                                nanmath::nanmath_vector *target);
  std::string nanai_support_just_filename(const std::string &path);
}

#endif /* nanai_common_h */
