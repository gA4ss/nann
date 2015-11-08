#ifndef nanai_ann_nnn_h
#define nanai_ann_nnn_h

#include <vector>
#include <nanai_ann_nanncalc.h>

namespace nanai {
  
  void nanai_ann_nnn_read(const std::string &json_context,
                          std::string &alg,
                          nanai_ann_nanncalc::ann_t &ann,
                          nanmath::nanmath_vector *target=nullptr);
  
  void nanai_ann_nnn_write(std::string &json_context,
                           const std::string &alg,
                           const nanai_ann_nanncalc::ann_t &ann,
                           nanmath::nanmath_vector *target=nullptr);
}

#endif /* nanai_ann_nnn_h */
