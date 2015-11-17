#ifndef nanai_ann_nnn_h
#define nanai_ann_nnn_h

#include <vector>
#include <nanai_ann_nanncalc.h>

namespace nanai {
  
  void nanai_ann_nnn_read(const std::string &json_context,
                          nanai_ann_nanncalc::ann_t &ann);
  
  void nanai_ann_nnn_write(std::string &json_context,
                           const nanai_ann_nanncalc::ann_t &ann);
}

#endif /* nanai_ann_nnn_h */
