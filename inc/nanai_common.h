#ifndef nanai_common_h
#define nanai_common_h

#define _REENTRANT

#include <nanai_memory.h>

namespace nanai {
  
  #define nanai_support_abs(x)          (((x) > 0.0) ? (x) : (-(x)))
  
  int nanai_support_nid(int adr);
  int nanai_support_tid();
}

#endif /* nanai_common_h */
