#include <stdio.h>

#include "nanai_common.h"

namespace nanai {
  int nanai_support_nid(int adr) {
    int r = rand();
    return r ^ adr;
  }
}
