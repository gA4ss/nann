#ifndef nanai_ann_nnn_h
#define nanai_ann_nnn_h

#include <vector>
#include <nanai_ann_nanncalc.h>

namespace nanai {
  
#define NNN_CURR_VERSION 0x100
#define NNN_MAGIC_CODE  0x08080808
#define NNN_EOF 0x08080808
  
  typedef struct _nanai_ann_nnn {
    int magic;
    int version;
    char algname[32];
    char taskname[32];
    int ninput;
    int nhidden;
    int noutput;
    int nneure[MAX_HIDDEN_NUMBER];
    int exist_weight_deltas;
  } nanai_ann_nnn;
  
  nanai_ann_nanncalc::ann_t nanai_ann_nnn_read(void *nnn);
  
  int nanai_ann_nnn_write(const nanai_ann_nanncalc::ann_t &ann,
                          const std::string &alg,
                          const std::string &task,
                          void *nnn,
                          int len);
}

#endif /* nanai_ann_nnn_h */
