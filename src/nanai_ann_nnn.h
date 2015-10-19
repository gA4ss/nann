#ifndef nanai_ann_nnn_h
#define nanai_ann_nnn_h

#define NNN_MAGIC_CODE  "nnn\0"
#define NNN_EOF "END\0"

namespace nanai {

  typedef struct _nanai_ann_nnn {
    char magic[4];
    int version;
    char algname[32];
    int nnet;
    char *ann;
  } nanai_ann_nnn;
  
}

#endif /* nanai_ann_nnn_h */
