#include <string.h>
#include <stdexcept>
#include <nanai_ann_nnn.h>

namespace nanai {
  
  static double *s_to_mat(int ninput,
                          int nhidden,
                          int noutput,
                          std::vector<int> &nneural,
                          std::vector<nanmath::nanmath_matrix> &mat,
                          double *source) {
    for (int i = 0; i < nhidden+1; i++) {
      
      int nl1 = 0, nl2 = 0;
      
      if (i == 0) {
        nl1 = ninput;
      } else {
        nl1 = nneural[i];
      }
      
      if (i == nhidden) {
        nl2 = noutput;
      } else {
        nl2 = nneural[i+1];
      }
      
      for (int n = 0; n < nl1; n++) {
        for (int m = 0; m < nl2; m++) {
          mat[i][n][m] = *source++;
        }
      }
    }
    
    return source;
  }
  
  static int s_to_nnn(int ninput,
                      int nhidden,
                      int noutput,
                      const std::vector<int> &nneural,
                      const std::vector<nanmath::nanmath_matrix> &mat,
                      double *dest) {
    double *s = dest;
    for (int i = 0; i < nhidden; i++) {
      
      int nl1 = 0, nl2 = 0;
      
      if (i == 0) {
        nl1 = ninput;
      } else {
        nl1 = nneural[i];
      }
      
      if (i == nhidden) {
        nl2 = noutput;
      } else {
        nl2 = nneural[i+1];
      }
      
      for (int n = 0; n < nl1; n++) {
        for (int m = 0; m < nl2; m++) {
          *dest++ = mat[i][n][m];
        }
      }
    }
    
    return (int)(dest - s);
  }
  
  nanai_ann_nanncalc::ann_t nanai_ann_nnn_read(void *nnn) {
    nanai_ann_nanncalc::ann_t ann;
    nanai_ann_nnn *f = (nanai_ann_nnn*)nnn;
    double *mat = (double*)((unsigned char*)nnn + sizeof(nanai_ann_nnn));
    
    if (f->magic != NNN_MAGIC_CODE) {
      throw std::logic_error("not a invalid nnn format");
    }
    
    if (f->version != NNN_CURR_VERSION) {
      throw std::logic_error("curr version not be match");
    }
    
    ann.ninput = f->ninput;
    ann.nhidden = f->nhidden;
    ann.noutput = f->noutput;
    
    for (int i = 0; i < ann.nneural.size(); i++) {
      ann.nneural[i] = f->nneure[i];
    }
    
    int exist_weight_deltas = f->exist_weight_deltas;
    
    mat = s_to_mat(ann.ninput, ann.nhidden, ann.noutput, ann.nneural, ann.weight_matrix, mat);
    
    if (exist_weight_deltas) {
      mat = s_to_mat(ann.ninput, ann.nhidden, ann.noutput, ann.nneural, ann.delta_weight_matrix, mat);
    }
    
    if (*(int*)mat != NNN_EOF) {
      throw std::logic_error("nnn format is broken");
    }
    
    return ann;
  }
  
  int nanai_ann_nnn_write(const nanai_ann_nanncalc::ann_t &ann,
                          const std::string &alg,
                          const std::string &task,
                          void *nnn,
                          int len) {
    nanai_ann_nnn f;
    int ret = sizeof(nanai_ann_nnn), total = 0;
    unsigned char* dest = (unsigned char*)nnn;
    
    if (alg.empty() == false) {
      strcpy(f.algname, alg.c_str());
    }
    
    if (task.empty() == false) {
      strcpy(f.taskname, task.c_str());
    }
    
    f.magic = NNN_MAGIC_CODE;
    f.version = NNN_CURR_VERSION;
    f.ninput = ann.ninput;
    f.nhidden = ann.nhidden;
    f.noutput = ann.noutput;
    f.exist_weight_deltas = ann.delta_weight_matrix.empty() ? 0 : 1;
    
    total = sizeof(nanai_ann_nnn);
    for (int i = 0; i < ann.nhidden+1; i++) {
      
      int nl1 = 0, nl2 = 0;
      
      if (i == 0) {
        nl1 = ann.ninput;
      } else {
        nl1 = ann.nneural[i];
      }
      
      if (i == ann.nhidden) {
        nl2 = ann.noutput;
      } else {
        nl2 = ann.nneural[i+1];
      }
      
      total += (sizeof(double) * (nl1 * nl2));
    }
    total += sizeof(int);
    
    if (len < total) {
      throw std::invalid_argument("nnn buffer size too small");
    }
    
    memcpy(dest, &f, sizeof(nanai_ann_nnn));
    dest += sizeof(nanai_ann_nnn);
    
    ret += nanai::s_to_nnn(ann.ninput, ann.nhidden, ann.noutput, ann.nneural, ann.weight_matrix, (double*)dest);
    dest += ret;
    
    if (f.exist_weight_deltas) {
      ret += nanai::s_to_nnn(ann.ninput, ann.nhidden, ann.noutput, ann.nneural, ann.delta_weight_matrix, (double*)dest);
      dest += ret;
    }
    
    *(int *)dest = NNN_EOF;
    
    return ret+sizeof(int);
  }
}