#include <cstring>
#include <string>
#include <sstream>
#include <stdexcept>
#include <iomanip>
#include <nanai_ann_nnn.h>
#include <nlang.h>

namespace nanai {
  
  static void parse_create_json_read_matrix(nlang::nlang_symbol_ptr sym,
                                            nanmath::nanmath_matrix &matrix) {
    int nrow = 0, ncol = 0, nprev = 0;
    nlang::nlang_symbol_ptr row = sym, col = nullptr;
    std::vector<double> row_v;
    matrix.clear();
    
    int idx = 0;
    std::pair<int, std::vector<double> > idx_row;
    std::vector<std::pair<int, std::vector<double> > > rows;
    
    while (row) {
      col = row->child;
      if (col == nullptr) {
        error(NANAI_ERROR_LOGIC_INVALID_CONFIG);
      }
      idx = atoi(row->name.c_str());
      ncol = 0;
      row_v.clear();
      while (col) {
        row_v.push_back(col->value.double_v);
        ncol++;
        col = col->next;
      }
      
      if (nprev) {
        /* 每列必须一样的数量 */
        if (nprev != ncol) {
          error(NANAI_ERROR_LOGIC_INVALID_CONFIG);
        }
      }
      nprev = ncol;
      
      /* 压入一行 */
      idx_row.first = idx;
      idx_row.second = row_v;
      rows.push_back(idx_row);
      
      nrow++;
      row = row->next;
    }
    
    /* 排序 */
    std::sort(rows.begin(), rows.end(), [](std::pair<int, std::vector<double> > &a,
                                           std::pair<int, std::vector<double> > &b) {
      return a.first < b.first;
    });
    
    matrix.clear();
    for (auto r : rows) {
      matrix.push_row(r.second);
    }
  }
  
  static void parse_create_json_matrixes(nlang::nlang_symbol_ptr sym,
                                         size_t &ninput,
                                         size_t &nhidden,
                                         size_t &noutput,
                                         std::vector<nanmath::nanmath_matrix> &matrixes) {
    ninput = 0;
    nhidden = 0;
    noutput = 0;
    
    nlang::nlang_symbol_ptr curr = sym, jmat = nullptr;
    int idx = 0;
    nanmath::nanmath_matrix mat;
    std::pair<int, nanmath::nanmath_matrix> idx_mat;
    std::vector<std::pair<int, nanmath::nanmath_matrix> >mats;
    
    while (curr) {
      jmat = curr->child;
      if (jmat == nullptr) { error(NANAI_ERROR_LOGIC_INVALID_CONFIG); }

      idx = atoi(curr->name.c_str());
      parse_create_json_read_matrix(jmat, mat);
      idx_mat.first = idx;
      idx_mat.second = mat;
      mats.push_back(idx_mat);
      curr = curr->next;
    }
    
    /* 排序 */
    std::sort(mats.begin(), mats.end(), [](std::pair<int, nanmath::nanmath_matrix> &a,
                                           std::pair<int, nanmath::nanmath_matrix> &b) {
      return a.first < b.first;
    });
    for (auto m : mats) {
      matrixes.push_back(m.second);
    }
    ninput = matrixes[0].row_size();
    nhidden = matrixes.size() - 1;
    noutput = matrixes.back().col_size();
  }
  
  static void parse_create_json(nlang::nlang_symbol_ptr sym,
                                nanai_ann_nanncalc::ann_t &ann) {
    if (sym == nullptr) error(NAN_ERROR_LOGIC_INVALID_ARGUMENT);
    nlang::nlang_symbol_ptr child = sym->child;
    if (child == nullptr) error(NANAI_ERROR_LOGIC_INVALID_CONFIG);
    
    /* 遍历根结点 */
    bool handle_alg = false, handle_ann = false;
    while (child) {
      if (child->name == "ann") {
        nlang::nlang_symbol_ptr next = child->child;
        size_t wm_ninput = 0, dwm_ninput = 0,
        wm_nhidden = 0, dwm_nhidden = 0,
        wm_noutput = 0, dwm_noutput = 0;
        
        std::vector<int> wm_nneure, dwm_nneure;
        while (next) {
          if (next->name == "weight matrixes") {
            parse_create_json_matrixes(next->child,
                                       wm_ninput,
                                       wm_nhidden,
                                       wm_noutput,
                                       ann.weight_matrixes);
          } else if (next->name == "delta weight matrixes") {
            parse_create_json_matrixes(next->child,
                                       dwm_ninput,
                                       dwm_nhidden,
                                       dwm_noutput,
                                       ann.delta_weight_matrixes);
          } else if (next->name == "alg") {
            ann.alg = next->value.string_v;
            handle_alg = true;
          }
          
          next = next->next;
        }/* end while */
        handle_ann = true;
        
        if (ann.weight_matrixes.empty()) error(NANAI_ERROR_LOGIC_INVALID_CONFIG);
        
        /* 偏差权值矩阵不为空 */
        if (ann.delta_weight_matrixes.empty() == false) {
          if ((wm_ninput != dwm_ninput) ||
              (wm_nhidden != dwm_nhidden) ||
              (wm_noutput != dwm_noutput) ||
              (ann.weight_matrixes.size() != ann.delta_weight_matrixes.size())) {
            error(NANAI_ERROR_LOGIC_INVALID_CONFIG);
          }
        }/* end if */
        
        ann.ninput = wm_ninput;
        ann.nhidden = wm_nhidden;
        ann.noutput = wm_noutput;
      }
      
      child = child->next;
    }/* end while */
    
    /* target是可选参数，所以不检查 */
    if ((handle_ann == false) || (handle_alg == false)) {
      error(NANAI_ERROR_LOGIC_INVALID_CONFIG);
    }
  }
  
  void nanai_ann_nnn_read(const std::string &source,
                          nanai_ann_nanncalc::ann_t &ann) {
    if (source.empty()) {
      error(NANAI_ERROR_LOGIC_INVALID_CONFIG);
    }
    
    nlang::nlang n(source);
    nlang::nlang_symbol_ptr obj = n.root();
    
    parse_create_json(obj, ann);
    
    /* 主动填充神经网络的nnearul */
    ann.fill_nneural();
  }
  
  void nanai_ann_nnn_write(std::string &source,
                           const nanai_ann_nanncalc::ann_t &ann,
                           int precision) {
    
    if (ann.empty()) {
      error(NAN_ERROR_LOGIC_INVALID_ARGUMENT);
    }
    
    std::string alg = ann.alg;
    if (alg.empty()) {
      error(NAN_ERROR_LOGIC_INVALID_ARGUMENT);
    }
    
    source.clear();
    std::ostringstream oss;
    oss << "{" << std::endl;
    oss << "\t" << "\"ann\": {" << std::endl;
    oss << "\t\t" << "\"alg\": " << "\"" << alg << "\"," << std::endl;
    oss << "\t\t" << "\"weight matrixes\": {" << std::endl;
    for (size_t i = 0; i < ann.weight_matrixes.size(); i++) {
      oss << "\t\t\t" << "\"" << i << "\": {" << std::endl;
      for (size_t n = 0; n < ann.weight_matrixes[i].row_size(); n++) {
        oss << "\t\t\t\t" << "\"" << n << "\": [";
        
        /* 输出一行 */
        for (size_t m = 0; m < ann.weight_matrixes[i].col_size(); m++) {
          oss << std::setiosflags(std::ios::fixed) << std::setiosflags(std::ios::left)
              << std::setprecision(precision) << std::setw(4) << ann.weight_matrixes[i][n][m];
          if (m < (ann.weight_matrixes[i].col_size()-1)) oss << ", ";
        }
        
        if (n < (ann.weight_matrixes[i].row_size()-1)) oss << "]," << std::endl;
        else oss << "]" << std::endl;
      }
      
      if (i < (ann.weight_matrixes.size()-1)) oss << "\t\t\t}," << std::endl;
      else oss << "\t\t\t}" << std::endl;
    }
    oss << "\t\t}," << std::endl;
    
    oss << "\t\t" << "\"delta weight matrixes\": {" << std::endl;
    for (size_t i = 0; i < ann.delta_weight_matrixes.size(); i++) {
      oss << "\t\t\t" << "\"" << i << "\": {" << std::endl;
      for (size_t n = 0; n < ann.delta_weight_matrixes[i].row_size(); n++) {
        oss << "\t\t\t\t" << "\"" << n << "\": [";
        
        /* 输出一行 */
        for (size_t m = 0; m < ann.delta_weight_matrixes[i].col_size(); m++) {
          oss << std::setiosflags(std::ios::fixed) << std::setiosflags(std::ios::left)
          << std::setprecision(precision) << std::setw(4) << ann.delta_weight_matrixes[i][n][m];
          if (m < (ann.delta_weight_matrixes[i].col_size()-1)) oss << ", ";
        }
        
        if (n < (ann.delta_weight_matrixes[i].row_size()-1)) oss << "]," << std::endl;
        else oss << "]" << std::endl;
      }
      
      if (i < (ann.delta_weight_matrixes.size()-1)) oss << "\t\t\t}," << std::endl;
      else oss << "\t\t\t}" << std::endl;
    }
    
    oss << "\t\t}" << std::endl;
    oss << "\t}" << std::endl;
    oss << "}" << std::endl;
    source = oss.str();
  }
}