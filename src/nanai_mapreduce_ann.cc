#include <vector>
#include <string>
#include <nanmath_vector.h>
#include <nanmath_matrix.h>
#include <nanai_object.h>
#include <nanai_common.h>
#include <nanai_ann_nanncalc.h>
#include <nanai_ann_nnn.h>
#include <nanai_mapreduce_ann.h>

namespace nanai {
  
  nanai_mapreduce_ann::nanai_mapreduce_ann() {
    
  }
  
  nanai_mapreduce_ann::nanai_mapreduce_ann(const std::string &task,
                                           nanai_mapreduce_ann_input_t &input)
    : nanai_mapreduce(task, input) {
      _map_results.resize(input.first.first.size());
  }
  
  nanai_mapreduce_ann::~nanai_mapreduce_ann() {
  }
  
  void nanai_mapreduce_ann::read_config(const nanai_mapreduce_ann_config_t &config) {
    _desc = config.desc;
    _log_dir = config.log_dir;
    _wt = config.wt;
  }
  
  void nanai_mapreduce_ann::map() {
    std::vector<nanmath::nanmath_vector> inputs = _input.first.first;
    std::vector<nanmath::nanmath_vector> targets = _input.first.second;
    nanai_ann_nanncalc::ann_t ann = _input.second;
    
    if (inputs.size() != targets.size()) {
      error(NANAI_ERROR_LOGIC_INVALID_ARGUMENT);
    }
    
    /* 调用自定义的map操作 */
    if (_desc.fptr_map) {
      nanai_mapreduce_ann_config_t config;
      config.desc = _desc;
      config.log_dir = _log_dir;
      config.wt = _wt;
      if (_desc.fptr_map(reinterpret_cast<void*>(&_task),
                         reinterpret_cast<void*>(&config),
                         reinterpret_cast<void*>(&inputs),
                         reinterpret_cast<void*>(&targets),
                         reinterpret_cast<void*>(&ann),
                         reinterpret_cast<void*>(&_map_results)) == NANAI_ANN_DESC_RETURN) {
        return;
      }
    }
    
    size_t count = inputs.size();
    if (_wt == 0) {
    
      std::vector<std::shared_ptr<nanai_ann_nanncalc> > calcs;
      
      for (size_t i = 0; i < count; i++) {
        std::shared_ptr<nanai_ann_nanncalc>calc(make(_desc));
        calc->ann_training(_task, inputs[i], targets[i], ann, &(_map_results[i]));
        calcs.push_back(calc);
      }
    
      for (auto i : _map_results) {
        while (i.first.size() == 0) {
          usleep(100);
        }
      }
      
      /* 停止所有计算结点 */
      for (auto i : calcs) {
        i->ann_stop();
      }
    } else {
      std::shared_ptr<nanai_ann_nanncalc>calc(make(_desc));
      nanai_ann_nanncalc::result_t result;
      for (size_t i = 0; i < count; i++) {
        calc->ann_training(_task, inputs[i], targets[i], ann, &result);
        /* 直到训练完毕 */
        while (result.first.size() == 0) {
          usleep(100);
        }
        /* 更新神经网络 */
        ann = result.second;
        _map_results[i] = result;
      }
      calc->ann_stop();
    }
    
    return;
  }
  
  static nanmath::nanmath_matrix s_merge_delta_matrix(nanmath::nanmath_matrix &dmat1,
                                                      nanmath::nanmath_matrix &dmat2) {
    
    if ((dmat1.row_size() != dmat2.row_size()) ||
        (dmat1.col_size() != dmat2.col_size())) {
      error(NANAI_ERROR_LOGIC_INVALID_ARGUMENT);
    }
    
    nanmath::nanmath_matrix c(dmat1.row_size(), dmat1.col_size());
    for (size_t i = 0; i < dmat1.row_size(); i++) {
      for (size_t j = 0; j < dmat1.col_size(); j++) {
        /* 偏差越小在融合后的矩阵中所占值越大 */
        double delta_a = 1 - (dmat1[i][j] / dmat1[i][j] + dmat2[i][j]);
        double delta_b = 1 - (dmat2[i][j] / dmat1[i][j] + dmat2[i][j]);
        
        double c_n = delta_a * dmat1[i][j] + delta_b * dmat2[i][j];
        c.set(i, j, c_n);
      }
    }
    
    return c;
  }
  
  static nanmath::nanmath_matrix s_merge_matrix(nanmath::nanmath_matrix &mat1,
                                                nanmath::nanmath_matrix &mat2,
                                                nanmath::nanmath_matrix &dmat1,
                                                nanmath::nanmath_matrix &dmat2) {
    if ((mat1.row_size() != mat2.row_size()) ||
        (mat1.col_size() != mat2.col_size()) ||
        (mat1.row_size() != dmat1.row_size()) ||
        (mat1.col_size() != dmat2.col_size())) {
      error(NANAI_ERROR_LOGIC_INVALID_ARGUMENT);
    }
    
    nanmath::nanmath_matrix c(mat1.row_size(), mat1.col_size());
    for (size_t i = 0; i < mat1.row_size(); i++) {
      for (size_t j = 0; j < mat1.col_size(); j++) {
        double delta_a = 1 - (dmat1[i][j] / dmat1[i][j] + dmat2[i][j]);
        double delta_b = 1 - (dmat2[i][j] / dmat1[i][j] + dmat2[i][j]);
        
        double c_n = delta_a * mat1[i][j] + delta_b * mat2[i][j];
        c.set(i, j, c_n);
      }
    }
    
    return c;
  }
  
  static nanai_ann_nanncalc::ann_t s_merge_ann(nanai_ann_nanncalc::ann_t &a,
                                               nanai_ann_nanncalc::ann_t &b) {
    nanai_ann_nanncalc::ann_t c;
    if (a.weight_matrixes.size() != b.weight_matrixes.size()) {
      error(NANAI_ERROR_LOGIC_INVALID_ARGUMENT);
    }
    
    if ((a.delta_weight_matrixes.size() == 0) ||
        (b.delta_weight_matrixes.size() == 0)) {
      error(NANAI_ERROR_LOGIC_INVALID_ARGUMENT);
    }
    
    c = a;
    std::vector<nanmath::nanmath_matrix> wm;
    std::vector<nanmath::nanmath_matrix> dwm;
    
    /* 遍历每层的权值矩阵 */
    for (size_t i = 0; i < a.weight_matrixes.size(); i++) {
      wm.push_back(s_merge_matrix(a.weight_matrixes[i], b.weight_matrixes[i],
                                  a.delta_weight_matrixes[i], b.delta_weight_matrixes[i]));
      dwm.push_back(s_merge_delta_matrix(a.delta_weight_matrixes[i], b.delta_weight_matrixes[i]));
    }
    c.weight_matrixes = wm;
    c.delta_weight_matrixes = dwm;
    
    return c;
  }
  
  void s_merge_anns(std::vector<nanai_ann_nanncalc::ann_t> &anns,
                    nanai_ann_nanncalc::ann_t &ann) {
    
    if (anns.size() <= 1) {
      nanai::error(NANAI_ERROR_LOGIC_ANN_NUMBER_LESS_2);
    }
    
    /* 进行合并 */
    ann = anns[0];
    for (size_t i = 1; i < anns.size(); i++) {
      ann = s_merge_ann(ann, anns[i]);
    }
  }
  
  nanmath::nanmath_vector s_merge_outputs(std::vector<nanmath::nanmath_vector> &outputs,
                                          nanmath::nanmath_vector &output) {
    output = outputs[0];
    for (size_t i = 1; i < outputs.size(); i++) {
      output = output.add(outputs[i]);
    }
    /* 取平均值 */
    output = output.mul(1 / outputs.size());
    return output;
  }
  
  void nanai_mapreduce_ann::reduce() {
    /* 调用自定义的reduce操作 */
    if (_desc.fptr_reduce) {
      if (_desc.fptr_reduce(_wt,
                            reinterpret_cast<void*>(&_task),
                            reinterpret_cast<void*>(&_map_results),
                            reinterpret_cast<void*>(&_reduce_result)) == NANAI_ANN_DESC_RETURN) {
        return;
      }
    }
    
    if (_wt == 1) {
      _reduce_result = _map_results.back();
    } else {
      std::vector<nanai_ann_nanncalc::ann_t> anns;
      nanai_ann_nanncalc::ann_t ann;
      std::vector<nanmath::nanmath_vector> outputs;
      nanmath::nanmath_vector output;
      
      for (auto r : _map_results) {
        outputs.push_back(r.first);
        anns.push_back(r.second);
      }
      
      s_merge_outputs(outputs, output);
      s_merge_anns(anns, ann);
      
      _reduce_result.first = output;
      _reduce_result.second = ann;
    }
  }

  nanai_ann_nanncalc *nanai_mapreduce_ann::make(const nanai_ann_nanndesc &desc) {
    nanai_ann_nanncalc *calc = new nanai_ann_nanncalc(desc, _log_dir.c_str());
    if (calc == NULL) {
      error(NANAI_ERROR_RUNTIME_ALLOC_MEMORY);
    }
    
    return calc;
  }
  
}