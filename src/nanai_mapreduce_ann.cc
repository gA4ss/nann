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
    
    return;
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
  }

  nanai_ann_nanncalc *nanai_mapreduce_ann::make(const nanai_ann_nanndesc &desc) {
    nanai_ann_nanncalc *calc = new nanai_ann_nanncalc(desc, _log_dir.c_str());
    if (calc == NULL) {
      error(NANAI_ERROR_RUNTIME_ALLOC_MEMORY);
    }
    
    return calc;
  }
  
}