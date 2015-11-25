#ifndef nanai_mapreduce_ann_h
#define nanai_mapreduce_ann_h

#include <string>
#include <vector>
#include <utility>

#include <nanmath_vector.h>
#include <nanmath_matrix.h>
#include <nanai_object.h>
#include <nanai_ann_nanndesc.h>
#include <nanai_ann_nanncalc.h>
#include <nanai_mapreduce.h>
#include <nanai_ann_nanncalc.h>

namespace nanai {
  typedef std::pair<std::pair<std::vector<nanmath::nanmath_vector>, std::vector<nanmath::nanmath_vector> >,
  nanai_ann_nanncalc::ann_t>nanai_mapreduce_ann_input_t;
  
  typedef nanai_mapreduce<nanai_mapreduce_ann_input_t, nanai_ann_nanncalc::result_t, nanai_ann_nanncalc::result_t> mapreduce_ann_t;
  
  class nanai_mapreduce_ann : public mapreduce_ann_t {
  public:
    nanai_mapreduce_ann();
    nanai_mapreduce_ann(const std::string &task,                  /*!< 任务名 */
                        nanai_mapreduce_ann_input_t &input        /*!< 输入 */
                        );
    
    virtual ~nanai_mapreduce_ann();
    
  public:
    typedef struct _nanai_mapreduce_ann_config_t {
      nanai_ann_nanndesc desc;
      std::string log_dir;
      int wt;
    } nanai_mapreduce_ann_config_t;
    /*! 读取配置 */
    virtual void read_config(const nanai_mapreduce_ann_config_t &config);
      
    /*! map操作 */
    virtual void map() override;
      
    /*! reduce操作 */
    virtual void reduce() override;
      
  protected:
    nanai_ann_nanndesc _desc;                                                     /*!< 算法描述结果 */
    std::string _log_dir;                                                         /*!< 日志记录目录 */
    int _wt;                                                                      /*!< 工作模式 */
  };
  
}

#endif /* nanai_mapreduce_ann_h */
