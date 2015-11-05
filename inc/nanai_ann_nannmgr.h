#ifndef nanai_ann_nannmgr_h
#define nanai_ann_nannmgr_h

#include <string>
#include <vector>
#include <pthread.h>

#include <nanai_ann_nanncalc.h>

namespace nanai {
  class nanai_ann_nannmgr : public nanai_object {
  public:
    nanai_ann_nannmgr(int max=1024, int now_start=0);
    nanai_ann_nannmgr(std::string alg,
                      nanai_ann_nanncalc::ann_t &ann,
                      nanmath::nanmath_vector *target,
                      int max,
                      int now_start);
    //nanai_ann_nannmgr(int max=1024, const char *ip="127.0.0.1", short port=8393);
    virtual ~nanai_ann_nannmgr();
    
  public:
    /* 输出结果，并调整误差 */
    virtual nanai_ann_nanncalc *training(nanmath::nanmath_vector &input,
                                         nanmath::nanmath_vector *target,
                                         nanai_ann_nanncalc *dcalc=nullptr,
                                         const char *task=nullptr,
                                         nanai_ann_nanncalc::ann_t *ann=nullptr,
                                         const char *alg=nullptr);
    
    /* 输出结果，不调整误差 */
    virtual nanai_ann_nanncalc *training_notarget(nanmath::nanmath_vector &input,
                                                  nanai_ann_nanncalc *dcalc=nullptr,
                                                  const char *task=nullptr,
                                                  nanai_ann_nanncalc::ann_t *ann=nullptr,
                                                  const char *alg=nullptr);
    /* 不输出结果，调整误差 */
    virtual nanai_ann_nanncalc *training_nooutput(nanmath::nanmath_vector &input,
                                                  nanmath::nanmath_vector *target,
                                                  nanai_ann_nanncalc *dcalc=nullptr,
                                                  const char *task=nullptr,
                                                  nanai_ann_nanncalc::ann_t *ann=nullptr,
                                                  const char *alg=nullptr);
    
    virtual nanai_ann_nanncalc *nnn_read(const std::string &nnn);
    virtual void nnn_write(const std::string &nnn,
                           nanai_ann_nanncalc *calc);
    virtual void waits();
    virtual void set_max(int max);
    
    /* 合并任务神经网络 */
    virtual void merge_ann_by_task(std::string task);
  protected:
    virtual nanai_ann_nanncalc::ann_t merge_ann(nanai_ann_nanncalc::ann_t &a,
                                                nanai_ann_nanncalc::ann_t &b);
    
    virtual nanmath::nanmath_matrix merge_matrix(nanmath::nanmath_matrix &mat1,
                                                 nanmath::nanmath_matrix &mat2,
                                                 nanmath::nanmath_matrix &dmat1,
                                                 nanmath::nanmath_matrix &dmat2);
    virtual nanmath::nanmath_matrix merge_delta_matrix(nanmath::nanmath_matrix &dmat1,
                                                       nanmath::nanmath_matrix &dmat2);
    
    
  public:
    /*
     * 针对任务的产查询
     */
    virtual int dead_task();
    virtual int exist_task(std::string task);
    
  public:
    /*
     * 静态函数
     */
    static const char *version();
    
  protected:
    virtual void configure();
    virtual void get_env();
    virtual void get_algs(std::string &path);
    virtual void get_def_algs();
    virtual bool add_alg(nanai_ann_nanndesc &desc);
    virtual nanai_ann_nanndesc *find_alg(std::string alg);
    virtual void lock();
    virtual void unlock();
    virtual nanai_ann_nanncalc *generate(nanai_ann_nanndesc &desc,
                                         nanai_ann_nanncalc::ann_t *ann=NULL,
                                         const char *task=NULL);
    virtual nanai_ann_nanncalc *make(nanai_ann_nanndesc &desc);
    
  protected:
    /*
     * 重载基类虚函数
     */
    void on_error(int err);
    
  protected:
    /*
     * 环境变量
     */
    std::string _home_dir;
    std::string _lib_dir;
    std::string _etc_dir;
    std::string _log_dir;
    
  protected:
    /* 
     * 管理器唯一性识别 
     */
    std::string _alg;
    nanai_ann_nanncalc::ann_t _ann;
    nanmath::nanmath_vector _target;
    
  protected:
    int _max_calc;
    int _curr_calc;
    std::vector<nanai_ann_nanncalc*> _calcs;
    std::vector<nanai_ann_nanndesc> _descs;
    std::vector<std::pair<void*, fptr_ann_alg_setup> > _algs;
    
    pthread_mutex_t _lock;
  };
}

#endif /* nanai_ann_nannmgr_h */
