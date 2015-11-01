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
    //nanai_ann_nannmgr(int max=1024, const char *ip="127.0.0.1", short port=8393);
    virtual ~nanai_ann_nannmgr();
    
  public:
    /* 输出结果，并调整误差 */
    virtual nanai_ann_nanncalc *train(nanmath::nanmath_vector &input,
                                      nanmath::nanmath_vector &target,
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
                                                  nanmath::nanmath_vector &target,
                                                  nanai_ann_nanncalc *dcalc=nullptr,
                                                  const char *task=nullptr,
                                                  nanai_ann_nanncalc::ann_t *ann=nullptr,
                                                  const char *alg=nullptr);
    
    virtual nanai_ann_nanncalc *nnn_read(const std::string &nnn);
    virtual void nnn_write(const std::string &nnn,
                           nanai_ann_nanncalc *calc);
    virtual void waits();
    virtual void set_max(int max);
    virtual int version() const;
    
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
    
    /*
     * 重载基类虚函数
     */
  protected:
    void on_error(int err);
    
    /*
     * 环境变量
     */
  protected:
    std::string _home_dir;
    std::string _lib_dir;
    std::string _etc_dir;
    std::string _log_dir;
    
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
