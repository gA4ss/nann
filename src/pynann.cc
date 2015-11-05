#include <cstdio>
#include <iostream>
#include <map>
#include <pthread.h>
#include <stdexcept>
#include <iomanip>
#include <Python/Python.h>
#include <nanai_ann_nannmgr.h>

#include "cJSON.h"

using namespace nanai;

#define PYNANN_ERROR_SUCCESS                        0

/* 错误信息，不可忽略 */
#define PYNANN_ERROR_INTERNAL                           0x88000000
#define PYNANN_ERROR_PARSE_JSON                         0x88000001
#define PYNANN_ERROR_INVALID_ARGUMENT                   0x88000100
#define PYNANN_ERROR_INVALID_ARGUMENT_TASK_EXIST        0x88000101
#define PYNANN_ERROR_INVALID_ARGUMENT_TASK_NOT_EXIST    0x88000102
#define PYNANN_ERROR_ALLOC_MEMORY                       0x88000200

/* 警告错误，可忽略 */
#define PYNANN_WARNING_INTERNAL                         0x87000000
#define PYNANN_WARNING_INVALID_ARGUMENT                 0x87000100
#define PYNANN_WARNING_INVALID_ARGUMENT_TASK_EXIST      0x87000101
#define PYNANN_WARNING_INVALID_ARGUMENT_TASK_NOT_EXIST  0x87000102

typedef struct _nannmgr_node {
  std::string task;
  std::string alg;
  nanai_ann_nanncalc::ann_t ann;
  nanai_ann_nannmgr *mgr;
} nannmgr_node_t;

pthread_mutex_t g_lock;
static std::map<std::string, nannmgr_node_t> g_mgrlist;
static const char *g_str_this_version_not_implemented = "this version  ot implemented!!!";
static bool g_throw_exp = true;

static PyObject *do_except(int code, const char *str=nullptr);
PyObject *do_except(int code, const char *str) {
  static const struct {
    int code;
    PyObject* pye;
    const char *msg;
  } msgs[] = {
    { static_cast<int>(PYNANN_ERROR_INTERNAL), PyExc_Exception, "internal error" },
    { static_cast<int>(PYNANN_ERROR_PARSE_JSON), PyExc_SyntaxError, "parse json error" },
    { static_cast<int>(PYNANN_ERROR_INVALID_ARGUMENT), PyExc_StandardError, "invalid argument" },
    { static_cast<int>(PYNANN_ERROR_INVALID_ARGUMENT_TASK_EXIST), PyExc_StandardError, "task already exist" },
    { static_cast<int>(PYNANN_ERROR_INVALID_ARGUMENT_TASK_NOT_EXIST), PyExc_StandardError, "task not exist" },
    { static_cast<int>(PYNANN_ERROR_ALLOC_MEMORY), PyExc_MemoryError, "alloc memory failed" },
    { static_cast<int>(PYNANN_WARNING_INTERNAL), PyExc_Warning, "internal warning" },
    { static_cast<int>(PYNANN_WARNING_INVALID_ARGUMENT), PyExc_RuntimeWarning, "invalid argument warning level" },
    { static_cast<int>(PYNANN_WARNING_INVALID_ARGUMENT_TASK_EXIST), PyExc_RuntimeWarning, "task already exist warning level" },
    { static_cast<int>(PYNANN_WARNING_INVALID_ARGUMENT_TASK_NOT_EXIST), PyExc_RuntimeWarning, "task not exist warning level" }
  };
  
  char *outstr = nullptr;
  for (int x = 0; x < (int)(sizeof(msgs) / sizeof(msgs[0])); x++) {
    if (msgs[x].code == code) {
      
      if (str) outstr = const_cast<char*>(str);
      else outstr = const_cast<char*>(msgs[x].msg);
      
      if (g_throw_exp) PyErr_SetString(msgs[x].pye, outstr);
      return Py_BuildValue("i", code);
    }
  }
  return Py_BuildValue("i", PYNANN_ERROR_SUCCESS);
}

static cJSON *parse_conf(char *text) {
  cJSON *json=cJSON_Parse(text);
  if (!json) {
    //printf("Error before: [%s]\n",cJSON_GetErrorPtr());
    return nullptr;
  }
  
  //if (json) {
  //  char *out=cJSON_Print(json);
  //  printf("%s\n",out);
  //  free(out);
  //}
  
  return json;
}

static void lock() {
  if (pthread_mutex_lock(&g_lock) != 0) {
    PyErr_SetString(PyExc_RuntimeWarning, "pthread_mutex_lock failed");
  }
}

static void unlock() {
  if (pthread_mutex_unlock(&g_lock) != 0) {
    PyErr_SetString(PyExc_RuntimeWarning, "pthread_mutex_lock failed");
  }
}

static PyObject *wrap_test(PyObject *self, PyObject *args) {
  std::cout << "hello nanan" << std::endl;
  Py_RETURN_NONE;
}

static int parse_create_json_read_matrix(cJSON *json,
                                         nanmath::nanmath_matrix &matrix) {
  int nrow = 0, ncol = 0, nprev = 0;
  cJSON *row = json, *col = nullptr;
  std::vector<double> row_v;
  matrix.clear();
  
  while (row) {
    col = row->child;
    if (col == nullptr) {
      return PYNANN_ERROR_PARSE_JSON;
    }
    
    ncol = 0;
    row_v.clear();
    while (col) {
      if (col->valuedouble == 0.0) row_v.push_back((double)col->valueint);
      else row_v.push_back(col->valuedouble);
      ncol++;
      col = col->next;
    }
    
    if (nprev) {
      /* 每列必须一样的数量 */
      if (nprev != ncol) {
        matrix.clear();
        return PYNANN_ERROR_PARSE_JSON;
      }
    }
    nprev = ncol;
    
    /* 压入一行 */
    matrix.push_row(row_v);
    
    nrow++;
    row = row->next;
  }
  
  return 0;
}

static int parse_create_json_matrixes(cJSON *json,
                                      size_t &ninput,
                                      size_t &nhidden,
                                      size_t &noutput,
                                      std::vector<nanmath::nanmath_matrix> &matrixes) {
  ninput = 0;
  nhidden = 0;
  noutput = 0;
  
  cJSON *curr = json, *jmat = nullptr;
  int i = 0, ret = 0;
  nanmath::nanmath_matrix mat;
  
  while (curr) {
    jmat = curr->child;
    if (jmat == nullptr) { ret = PYNANN_ERROR_PARSE_JSON; goto _error; }
    
    if (i == 0) {
      /* 输入层到隐藏层 */
      ret = parse_create_json_read_matrix(jmat, mat);
      if (ret) goto _error;
      
      ninput = mat.row_size();
      
    } else if (curr->next == nullptr) {
      /* 隐藏层到输出层 */
      
      ret = parse_create_json_read_matrix(jmat, mat);
      if (ret) goto _error;
      
      noutput = mat.col_size();
      
    } else {
      /* 隐藏层之间 */
      ret = parse_create_json_read_matrix(jmat, mat);
      if (ret) goto _error;
    }
    
    matrixes.push_back(mat);
    i++;
    curr = curr->next;
  }
  nhidden = i - 1;
  
  return 0;
_error:
  matrixes.clear();
  return ret;
}

static int parse_create_json(cJSON *json,
                             std::string &alg,
                             nanai_ann_nanncalc::ann_t &ann,
                             nanmath::nanmath_vector &target) {
  int ret = PYNANN_ERROR_SUCCESS;
  if (json == nullptr) {
    return PYNANN_ERROR_INTERNAL;
  }
  cJSON *json_child = json->child;
  if (json_child == nullptr) {
    return PYNANN_ERROR_PARSE_JSON;
  }
  
  /* 遍历根结点 */
  bool handle_alg = false, handle_ann = false;
  while (json_child) {
    if (strcmp(json_child->string, "alg") == 0) {
      alg = json_child->valuestring;
      handle_alg = true;
    } else if (strcmp(json_child->string, "ann") == 0) {
      cJSON *json_next = json_child->child;
      size_t wm_ninput = 0, dwm_ninput = 0,
             wm_nhidden = 0, dwm_nhidden = 0,
             wm_noutput = 0, dwm_noutput = 0;
      
      std::vector<int> wm_nneure, dwm_nneure;
      while (json_next) {
        if (strcmp(json_next->string, "weight matrixes") == 0) {
          if ((ret = parse_create_json_matrixes(json_next->child,
                                                wm_ninput,
                                                wm_nhidden,
                                                wm_noutput,
                                                ann.weight_matrixes)) != PYNANN_ERROR_SUCCESS) {
            goto _error;
          }
        } else if (strcmp(json_next->string, "delta weight matrixes") == 0) {
          if ((ret = parse_create_json_matrixes(json_next->child,
                                                dwm_ninput,
                                                dwm_nhidden,
                                                dwm_noutput,
                                                ann.delta_weight_matrixes)) != PYNANN_ERROR_SUCCESS) {
            goto _error;
          }
        }
        json_next = json_next->next;
      }/* end while */
      handle_ann = true;
      
      if (ann.weight_matrixes.empty()) {
        ret = PYNANN_ERROR_PARSE_JSON;
        goto _error;
      }
      
      /* 偏差权值矩阵不为空 */
      if (ann.delta_weight_matrixes.empty() == false) {
        if ((wm_ninput != dwm_ninput) ||
            (wm_nhidden != dwm_nhidden) ||
            (wm_noutput != dwm_noutput) ||
            (ann.weight_matrixes.size() != ann.delta_weight_matrixes.size())) {
          ret = PYNANN_ERROR_PARSE_JSON;
          goto _error;
        }
      }/* end if */
      
      ann.ninput = wm_ninput;
      ann.nhidden = wm_nhidden;
      ann.noutput = wm_noutput;
      
    } else if (strcmp(json_child->string, "target") == 0) {
      cJSON *json_next = json_child->child;
      if (json_next == nullptr) { ret = PYNANN_ERROR_PARSE_JSON; goto _error; }
      target.clear();
      while (json_next) {
        if (json_next->valuedouble) target.push(json_next->valuedouble);
        else target.push((double)json_next->valueint);
        json_next = json_next->next;
      }
    }
    
    json_child = json_child->next;
  }/* end while */
  
  if ((handle_ann == false) || (handle_alg == false)) {
    ret = PYNANN_ERROR_PARSE_JSON;
    goto _error;
  }
  
  if (json) cJSON_Delete(json);
  return 0;
_error:
  alg.clear();
  ann.clear();
  target.clear();
  return ret;
}

static PyObject *wrap_create(PyObject *self, PyObject *args) {
  int max_calc = 0, now_calc = 0;
  char *task = nullptr, *json = nullptr;
  std::string alg;
  nanai_ann_nanncalc::ann_t ann;
  nanmath::nanmath_vector target;
  int ret = PYNANN_ERROR_SUCCESS;
  
  if (!PyArg_ParseTuple(args, "ssi|i", &task, &json, &max_calc, &now_calc)) {
    return do_except(PYNANN_ERROR_INVALID_ARGUMENT);
  }
  
  /* 解析json */
  cJSON *cjson = parse_conf(json);
  if (cjson == nullptr) {
    return do_except(PYNANN_ERROR_PARSE_JSON);
  }
  
  if ((ret = parse_create_json(cjson, alg, ann, target))) {
    return do_except(ret);
  }
  
  if (max_calc < now_calc) {
    return do_except(PYNANN_ERROR_INVALID_ARGUMENT);
  }
  
  nanai_ann_nannmgr *nannmgr = nullptr;
  try {
    nannmgr = new nanai_ann_nannmgr(max_calc, now_calc);
  } catch (std::exception e) {
    return do_except(PYNANN_ERROR_INTERNAL, e.what());
  }
  
  if (nannmgr == nullptr) {
    return do_except(PYNANN_ERROR_ALLOC_MEMORY);
  }
  
  nannmgr_node_t node;
  node.task = task;
  node.alg = alg;
  node.ann = ann;
  node.mgr = nannmgr;
  
  lock();
  if (g_mgrlist.find(task) != g_mgrlist.end()) {
    /* 任务已经存在 ...  */
    unlock();
    return do_except(PYNANN_ERROR_INVALID_ARGUMENT_TASK_EXIST);
  }
  g_mgrlist[task] = node;
  unlock();
  
  return Py_BuildValue("i", g_mgrlist.size()-1);
}

static PyObject *wrap_destroy(PyObject *self, PyObject *args) {
  char *task_arg = nullptr;
  std::string task;
  
  if (!PyArg_ParseTuple(args, "s", &task)) {
    return do_except(PYNANN_ERROR_INVALID_ARGUMENT);
  }
  task = task_arg;
  
  
  lock();
  if (g_mgrlist.empty()) {
    return do_except(PYNANN_WARNING_INVALID_ARGUMENT);
  }
  
  if (g_mgrlist.find(task) == g_mgrlist.end()) {
    /* 任务不存在 ...  */
    unlock();
    return do_except(PYNANN_WARNING_INVALID_ARGUMENT_TASK_NOT_EXIST);
  }
  
  if (g_mgrlist[task].mgr) {
    try {
      delete g_mgrlist[task].mgr; g_mgrlist[task].mgr = nullptr;
      g_mgrlist.erase(task);
    } catch (std::exception e) {
      return do_except(PYNANN_ERROR_INTERNAL, e.what());
    }
  }
  unlock();
  
  return Py_BuildValue("i", PYNANN_ERROR_SUCCESS);
}

static PyObject *wrap_version(PyObject *self, PyObject *args) {
  return Py_BuildValue("s", nanai_ann_nannmgr::version());
}

static PyObject *wrap_exptype(PyObject *self, PyObject *args) {
  int code = 0;
  
  if (!PyArg_ParseTuple(args, "i", &code)) {
    return do_except(PYNANN_ERROR_INVALID_ARGUMENT);
  }
  
  g_throw_exp = (code > 0);
  
  Py_RETURN_NONE;
}


static int parse_samples(cJSON *json,
                         std::vector<nanmath::nanmath_vector> &samples,
                         nanmath::nanmath_vector *target) {
  nanmath::nanmath_vector input;
  int ncol = 0, nprev = 0;
  int ret = PYNANN_ERROR_SUCCESS;
  cJSON *curr = json->child;
  if (curr == nullptr) {
    ret = PYNANN_ERROR_PARSE_JSON;
    goto _error;
  }
  
  while (curr) {
    if (strcmp(curr->string, "samples") == 0) {
      cJSON *vec = curr->child;
      if (vec == nullptr) {
        ret = PYNANN_ERROR_PARSE_JSON;
        goto _error;
      }
      
      while (vec) {
        cJSON *val = vec->child;
        
        ncol = 0;
        input.clear();
        while (val) {
          if (val->valuedouble) input.push(val->valuedouble);
          else input.push(val->valueint);
          ncol++;
          val = val->next;
        }
        
        /* 每个输入的向量个数必须一样 */
        if (nprev) {
          if (nprev != ncol) {
            ret = PYNANN_ERROR_PARSE_JSON;
            goto _error;
          }
        }
        nprev = ncol;
        samples.push_back(input);
        vec = vec->next;
      }
      
    } else if (strcmp(curr->string, "target") == 0) {
      
      if (target == nullptr) {
        continue;
      }
      
      cJSON *vec = curr->child;
      if (vec == nullptr) {
        ret = PYNANN_ERROR_PARSE_JSON;
        goto _error;
      }
      
      /* 读入一条向量 */
      while (vec) {
        if (vec->valuedouble) target->push(vec->valuedouble);
        else target->push(vec->valueint);
        vec = vec->next;
      }
      
    }
    
    curr = curr->next;
  }
  
  return PYNANN_ERROR_SUCCESS;
_error:
  samples.clear();
  if (target) target->clear();
  return ret;
}

static PyObject *wrap_training(PyObject *self, PyObject *args) {
  char *task_arg = nullptr, *json = nullptr;
  std::string task;
  nanmath::nanmath_vector target;
  std::vector<nanmath::nanmath_vector> samples;
  int ret = PYNANN_ERROR_SUCCESS;
  
  if (!PyArg_ParseTuple(args, "ss", &task_arg, &json)) {
    PyErr_SetString(PyExc_TypeError, "type error");
    return nullptr;
  }
  
  task = task_arg;
  if (g_mgrlist.find(task) == g_mgrlist.end()) {
    /* 没有找到任务，则返回警告 */
    return do_except(PYNANN_WARNING_INVALID_ARGUMENT_TASK_NOT_EXIST);
  }
  
  /* 解析json文件 */
  cJSON *cjson = parse_conf(json);
  if (cjson == nullptr) {
    return do_except(PYNANN_ERROR_PARSE_JSON);
  }
  
  target.clear();
  if ((ret = parse_samples(cjson, samples, &target))) {
    return do_except(ret);
  }
  
  try {
    for (auto i : samples) {
      if (target.size() == 0) g_mgrlist[task].mgr->training(i, nullptr);
      else g_mgrlist[task].mgr->training(i, &target);
    }
  } catch (std::exception e) {
    return do_except(PYNANN_ERROR_INTERNAL, e.what());
  }
  
  return Py_BuildValue("i", PYNANN_ERROR_SUCCESS);
}

static PyObject *wrap_training_notarget(PyObject *self, PyObject *args) {
  char *task_arg = nullptr, *json = nullptr;
  std::string task;
  std::vector<nanmath::nanmath_vector> samples;
  int ret = PYNANN_ERROR_SUCCESS;
  
  if (!PyArg_ParseTuple(args, "ss", &task_arg, &json)) {
    return do_except(PYNANN_ERROR_SUCCESS);
  }
  
  task = task_arg;
  if (g_mgrlist.find(task) == g_mgrlist.end()) {
    /* 没有找到任务，则返回警告 */
    return do_except(PYNANN_WARNING_INVALID_ARGUMENT_TASK_NOT_EXIST);
  }
  
  /* 解析json文件 */
  cJSON *cjson = parse_conf(json);
  if (cjson == nullptr) {
    return do_except(PYNANN_ERROR_PARSE_JSON);
  }
  
  if ((ret = parse_samples(cjson, samples, nullptr))) {
    return do_except(ret);
  }
  
  try {
    for (auto i : samples) {
      g_mgrlist[task].mgr->training_notarget(i);
    }
  } catch (std::exception e) {
    return do_except(PYNANN_ERROR_INTERNAL, e.what());
  }
  
  return Py_BuildValue("i", PYNANN_ERROR_SUCCESS);
}

static PyObject *wrap_training_nooutput(PyObject *self, PyObject *args) {
  char *task_arg = nullptr, *json = nullptr;
  std::string task;
  nanmath::nanmath_vector target;
  std::vector<nanmath::nanmath_vector> samples;
  
  if (!PyArg_ParseTuple(args, "ss", &task_arg, &json)) {
    return do_except(PYNANN_WARNING_INVALID_ARGUMENT);
  }
  
  task = task_arg;
  if (g_mgrlist.find(task) == g_mgrlist.end()) {
    /* 没有找到任务，则返回警告 */
    return do_except(PYNANN_WARNING_INVALID_ARGUMENT_TASK_NOT_EXIST);
  }
  
  /* 解析json文件 */
  cJSON *cjson = parse_conf(json);
  if (cjson == nullptr) {
    return do_except(PYNANN_ERROR_PARSE_JSON);
  }
  
  try {
    for (auto i : samples) {
      if (target.size() == 0) g_mgrlist[task].mgr->training_nooutput(i, &target);
      else g_mgrlist[task].mgr->training_nooutput(i, nullptr);
    }
  } catch (std::exception e) {
    return do_except(PYNANN_ERROR_INTERNAL, e.what());
  }
  
  return Py_BuildValue("i", PYNANN_ERROR_SUCCESS);
}

static PyObject *wrap_training2(PyObject *self, PyObject *args) {
  std::cout << g_str_this_version_not_implemented << std::endl;
  Py_RETURN_NONE;
}

static PyObject *wrap_training2_notarget(PyObject *self, PyObject *args) {
  std::cout << g_str_this_version_not_implemented << std::endl;
  Py_RETURN_NONE;
}

static PyObject *wrap_training2_nooutput(PyObject *self, PyObject *args) {
  std::cout << g_str_this_version_not_implemented << std::endl;
  Py_RETURN_NONE;
}

static PyObject *wrap_nnn_read(PyObject *self, PyObject *args) {
  std::cout << g_str_this_version_not_implemented << std::endl;
  Py_RETURN_NONE;
}

static PyObject *wrap_nnn_write(PyObject *self, PyObject *args) {
  std::cout << g_str_this_version_not_implemented << std::endl;
  Py_RETURN_NONE;
}

static PyObject *wrap_load(PyObject *self, PyObject *args) {
  if (pthread_mutex_init(&g_lock, NULL) != 0) {
    return do_except(PYNANN_ERROR_INTERNAL);
  }
  
  return Py_BuildValue("i", PYNANN_ERROR_SUCCESS);
}

static PyObject *wrap_unload(PyObject *self, PyObject *args) {
  
  for (auto i : g_mgrlist) {
    if (i.second.mgr) delete i.second.mgr;
  }
  g_mgrlist.clear();
  
  if (pthread_mutex_destroy(&g_lock) != 0) {
    return do_except(PYNANN_WARNING_INTERNAL);
  }
  
  return Py_BuildValue("i", PYNANN_ERROR_SUCCESS);
}

static PyObject *wrap_print(PyObject *self, PyObject *args) {
  char *task = nullptr;
  if (!PyArg_ParseTuple(args, "s", &task)) {
    return do_except(PYNANN_WARNING_INVALID_ARGUMENT);
  }
  
  if (g_mgrlist.find(task) == g_mgrlist.end()) {
    std::cout << "this task is not exist" << std::endl;
  }
  
  std::cout << "algorithm = " << g_mgrlist[task].alg << std::endl;
  std::cout << "number of input = " << g_mgrlist[task].ann.ninput << std::endl;
  std::cout << "number of output = " << g_mgrlist[task].ann.noutput << std::endl;
  std::cout << "number of hidden = " << g_mgrlist[task].ann.nhidden << std::endl;

  std::cout << "numbers of neural on each hidden = [";
  for (auto i : g_mgrlist[task].ann.nneural) {
    std::cout << std::setw(4) << i;
  }
  std::cout << "]" << std::endl;
  
  for (size_t n = 0; n < g_mgrlist[task].ann.weight_matrixes.size(); n++) {
    std::cout << "weight matrix[" << n << "] = " << std::endl;
    g_mgrlist[task].ann.weight_matrixes[n].print();
    std::cout << std::endl;
  }
  
  for (size_t n = 0; n < g_mgrlist[task].ann.delta_weight_matrixes.size(); n++) {
    std::cout << "delta weight matrix[" << n << "] = " << std::endl;
    g_mgrlist[task].ann.delta_weight_matrixes[n].print();
    std::cout << std::endl;
  }
  
  Py_RETURN_NONE;
}

static PyObject *wrap_iscalcing(PyObject *self, PyObject *args) {
  
  char *task = nullptr;
  if (!PyArg_ParseTuple(args, "s", &task)) {
    return do_except(PYNANN_WARNING_INVALID_ARGUMENT);
  }
  
  if (g_mgrlist.find(task) == g_mgrlist.end()) {
    return do_except(PYNANN_ERROR_INVALID_ARGUMENT_TASK_NOT_EXIST);
    //return Py_BuildValue("i", 0);
  }
  
  /* 检查正在计算的数量 */
  return Py_BuildValue("i", g_mgrlist[task].mgr->exist_task(task));
}

static PyObject *wrap_merge(PyObject *self, PyObject *args) {
  
  char *task = nullptr;
  if (!PyArg_ParseTuple(args, "s", &task)) {
    return do_except(PYNANN_WARNING_INVALID_ARGUMENT);
  }
  
  if (g_mgrlist.find(task) == g_mgrlist.end()) {
    return do_except(PYNANN_ERROR_INVALID_ARGUMENT_TASK_NOT_EXIST);
  }
  
  /* 检查正在计算的数量 */
  try {
    g_mgrlist[task].mgr->merge_ann_by_task(task);
  } catch (std::exception e) {
    return do_except(PYNANN_ERROR_INTERNAL, e.what());
  }
  
  return Py_BuildValue("i", PYNANN_ERROR_SUCCESS);
}

static PyMethodDef nannMethods[] = {
  { "load", wrap_load, METH_NOARGS, "load nann." },
  { "unload", wrap_unload, METH_NOARGS, "unload nann." },
  { "test", wrap_test, METH_NOARGS, "test nann." },
  { "version", wrap_version, METH_NOARGS, "get nann version." },
  { "create", wrap_create, METH_VARARGS, "nann init." },
  { "destroy", wrap_destroy, METH_NOARGS, "nann close." },
  { "exptype", wrap_exptype, METH_VARARGS, "change handle except type." },
  { "training", wrap_training, METH_VARARGS, "train sample, adjust weight & output result." },
  { "training_notarget", wrap_training_notarget, METH_VARARGS, "train sample, not adjust weight & output result." },
  { "training_nooutput", wrap_training_nooutput, METH_VARARGS, "train sample, adjust weight & not output result." },
  { "training2", wrap_training2, METH_VARARGS, "train sample, adjust weight & output result immediately, not call callback." },
  { "training2_notarget", wrap_training2_notarget, METH_VARARGS,
    "train sample, not adjust weight & output result immediately, not call callback." },
  { "training2_nooutput", wrap_training2_nooutput, METH_VARARGS,
    "train sample, adjust weight & not output result, not call callback." },
  { "nnn_read", wrap_nnn_read, METH_VARARGS, "read an ann from nnn file." },
  { "nnn_write", wrap_nnn_write, METH_VARARGS, "write an ann to nnn file." },
  { "print", wrap_print, METH_NOARGS, "print ann from task." },
  { "iscalcing", wrap_iscalcing, METH_VARARGS, "some task is running." },
  { "merge", wrap_merge, METH_VARARGS, "merge task's all ann to one." },
  { NULL, NULL, 0, NULL }
};

PyMODINIT_FUNC initnann(void) {
  PyObject *m = Py_InitModule("nann", nannMethods);
  if (m == NULL)
    return;
}