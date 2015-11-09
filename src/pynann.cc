#include <cstdio>
#include <iostream>
#include <fstream>
#include <map>
#include <pthread.h>
#include <stdexcept>
#include <iomanip>
#include <Python/Python.h>
#include <nanai_common.h>
#include <nanai_ann_nannmgr.h>
#include <nanai_object.h>
#include <nanai_ann_nnn.h>

using namespace nanai;

#define PYNANN_ERROR_SUCCESS                        0

/* 错误信息，不可忽略 */
#define PYNANN_ERROR_INTERNAL                           0x88000000
#define PYNANN_ERROR_PARSE_JSON                         0x88000001
#define PYNANN_ERROR_OPEN_FILE                          0x88000002
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
//static const char *g_str_this_version_not_implemented = "this version  ot implemented!!!";
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
    { static_cast<int>(PYNANN_ERROR_OPEN_FILE), PyExc_OSError, "open file error" },
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

static PyObject *create_nannmgr(const char *task,
                                const char *json,
                                int max_calc,
                                int now_calc) {
  std::string alg;
  nanai_ann_nanncalc::ann_t ann;
  nanmath::nanmath_vector target;
  
  if (max_calc) {
    if ((max_calc < now_calc) || (now_calc < 0)) {
      return do_except(PYNANN_ERROR_INVALID_ARGUMENT);
    }
  }
  
  /* 读取create.json */
  try {
    nanai_ann_nnn_read(json, alg, ann, &target);
  } catch (nanai_error_logic_invalid_argument e) {
    return do_except(PYNANN_ERROR_INVALID_ARGUMENT, e.what());
  } catch (nanai_error_logic_invalid_config e) {
    return do_except(PYNANN_ERROR_PARSE_JSON, e.what());
  } catch (...) {
    return do_except(PYNANN_ERROR_INTERNAL);
  }
  
  lock();
  if (g_mgrlist.find(task) != g_mgrlist.end()) {
    /* 任务已经存在 ...  */
    unlock();
    return do_except(PYNANN_ERROR_INVALID_ARGUMENT_TASK_EXIST);
  }
  
  nanai_ann_nannmgr *nannmgr = nullptr;
  try {
    if (max_calc == 0) nannmgr = new nanai_ann_nannmgr(alg, ann, &target, task);
    else nannmgr = new nanai_ann_nannmgr(alg, ann, &target, task, max_calc, now_calc);
  } catch (...) {
    return do_except(PYNANN_ERROR_INTERNAL);
  }
  
  if (nannmgr == nullptr) {
    return do_except(PYNANN_ERROR_ALLOC_MEMORY);
  }
  
  nannmgr_node_t node;
  node.task = task;
  node.alg = alg;
  node.ann = ann;
  node.mgr = nannmgr;
  
  g_mgrlist[task] = node;
  unlock();
  
  return Py_BuildValue("i", g_mgrlist.size()-1);
}

static PyObject *wrap_create(PyObject *self, PyObject *args) {
  int max_calc = 0, now_calc = 0;
  char *task = nullptr, *json = nullptr;
  std::string alg;
  nanai_ann_nanncalc::ann_t ann;
  nanmath::nanmath_vector target;
  
  if (!PyArg_ParseTuple(args, "ssi|i", &task, &json, &max_calc, &now_calc)) {
    return do_except(PYNANN_ERROR_INVALID_ARGUMENT);
  }
  
  return create_nannmgr(task, json, max_calc, now_calc);
}

static PyObject *wrap_destroy(PyObject *self, PyObject *args) {
  char *task_arg = nullptr;
  std::string task;
  
  if (!PyArg_ParseTuple(args, "s", &task_arg)) {
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
    } catch (...) {
      return do_except(PYNANN_ERROR_INTERNAL);
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

static PyObject *wrap_training(PyObject *self, PyObject *args) {
  char *task_arg = nullptr, *json = nullptr;
  std::string task;
  std::vector<nanmath::nanmath_vector> samples;
  nanmath::nanmath_vector target;
  
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
  try {
    nanai_support_input_json(json, samples, &target);
  } catch (nanai_error_logic_invalid_argument e) {
    return do_except(PYNANN_ERROR_INVALID_ARGUMENT, e.what());
  } catch (nanai_error_logic_invalid_config e) {
    return do_except(PYNANN_ERROR_PARSE_JSON, e.what());
  } catch (...) {
    return do_except(PYNANN_ERROR_INTERNAL);
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
  
  if (!PyArg_ParseTuple(args, "ss", &task_arg, &json)) {
    return do_except(PYNANN_ERROR_SUCCESS);
  }
  
  task = task_arg;
  if (g_mgrlist.find(task) == g_mgrlist.end()) {
    /* 没有找到任务，则返回警告 */
    return do_except(PYNANN_WARNING_INVALID_ARGUMENT_TASK_NOT_EXIST);
  }
  
  /* 解析json文件 */
  try {
    nanai_support_input_json(json, samples, nullptr);
  } catch (nanai_error_logic_invalid_argument e) {
    return do_except(PYNANN_ERROR_INVALID_ARGUMENT, e.what());
  } catch (nanai_error_logic_invalid_config e) {
    return do_except(PYNANN_ERROR_PARSE_JSON, e.what());
  } catch (...) {
    return do_except(PYNANN_ERROR_INTERNAL);
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
  try {
    nanai_support_input_json(json, samples, nullptr);
  } catch (nanai_error_logic_invalid_argument e) {
    return do_except(PYNANN_ERROR_INVALID_ARGUMENT, e.what());
  } catch (nanai_error_logic_invalid_config e) {
    return do_except(PYNANN_ERROR_PARSE_JSON, e.what());
  } catch (...) {
    return do_except(PYNANN_ERROR_INTERNAL);
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

static PyObject *wrap_nnn_read(PyObject *self, PyObject *args) {
  char *json_path = nullptr;
  
  if (!PyArg_ParseTuple(args, "s", &json_path)) {
    return do_except(PYNANN_ERROR_SUCCESS);
  }
  
  /* 取出任务名 */
  std::string task = nanai_support_just_filename(json_path);
  
  std::fstream file;
  file.open(json_path, std::ios::in|std::ios::binary);
  if (file.is_open() == false) {
    do_except(PYNANN_ERROR_INVALID_ARGUMENT, json_path);
  }
  file.seekg(0, std::ios::end);
  size_t fs = static_cast<size_t>(file.tellg());
  char *buf = new char [fs+1];
  if (buf == nullptr) {
    error(NANAI_ERROR_RUNTIME_ALLOC_MEMORY);
  }
  memset(buf, 0, fs+1);
  file.read(buf, fs);
  std::string json_context = buf;
  delete [] buf;
  file.close();
  
  return create_nannmgr(task.c_str(), json_context.c_str(), 0, 0);
}

static PyObject *wrap_nnn_write(PyObject *self, PyObject *args) {
  char *json_path = nullptr;
  char *task = nullptr;
  
  if (!PyArg_ParseTuple(args, "s|s", &task, &json_path)) {
    return do_except(PYNANN_ERROR_SUCCESS);
  }
  
  lock();
  if (g_mgrlist.find(task) == g_mgrlist.end()) {
    /* 任务已经存在 ...  */
    unlock();
    return do_except(PYNANN_ERROR_INVALID_ARGUMENT_TASK_NOT_EXIST);
  }
  unlock();
  
  std::string dir;
  if (json_path) dir = json_path;
  else dir = g_mgrlist[task].mgr->get_home_dir();
  
  try {
    /* 首先合并所有网络 */
    nanai_ann_nanncalc::ann_t ann;
    g_mgrlist[task].mgr->merge_ann_by_task(task, ann);
    
    std::string alg = g_mgrlist[task].mgr->get_alg();
    nanmath::nanmath_vector target = g_mgrlist[task].mgr->get_target();
    std::string json_context;
    nanai_ann_nnn_write(json_context, alg, ann, &target);
    
    /* 写入到文件 */
    std::ofstream file;
    std::string outfile = dir;
    if (outfile.back() != '/') outfile += "/";
    outfile += task;
    outfile += ".json";
    
    file.open(outfile, std::ios::out);
    if (file.is_open() == false) {
      return do_except(PYNANN_ERROR_OPEN_FILE);
    }
    
    file << json_context;
    file.close();
    
  } catch (nanai_error_logic_invalid_argument e) {
    return do_except(PYNANN_ERROR_INVALID_ARGUMENT, e.what());
  } catch (nanai_error_logic_invalid_config e) {
    return do_except(PYNANN_ERROR_PARSE_JSON, e.what());
  } catch (nanai_error_logic_ann_number_less_2 e) {
    return do_except(PYNANN_ERROR_INTERNAL, e.what());
  } catch (...) {
    return do_except(PYNANN_ERROR_INTERNAL);
  }
  
  return Py_BuildValue("i", PYNANN_ERROR_SUCCESS);
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

/*! 打印指定任务配置信息 */
static PyObject *wrap_print_info(PyObject *self, PyObject *args) {
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
    std::cout << std::setiosflags(std::ios::left) << std::setw(4) << i;
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
  
  std::string json_context;
  try {
    /* 首先合并所有网络 */
    nanai_ann_nanncalc::ann_t ann;
    g_mgrlist[task].mgr->merge_ann_by_task(task, ann);
    
    std::string alg = g_mgrlist[task].mgr->get_alg();
    nanmath::nanmath_vector target = g_mgrlist[task].mgr->get_target();
    nanai_ann_nnn_write(json_context, alg, ann, &target);
  } catch (nanai_error_logic_invalid_argument e) {
    return do_except(PYNANN_ERROR_INVALID_ARGUMENT, e.what());
  } catch (nanai_error_logic_invalid_config e) {
    return do_except(PYNANN_ERROR_PARSE_JSON, e.what());
  } catch (nanai_error_logic_ann_number_less_2 e) {
    return do_except(PYNANN_ERROR_INTERNAL, e.what());
  } catch (...) {
    return do_except(PYNANN_ERROR_INTERNAL);
  }
  
  return Py_BuildValue("s", json_context.c_str());
}

/**************************************************************************************************************/

static PyMethodDef nannMethods[] = {
  { "load", wrap_load, METH_NOARGS, "load nann." },
  { "unload", wrap_unload, METH_NOARGS, "unload nann." },
  { "test", wrap_test, METH_NOARGS, "test nann." },
  { "version", wrap_version, METH_NOARGS, "get nann version." },
  { "create", wrap_create, METH_VARARGS, "create nann manager." },
  { "destroy", wrap_destroy, METH_VARARGS, "nann manager close." },
  { "exptype", wrap_exptype, METH_VARARGS, "change handle except type." },
  { "training", wrap_training, METH_VARARGS, "train sample, adjust weight & output result." },
  { "training_notarget", wrap_training_notarget, METH_VARARGS, "train sample, not adjust weight & output result." },
  { "training_nooutput", wrap_training_nooutput, METH_VARARGS, "train sample, adjust weight & not output result." },
  { "nnn_read", wrap_nnn_read, METH_VARARGS, "read an ann from nnn file." },
  { "nnn_write", wrap_nnn_write, METH_VARARGS, "write an ann to nnn file." },
  { "print_info", wrap_print_info, METH_VARARGS, "print ann from task." },
  { "iscalcing", wrap_iscalcing, METH_VARARGS, "some task is running." },
  { "merge", wrap_merge, METH_VARARGS, "merge task's all ann to one." },
  { NULL, NULL, 0, NULL }
};

PyMODINIT_FUNC initnann(void) {
  PyObject *m = Py_InitModule("nann", nannMethods);
  if (m == NULL)
    return;
}