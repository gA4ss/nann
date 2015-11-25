#include <cstdio>
#include <iostream>
#include <fstream>
#include <ostream>
#include <sstream>
#include <string>
#include <map>
#include <stdexcept>
#include <iomanip>

#include <pthread.h>
#include <Python/Python.h>

#include <nanai_common.h>
#include <nanai_ann_nannmgr.h>
#include <nan_object.h>
#include <nanai_object.h>
#include <nanai_ann_nnn.h>

using namespace nanai;

#define PYNANN_ERROR_SUCCESS                           0

/* 错误信息，不可忽略 */
#define PYNANN_ERROR_INTERNAL                           0x821F0000
#define PYNANN_ERROR_PARSE_JSON                         0x821F0001
#define PYNANN_ERROR_OPEN_FILE                          0x821F0002
#define PYNANN_ERROR_INVALID_ARGUMENT                   0x821F0103
#define PYNANN_ERROR_INVALID_ARGUMENT_TASK_EXIST        0x821F0104
#define PYNANN_ERROR_INVALID_ARGUMENT_TASK_NOT_EXIST    0x821F0105
#define PYNANN_ERROR_ALLOC_MEMORY                       0x821F0106
#define PYNANN_ERROR_BY_NANN                            0x821FFFFE
#define PYNANN_ERROR_PY_INVALID_ARGUMENT                0x821FFFFF

static std::shared_ptr<nanai::nanai_ann_nannmgr> g_mgrlist = nullptr;
//static const char *g_str_this_version_not_implemented = "this version  ot implemented!!!";
static bool g_throw_exp = true;
static int g_precision = 4;

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
    { static_cast<int>(PYNANN_ERROR_BY_NANN), PyExc_StandardError, "error on nann" },
    { static_cast<int>(PYNANN_ERROR_PY_INVALID_ARGUMENT), PyExc_TypeError, "pass to python argument type error" },
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

static PyObject *wrap_test(PyObject *self, PyObject *args) {
  std::cout << "hello nanan" << std::endl;
  Py_RETURN_NONE;
}

static PyObject *wrap_version(PyObject *self, PyObject *args) {
  return Py_BuildValue("s", nanai_ann_nannmgr::version());
}

static PyObject *wrap_exptype(PyObject *self, PyObject *args) {
  int code = 0;
  
  if (!PyArg_ParseTuple(args, "i", &code)) {
    return do_except(PYNANN_ERROR_PY_INVALID_ARGUMENT);
  }
  
  g_throw_exp = (code > 0);
  
  Py_RETURN_NONE;
}

static PyObject *wrap_training(PyObject *self, PyObject *args) {
  char *task = nullptr, *ann_json = nullptr, *input_json = nullptr;
  int wt = 0;
  
  if (!PyArg_ParseTuple(args, "sssi", &task, &ann_json, &input_json, &wt)) {
    return do_except(PYNANN_ERROR_PY_INVALID_ARGUMENT);
  }
  
  try {
    
    if (g_mgrlist == nullptr) {
      g_mgrlist = std::shared_ptr<nanai::nanai_ann_nannmgr>(new nanai::nanai_ann_nannmgr());
      if (g_mgrlist == nullptr) {
        return do_except(PYNANN_ERROR_ALLOC_MEMORY);
      }
    }
    
    g_mgrlist->training(task, ann_json, input_json, wt);
  } catch (nanan::nan_error e) {
    return do_except(PYNANN_ERROR_BY_NANN, e.what());
  } catch (...) {
    return do_except(PYNANN_ERROR_INTERNAL, "training error");
  }
  
  return Py_BuildValue("i", PYNANN_ERROR_SUCCESS);
}

/*! 指定任务名的正在计算结点有多少个 */
static PyObject *wrap_done(PyObject *self, PyObject *args) {
  char *task = nullptr;
  if (!PyArg_ParseTuple(args, "s", &task)) {
    return do_except(PYNANN_ERROR_PY_INVALID_ARGUMENT);
  }

  bool b = false;
  try {
    b = g_mgrlist->mapreduce_is_done(task);
  } catch (nanan::nan_error e) {
    return do_except(PYNANN_ERROR_BY_NANN, e.what());
  } catch (...) {
    return do_except(PYNANN_ERROR_INTERNAL, "done error");
  }
  
  /* 检查正在计算的数量 */
  return Py_BuildValue("i", static_cast<int>(b));
}

static PyObject *wrap_clear(PyObject *self, PyObject *args) {
  char *task = nullptr;
  if (!PyArg_ParseTuple(args, "s", &task)) {
    return do_except(PYNANN_ERROR_PY_INVALID_ARGUMENT);
  }
  
  try {
    g_mgrlist->clear_mapreduce(task);
  } catch (nanan::nan_error e) {
    return do_except(PYNANN_ERROR_BY_NANN, e.what());
  } catch (...) {
    return do_except(PYNANN_ERROR_INTERNAL, "clear error");
  }
  
  /* 检查正在计算的数量 */
  return Py_BuildValue("i", PYNANN_ERROR_SUCCESS);
}

static PyObject *wrap_clears(PyObject *self, PyObject *args) {
  try {
    g_mgrlist->clears();
  } catch (nanan::nan_error e) {
    return do_except(PYNANN_ERROR_BY_NANN, e.what());
  } catch (...) {
    return do_except(PYNANN_ERROR_INTERNAL, "clears error");
  }
  
  /* 检查正在计算的数量 */
  return Py_BuildValue("i", PYNANN_ERROR_SUCCESS);
}

static PyObject *wrap_get_map_results(PyObject *self, PyObject *args) {
  char *task = nullptr;
  if (!PyArg_ParseTuple(args, "s", &task)) {
    return do_except(PYNANN_ERROR_PY_INVALID_ARGUMENT);
  }
  
  std::vector<nanai_ann_nanncalc::result_t> results;
  
  try {
    results = g_mgrlist->get_map_result(task);
  } catch (nanan::nan_error e) {
    return do_except(PYNANN_ERROR_BY_NANN, e.what());
  } catch (...) {
    return do_except(PYNANN_ERROR_INTERNAL, "get_map_results error");
  }
  
  std::ostringstream oss;
  nanai_ann_nanncalc::ann_t ann;
  try {
    oss << "{\n";
    
    size_t job = 0;
    for (auto i : results) {
      /* 打印输出向量 */
      oss << "\t" << "\"Job" << job + 1 << "\": {\n";
      
      oss << "\t\t" << "\"Result\":" << "[";
      for (size_t j = 0; j < i.first.size(); j++) {
        oss << i.first[j];
        if (j < i.first.size() - 1) oss << ", ";
      }
      oss << "],\n";
      
      /* 打印神经网络 */
      oss << "\t\t" << "\"Artificial Neural Network\": ";
      ann = i.second;
      std::string ann_json;
      nanai_ann_nnn_write(ann_json, ann, g_precision);
      oss << ann_json;
      
      oss << "\t" << "}";
      
      if (job < results.size() - 1) {
        oss << ",";
      }
      
      oss << "\n";
      
      job++;
    }/* end for */
    
    oss << "}";
  } catch (nanan::nan_error e) {
    return do_except(PYNANN_ERROR_BY_NANN, e.what());
  } catch (...) {
    return do_except(PYNANN_ERROR_INTERNAL, "output map results error");
  }
  
  return Py_BuildValue("s", oss.str().c_str());
}

static PyObject *wrap_get_reduce_result(PyObject *self, PyObject *args) {
  char *task = nullptr;
  if (!PyArg_ParseTuple(args, "s", &task)) {
    return do_except(PYNANN_ERROR_PY_INVALID_ARGUMENT);
  }
  
  nanai_ann_nanncalc::result_t result;
  
  try {
    result = g_mgrlist->get_reduce_result(task);
  } catch (nanan::nan_error e) {
    return do_except(PYNANN_ERROR_BY_NANN, e.what());
  } catch (...) {
    return do_except(PYNANN_ERROR_INTERNAL, "get_reduce_result error");
  }
  
  std::ostringstream oss;
  nanmath::nanmath_vector output = result.first;
  nanai_ann_nanncalc::ann_t ann = result.second;
  try {
    oss << "{\n";
    oss << "\t" << "\"Result\": " << "[";
    for (size_t j = 0; j < output.size(); j++) {
      oss << output[j];
      if (j < output.size() - 1) oss << ", ";
    }
    oss << "],\n";

    oss << "\t" << "\"Artificial Neural Network\": ";
    std::string ann_json;
    nanai_ann_nnn_write(ann_json, ann, g_precision);
    oss << ann_json;
    oss << "}";
  } catch (nanan::nan_error e) {
    return do_except(PYNANN_ERROR_BY_NANN, e.what());
  } catch (...) {
    return do_except(PYNANN_ERROR_INTERNAL, "output reduce result error");
  }
  
  return Py_BuildValue("s", oss.str().c_str());
}

static PyObject *wrap_set_precision(PyObject *self, PyObject *args) {
  int p = 0;
  if (!PyArg_ParseTuple(args, "i", &p)) {
    return do_except(PYNANN_ERROR_PY_INVALID_ARGUMENT);
  }
  
  if ((p < 0) || (p > 12)) {
    g_precision = 8;
  } else {
    g_precision = p;
  }
  
  Py_RETURN_NONE;
}

static PyObject *wrap_wait(PyObject *self, PyObject *args) {
  char *task = nullptr;
  if (!PyArg_ParseTuple(args, "s", &task)) {
    return do_except(PYNANN_ERROR_PY_INVALID_ARGUMENT);
  }
  
  try {
    g_mgrlist->wait(task);
  } catch (nanan::nan_error e) {
    return do_except(PYNANN_ERROR_BY_NANN, e.what());
  } catch (...) {
    return do_except(PYNANN_ERROR_INTERNAL, "wait task done error");
  }
  
  Py_RETURN_NONE;
}

static PyObject *wrap_waits(PyObject *self, PyObject *args) {
  g_mgrlist->waits();
  Py_RETURN_NONE;
}

static PyObject *wrap_start_auto_clear(PyObject *self, PyObject *args) {
  g_mgrlist->start_auto_clear();
  Py_RETURN_NONE;
}

static PyObject *wrap_stop_auto_clear(PyObject *self, PyObject *args) {
  g_mgrlist->stop_auto_clear();
  Py_RETURN_NONE;
}

static PyObject *wrap_create(PyObject *self, PyObject *args) {
  if (g_mgrlist) {
    return Py_BuildValue("i", PYNANN_ERROR_SUCCESS);
  }
  
  try {
    
    if (g_mgrlist == nullptr) {
      g_mgrlist = std::shared_ptr<nanai::nanai_ann_nannmgr>(new nanai::nanai_ann_nannmgr());
      if (g_mgrlist == nullptr) {
        return do_except(PYNANN_ERROR_ALLOC_MEMORY);
      }
    }
  } catch (nanan::nan_error e) {
    return do_except(PYNANN_ERROR_BY_NANN, e.what());
  } catch (...) {
    return do_except(PYNANN_ERROR_INTERNAL, "create error");
  }
  
  return Py_BuildValue("i", PYNANN_ERROR_SUCCESS);
}

static PyObject *wrap_destroy(PyObject *self, PyObject *args) {
  if (g_mgrlist == nullptr) {
    return Py_BuildValue("i", PYNANN_ERROR_SUCCESS);
  }
  
  try {
    
    if (g_mgrlist) {
      g_mgrlist = nullptr;
    }
  } catch (nanan::nan_error e) {
    return do_except(PYNANN_ERROR_BY_NANN, e.what());
  } catch (...) {
    return do_except(PYNANN_ERROR_INTERNAL, "destroy error");
  }
  
  return Py_BuildValue("i", PYNANN_ERROR_SUCCESS);
}

/**************************************************************************************************************/

static PyMethodDef nannMethods[] = {
  { "test", wrap_test, METH_NOARGS, "test nann." },
  { "version", wrap_version, METH_NOARGS, "get nann version." },
  { "exptype", wrap_exptype, METH_VARARGS, "change handle except type." },
  { "training", wrap_training, METH_VARARGS, "train sample." },
  { "done", wrap_done, METH_VARARGS, "some task is running." },
  { "clear", wrap_clear, METH_VARARGS, "clear task which is done." },
  { "clears", wrap_clears, METH_NOARGS, "clear all tasks which is done." },
  { "get_map_results", wrap_get_map_results, METH_VARARGS, "get task map results." },
  { "get_reduce_result", wrap_get_reduce_result, METH_VARARGS, "get task reduce result." },
  { "set_precision", wrap_set_precision, METH_VARARGS, "set ann output precision." },
  { "wait", wrap_wait, METH_VARARGS, "wait task done." },
  { "waits", wrap_waits, METH_NOARGS, "wait all task done." },
  { "start_auto_clear", wrap_start_auto_clear, METH_NOARGS, "start auto clear thread." },
  { "stop_auto_clear", wrap_stop_auto_clear, METH_NOARGS, "stop auto clear thread." },
  { "create", wrap_create, METH_NOARGS, "create nann manager." },
  { "destroy", wrap_destroy, METH_NOARGS, "destroy nann manager." },
  { NULL, NULL, 0, NULL }
};

PyMODINIT_FUNC initnann(void) {
  PyObject *m = Py_InitModule("nann", nannMethods);
  if (m == NULL)
    return;
}