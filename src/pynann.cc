#include <stdio.h>
#include <Python/Python.h>
#include <nanai_ann_nannmgr.h>

using namespace nanai;

/* 每个pynann唯一的实例子 */
nanai_ann_nannmgr *g_nannmgr;

#define PYNANN_ERROR_SUCCESS                        0
#define PYNANN_ERROR_INVALID_ARGUMENT               -1
#define PYNANN_ERROR_ALLOC_MEMORY                   -2

#ifdef __cplusplus
extern "C"
{
#endif

int init(int max, int now) {
  
  if (max < now) {
    return PYNANN_ERROR_INVALID_ARGUMENT;
  }
  
  g_nannmgr = new nanai_ann_nannmgr(max, now);
  if (g_nannmgr == nullptr) {
    return PYNANN_ERROR_ALLOC_MEMORY;
  }
  
  return PYNANN_ERROR_SUCCESS;
}

void destroy() {
  if (g_nannmgr) {
    delete g_nannmgr;
  }
}

const char *version() {
  if (g_nannmgr == nullptr) {
    return nullptr;
  }
  return g_nannmgr->version();
}

const char * const errstr(int code) {
  
  static const struct {
    int code;
    const char *msg;
  } msgs[] = {
    { PYNANN_ERROR_SUCCESS, "succeeful" },
    { PYNANN_ERROR_INVALID_ARGUMENT, "invalid argument" },
    { PYNANN_ERROR_ALLOC_MEMORY, "alloc memory failed" }
  };
  
  int x;
  for (x = 0; x < (int)(sizeof(msgs) / sizeof(msgs[0])); x++) {
    if (msgs[x].code == code) {
      return msgs[x].msg;
    }
  }
  return "invalid error code";
}

#ifdef __cplusplus
}
#endif

static PyObject *wrap_init(PyObject *self, PyObject *args) {
  int max_calc = 0, now_calc = 0;
  
  if (!PyArg_ParseTuple(args, "i|i", &max_calc, &now_calc)) {
    PyErr_SetString(PyExc_TypeError, "both of integer type");
    return nullptr;
  }
  
  return Py_BuildValue("i", init(max_calc, now_calc));
}

static PyObject *wrap_destroy(PyObject *self, PyObject *args) {
  destroy();
  Py_RETURN_NONE;
}

static PyObject *wrap_version(PyObject *self, PyObject *args) {
  return Py_BuildValue("s", version());
}

static PyObject *wrap_errstr(PyObject *self, PyObject *args) {
  int code = 0;
  
  if (!PyArg_ParseTuple(args, "i", &code)) {
    PyErr_SetString(PyExc_TypeError, "argument must be integer type");
    return nullptr;
  }
  
  return Py_BuildValue("s", errstr(code));
}

static PyMethodDef nannMethods[] = {
  { "test", wrap_version, METH_NOARGS, "get nann version." },
  { "init", wrap_init, METH_VARARGS, "nann init." },
  { "destroy", wrap_destroy, METH_NOARGS, "nann close." },
  { "errstr", wrap_errstr, METH_VARARGS, "return string descript error." },
  { NULL, NULL, 0, NULL }
};

#ifdef __cplusplus
extern "C"
{
#endif

PyMODINIT_FUNC initnann(void) {
  PyObject *m = Py_InitModule("nann", nannMethods);
  if (m == NULL)
    return;
}

#ifdef __cplusplus
}
#endif