// quadfitmodule.c
#define NPY_NO_DEPRECATED_API NPY_1_19_API_VERSION
#include <Python.h>
#include <math.h>
#include "numpy/arrayobject.h"
#include "structmember.h"  // for PyMemberDef

// ------------------------ Line Type ------------------------

typedef struct {
    PyObject_HEAD
    double A, B, C;
    double norm;
} LineObject;

static int Line_compute_norm(LineObject *self) {
    self->norm = sqrt(self->A * self->A + self->B * self->B);
    if (self->norm == 0.0) {
        PyErr_SetString(PyExc_ValueError, "Degenerate line with zero normal length");
        return -1;
    }
    return 0;
}

static int Line_init(LineObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"x1","y1","x2","y2","A","B","C", NULL};
    PyObject *x1o=NULL, *y1o=NULL, *x2o=NULL, *y2o=NULL, *Ao=NULL, *Bo=NULL, *Co=NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOOOO:Line", kwlist,
                                     &x1o,&y1o,&x2o,&y2o,&Ao,&Bo,&Co)) {
        return -1;
    }

    int have_points = (x1o && y1o && x2o && y2o);
    int have_abc    = (Ao && Bo && Co);

    if (have_points) {
        double x1 = PyFloat_AsDouble(x1o);
        double y1 = PyFloat_AsDouble(y1o);
        double x2 = PyFloat_AsDouble(x2o);
        double y2 = PyFloat_AsDouble(y2o);
        if (PyErr_Occurred()) return -1;

        // From two-point form to Ax + By + C = 0
        self->A = y2 - y1;
        self->B = x1 - x2;
        self->C = x2*y1 - x1*y2;
    } else if (have_abc) {
        self->A = PyFloat_AsDouble(Ao);
        self->B = PyFloat_AsDouble(Bo);
        self->C = PyFloat_AsDouble(Co);
        if (PyErr_Occurred()) return -1;
    } else {
        PyErr_SetString(PyExc_ValueError, "Either (x1,y1,x2,y2) or (A,B,C) must be specified");
        return -1;
    }

    if (Line_compute_norm(self) < 0) return -1;
    return 0;
}

static PyObject* Line_copy(LineObject *self, PyObject *Py_UNUSED(ignored)) {
    PyTypeObject *tp = Py_TYPE(self);
    LineObject *obj = (LineObject *)tp->tp_alloc(tp, 0);
    if (!obj) return NULL;
    obj->A = self->A; obj->B = self->B; obj->C = self->C; obj->norm = self->norm;
    return (PyObject*)obj;
}

static PyObject* Line_move_line(LineObject *self, PyObject *args) {
    double distance;
    if (!PyArg_ParseTuple(args, "d", &distance)) return NULL;
    double delta_C = -distance * self->norm;
    self->C += delta_C;
    Py_RETURN_NONE;
}

static PyObject* Line_move_line_to_intersect_point(LineObject *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"x", "y", NULL};
    double x, y;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "dd:move_line_to_intersect_point", kwlist, &x, &y))
        return NULL;
    self->C = -(self->A * x + self->B * y);
    PyObject *t = PyTuple_New(3);
    if (!t) return NULL;
    PyTuple_SET_ITEM(t, 0, PyFloat_FromDouble(self->A));
    PyTuple_SET_ITEM(t, 1, PyFloat_FromDouble(self->B));
    PyTuple_SET_ITEM(t, 2, PyFloat_FromDouble(self->C));
    return t;
}

static PyObject* Line_point_line_position(LineObject *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"x", "y", NULL};
    double x, y;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "dd:point_line_position", kwlist, &x, &y))
        return NULL;
    double v = self->A * x + self->B * y + self->C;
    return PyFloat_FromDouble(v);
}

static PyObject* Line_distance_from_point(LineObject *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"x", "y", NULL};
    double x, y;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "dd:distance_from_point", kwlist, &x, &y))
        return NULL;
    double v = fabs(self->A * x + self->B * y + self->C) / self->norm;
    return PyFloat_FromDouble(v);
}

static PyObject* Line_distances_from_points(LineObject *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"points", NULL};
    PyObject *points_obj;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O:distances_from_points", kwlist, &points_obj))
        return NULL;

    PyArrayObject *points = (PyArrayObject*)PyArray_FROM_OTF(points_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!points) return NULL;

    if (PyArray_NDIM(points) != 2 || PyArray_DIM(points,1) != 2) {
        Py_DECREF(points);
        PyErr_SetString(PyExc_AssertionError, "Input array must be of shape (N, 2)");
        return NULL;
    }

    npy_intp N = PyArray_DIM(points,0);
    npy_intp dims[4] = {N};
    PyArrayObject *out = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!out) { Py_DECREF(points); return NULL; }

    double *p = (double*)PyArray_DATA(points);
    double *o = (double*)PyArray_DATA(out);
    npy_intp stride0 = PyArray_STRIDE(points,0) / sizeof(double);
    npy_intp stride1 = PyArray_STRIDE(points,1) / sizeof(double);

    npy_intp i;
    for (i = 0; i < N; ++i) {
        double x = *(p + i*stride0 + 0*stride1);
        double y = *(p + i*stride0 + 1*stride1);
        double val = fabs(self->A * x + self->B * y + self->C) / self->norm;
        o[i] = val;
    }

    Py_DECREF(points);
    return (PyObject*)out;
}

static PyObject* Line_get_intersection(LineObject *self, PyObject *args) {
    PyObject *other_obj;
    if (!PyArg_ParseTuple(args, "O", &other_obj)) return NULL;
    if (!PyObject_TypeCheck(other_obj, Py_TYPE(self))) {
        PyErr_SetString(PyExc_TypeError, "other_line must be a Line");
        return NULL;
    }
    LineObject *other = (LineObject*)other_obj;
    double det = self->A * other->B - other->A * self->B;
    if (det == 0.0) {
        Py_RETURN_NONE;
    }
    double x = (-self->C * other->B + other->C * self->B) / det;
    double y = (-self->A * other->C + other->A * self->C) / det;
    PyObject *t = PyTuple_New(2);
    if (!t) return NULL;
    PyTuple_SET_ITEM(t, 0, PyFloat_FromDouble(x));
    PyTuple_SET_ITEM(t, 1, PyFloat_FromDouble(y));
    return t;
}

// Read-only getters
static PyObject* Line_get_A(LineObject* self, void* closure) {
    return PyFloat_FromDouble(self->A);
}
static PyObject* Line_get_B(LineObject* self, void* closure) {
    return PyFloat_FromDouble(self->B);
}
static PyObject* Line_get_C(LineObject* self, void* closure) {
    return PyFloat_FromDouble(self->C);
}
static PyObject* Line_get_norm(LineObject* self, void* closure) {
    return PyFloat_FromDouble(self->norm);
}

static PyGetSetDef Line_getset[] = {
    {"A", (getter)Line_get_A, NULL, "A coefficient", NULL},
    {"B", (getter)Line_get_B, NULL, "B coefficient", NULL},
    {"C", (getter)Line_get_C, NULL, "C constant", NULL},
    {"norm", (getter)Line_get_norm, NULL, "sqrt(A^2 + B^2)", NULL},
    {NULL}
};

static PyMethodDef Line_methods[] = {
    {"copy", (PyCFunction)Line_copy, METH_NOARGS, "Return a copy of the line"},
    {"move_line", (PyCFunction)Line_move_line, METH_VARARGS, "Translate the line orthogonally by distance"},
    {"move_line_to_intersect_point", (PyCFunction)Line_move_line_to_intersect_point, METH_VARARGS | METH_KEYWORDS, "Move line to pass through point (x,y) and return (A,B,C)"},
    {"point_line_position", (PyCFunction)Line_point_line_position, METH_VARARGS | METH_KEYWORDS, "Return Ax + By + C"},
    {"distance_from_point", (PyCFunction)Line_distance_from_point, METH_VARARGS | METH_KEYWORDS, "Perpendicular distance from point (x,y)"},
    {"distances_from_points", (PyCFunction)Line_distances_from_points, METH_VARARGS | METH_KEYWORDS, "Vectorized distances from Nx2 array"},
    {"get_intersection", (PyCFunction)Line_get_intersection, METH_VARARGS, "Intersection with another line or None"},
    {NULL, NULL, 0, NULL}
};

static PyMemberDef Line_members[] = {
    {"A", T_DOUBLE, offsetof(LineObject, A), 0, "A coefficient"},
    {"B", T_DOUBLE, offsetof(LineObject, B), 0, "B coefficient"},
    {"C", T_DOUBLE, offsetof(LineObject, C), 0, "C constant"},
    {"norm", T_DOUBLE, offsetof(LineObject, norm), READONLY, "sqrt(A^2 + B^2)"},
    {NULL}
};

static PyTypeObject LineType = { PyVarObject_HEAD_INIT(NULL, 0) };

// ------------------------ Helpers ------------------------

typedef struct {
    double x, y, ang;
} PtAng;

static int cmp_ptang(const void *a, const void *b) {
    const PtAng *pa = (const PtAng*)a;
    const PtAng *pb = (const PtAng*)b;
    if (pa->ang < pb->ang) return -1;
    if (pa->ang > pb->ang) return 1;
    return 0;
}

// order_points_clockwise(points: array[N,2]) -> array[N,2]
static PyObject* mod_order_points_clockwise(PyObject *self, PyObject *args) {
    PyObject *points_obj;
    if (!PyArg_ParseTuple(args, "O", &points_obj)) return NULL;

    PyArrayObject *points = (PyArrayObject*)PyArray_FROM_OTF(points_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!points) return NULL;

    if (PyArray_NDIM(points) != 2 || PyArray_DIM(points,1) != 2) {
        Py_DECREF(points);
        PyErr_SetString(PyExc_AssertionError, "Input array must be of shape (N, 2)");
        return NULL;
    }

    npy_intp N = PyArray_DIM(points,0);
    double *p = (double*)PyArray_DATA(points);
    npy_intp s0 = PyArray_STRIDE(points,0) / sizeof(double);
    npy_intp s1 = PyArray_STRIDE(points,1) / sizeof(double);

    double cx = 0.0, cy = 0.0;
    npy_intp i;
    for (i = 0; i < N; ++i) {
        cx += *(p + i*s0 + 0*s1);
        cy += *(p + i*s0 + 1*s1);
    }
    cx /= (double)N;
    cy /= (double)N;

    PtAng *buf = (PtAng*)PyMem_Malloc(sizeof(PtAng)* (size_t)N);
    if (!buf) { Py_DECREF(points); PyErr_NoMemory(); return NULL; }
    for (i = 0; i < N; ++i) {
        double x = *(p + i*s0 + 0*s1);
        double y = *(p + i*s0 + 1*s1);
        buf[i].x = x; buf[i].y = y;
        buf[i].ang = atan2(y - cy, x - cx);
    }

    qsort(buf, (size_t)N, sizeof(PtAng), cmp_ptang);

    npy_intp dims[2] = {N, 2};
    PyArrayObject *out = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!out) { PyMem_Free(buf); Py_DECREF(points); return NULL; }
    double *o = (double*)PyArray_DATA(out);
    for (i = 0; i < N; ++i) {
        o[2*i+0] = buf[i].x;
        o[2*i+1] = buf[i].y;
    }

    PyMem_Free(buf);
    Py_DECREF(points);
    return (PyObject*)out;
}

// polygon_vertices_from_lines((Line,...)) -> tuple[(x,y), ...]
// Intersect consecutive lines and return points ordered clockwise
static PyObject* mod_polygon_vertices_from_lines(PyObject *self, PyObject *args) {
    PyObject *seq;
    if (!PyArg_ParseTuple(args, "O", &seq)) return NULL;

    PyObject *fast = PySequence_Fast(seq, "Expected a sequence of Line");
    if (!fast) return NULL;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    PyObject **items = PySequence_Fast_ITEMS(fast);

    if (n < 3) {
        Py_DECREF(fast);
        PyErr_SetString(PyExc_ValueError, "Need at least 3 lines");
        return NULL;
    }

    PtAng *buf = (PtAng*)PyMem_Malloc(sizeof(PtAng) * (size_t)n);
    if (!buf) { Py_DECREF(fast); PyErr_NoMemory(); return NULL; }

    double cx = 0.0, cy = 0.0;
    Py_ssize_t i;
    for (i = 0; i < n; ++i) {
        PyObject *li = items[i];
        PyObject *lj = items[(i+1) % n];
        if (!PyObject_TypeCheck(li, &LineType) || !PyObject_TypeCheck(lj, &LineType)) {
            PyMem_Free(buf); Py_DECREF(fast);
            PyErr_SetString(PyExc_TypeError, "All items must be Line");
            return NULL;
        }
        LineObject *L1 = (LineObject*)li;
        LineObject *L2 = (LineObject*)lj;
        double det = L1->A * L2->B - L2->A * L1->B;
        if (det == 0.0) {
            PyMem_Free(buf); Py_DECREF(fast);
            PyErr_SetString(PyExc_ValueError, "Parallel adjacent lines have no intersection");
            return NULL;
        }
        double x = (-L1->C * L2->B + L2->C * L1->B) / det;
        double y = (-L1->A * L2->C + L2->A * L1->C) / det;
        buf[i].x = x; buf[i].y = y;
        cx += x; cy += y;
    }

    cx /= (double)n;
    cy /= (double)n;

    for (i = 0; i < n; ++i) {
        buf[i].ang = atan2(buf[i].y - cy, buf[i].x - cx);
    }
    qsort(buf, (size_t)n, sizeof(PtAng), cmp_ptang);

    PyObject *out = PyTuple_New(n);
    if (!out) { PyMem_Free(buf); Py_DECREF(fast); return NULL; }
    for (i = 0; i < n; ++i) {
        PyObject *pt = Py_BuildValue("(dd)", buf[i].x, buf[i].y);
        if (!pt) { Py_DECREF(out); PyMem_Free(buf); Py_DECREF(fast); return NULL; }
        PyTuple_SET_ITEM(out, i, pt);
    }

    PyMem_Free(buf);
    Py_DECREF(fast);
    return out;
}

// ------------------------ Module ------------------------

static PyMethodDef module_methods[] = {
    {"order_points_clockwise", (PyCFunction)mod_order_points_clockwise, METH_VARARGS, "Order Nx2 points clockwise"},
    {"polygon_vertices_from_lines", (PyCFunction)mod_polygon_vertices_from_lines, METH_VARARGS, "Intersect consecutive lines to polygon vertices (ordered)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "quadfit",
    "Quadrilateral fitting helpers in C",
    -1,
    module_methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_quadfit(void) {
    import_array();

    // set LineType fields (MSVC-friendly)
    LineType.tp_name = "quadfit.Line";
    LineType.tp_basicsize = sizeof(LineObject);
    LineType.tp_flags = Py_TPFLAGS_DEFAULT;
    LineType.tp_doc = "2D Line Ax + By + C = 0";
    LineType.tp_init = (initproc)Line_init;
    LineType.tp_new = PyType_GenericNew;
    LineType.tp_methods = Line_methods;
    LineType.tp_getset = Line_getset;

    if (PyType_Ready(&LineType) < 0) return NULL;

    PyObject *m = PyModule_Create(&moduledef);
    if (!m) return NULL;
    Py_INCREF(&LineType);
    if (PyModule_AddObject(m, "Line", (PyObject*)&LineType) < 0) {
        Py_DECREF(&LineType);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
