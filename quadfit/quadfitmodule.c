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

// Forward declarations for new accelerated functions
static PyObject* mod_best_iou_quadrilateral(PyObject* self, PyObject* args, PyObject* kwargs);
static PyObject* mod_finetune_quadrilateral(PyObject* self, PyObject* args, PyObject* kwargs);
static PyObject* mod_expand_quadrilateral(PyObject* self, PyObject* args, PyObject* kwargs);
static PyObject* mod_simplify_polygon_dp(PyObject* self, PyObject* args, PyObject* kwargs);
static PyObject* mod_convex_hull_monotone(PyObject* self, PyObject* args, PyObject* kwargs);

static PyMethodDef module_methods[] = {
    {"order_points_clockwise", (PyCFunction)mod_order_points_clockwise, METH_VARARGS, "Order Nx2 points clockwise"},
    {"polygon_vertices_from_lines", (PyCFunction)mod_polygon_vertices_from_lines, METH_VARARGS, "Intersect consecutive lines to polygon vertices (ordered)"},
    // New high-level accelerated helpers
    // best_iou_quadrilateral(points: array[N,2], max_combinations: int = 300, seed: int|None = None) -> tuple[(x,y), (x,y), (x,y), (x,y)]
    // finetune_quadrilateral(points: array[M,2], initial_vertices: array[4,2]) -> (tuple[Line, ...4], tuple[(x,y), ...4])
    // expand_quadrilateral(lines: tuple[Line, ...4], hull_points: array[K,2]) -> (tuple[Line, ...4], tuple[(x,y), ...4])
    {"best_iou_quadrilateral", (PyCFunction)mod_best_iou_quadrilateral, METH_VARARGS | METH_KEYWORDS, "Find quadrilateral (from hull vertices) with max IoU vs convex hull"},
    {"finetune_quadrilateral", (PyCFunction)mod_finetune_quadrilateral, METH_VARARGS | METH_KEYWORDS, "Assign points to nearest side, fit TLS lines, return lines and vertices"},
    {"expand_quadrilateral", (PyCFunction)mod_expand_quadrilateral, METH_VARARGS | METH_KEYWORDS, "Push lines outward to cover hull, return lines and vertices"},
    {"simplify_polygon_dp", (PyCFunction)mod_simplify_polygon_dp, METH_VARARGS | METH_KEYWORDS, "Douglas–Peucker simplification for closed polygon with IoU threshold against original (convex)"},
    {"convex_hull_monotone", (PyCFunction)mod_convex_hull_monotone, METH_VARARGS | METH_KEYWORDS, "Compute convex hull (monotone chain). Returns closed ring (H+1,2) with last point equal to first."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "quadfitmodule",
    "Quadrilateral fitting helpers in C",
    -1,
    module_methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_quadfitmodule(void) {
    import_array();

    // set LineType fields (MSVC-friendly)
    LineType.tp_name = "quadfitmodule.Line";
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

// ------------------------ Accelerated Routines ------------------------

typedef struct { double x, y; } Pt;

static inline double cross2(double ax, double ay, double bx, double by) {
    return ax*by - ay*bx;
}

static double polygon_signed_area(const Pt* pts, int n) {
    double s = 0.0;
    for (int i=0;i<n;i++) {
        int j = (i+1==n)?0:(i+1);
        s += pts[i].x*pts[j].y - pts[j].x*pts[i].y;
    }
    return 0.5*s;
}

static double polygon_area_abs(const Pt* pts, int n) {
    double a = polygon_signed_area(pts, n);
    return a < 0 ? -a : a;
}

// Ensure polygon is oriented counter-clockwise (CCW). In-place reverse if needed.
static void ensure_ccw(Pt* pts, int n) {
    if (n <= 2) return;
    double s = polygon_signed_area(pts, n);
    if (s < 0.0) {
        for (int i = 0, j = n-1; i < j; ++i, --j) {
            Pt tmp = pts[i];
            pts[i] = pts[j];
            pts[j] = tmp;
        }
    }
}

// Compute centroid of polygon (non-self-intersecting). Returns 0 on success, -1 if degenerate area.
static int polygon_centroid(const Pt* pts, int n, double* outx, double* outy) {
    double A = polygon_signed_area(pts, n);
    if (A == 0.0) { *outx = pts[0].x; *outy = pts[0].y; return -1; }
    double cx = 0.0, cy = 0.0;
    for (int i=0;i<n;i++) {
        int j = (i+1==n)?0:(i+1);
        double cross = pts[i].x*pts[j].y - pts[j].x*pts[i].y;
        cx += (pts[i].x + pts[j].x) * cross;
        cy += (pts[i].y + pts[j].y) * cross;
    }
    cx /= (6.0*A);
    cy /= (6.0*A);
    *outx = cx; *outy = cy;
    return 0;
}

// Sutherland–Hodgman clipping: clip subject polygon by convex clip polygon half-planes.
static int clip_polygon_against_edge(const Pt* subj, int sn, Pt a, Pt b, Pt* out) {
    if (sn <= 0) return 0;
    // Determine inside test based on clip edge orientation (assume CCW => inside is left side)
    double ex = b.x - a.x, ey = b.y - a.y;
    int outn = 0;
    Pt S = subj[sn-1];
    double Sc = cross2(ex,ey, S.x - a.x, S.y - a.y);
    for (int i=0;i<sn;i++) {
        Pt E = subj[i];
        double Ec = cross2(ex,ey, E.x - a.x, E.y - a.y);
        int Ein = (Ec >= 0.0);
        int Sin = (Sc >= 0.0);
        if (Ein) {
            if (!Sin) {
                // compute intersection S->E with line a->b
                double sx = S.x, sy = S.y;
                double ex2 = E.x, ey2 = E.y;
                double dx1 = ex2 - sx, dy1 = ey2 - sy;
                double dx2 = ex, dy2 = ey; // b - a
                double denom = cross2(dx1,dy1, dx2,dy2);
                Pt It = E; // fallback
                if (denom != 0.0) {
                    double t = cross2(a.x - sx, a.y - sy, dx2, dy2) / denom;
                    It.x = sx + t*dx1;
                    It.y = sy + t*dy1;
                }
                out[outn++] = It;
            }
            out[outn++] = E;
        } else if (Sin) {
            // leaving, add intersection
            double sx = S.x, sy = S.y;
            double ex2 = E.x, ey2 = E.y;
            double dx1 = ex2 - sx, dy1 = ey2 - sy;
            double dx2 = ex, dy2 = ey;
            double denom = cross2(dx1,dy1, dx2,dy2);
            Pt It = S; // fallback
            if (denom != 0.0) {
                double t = cross2(a.x - sx, a.y - sy, dx2, dy2) / denom;
                It.x = sx + t*dx1;
                It.y = sy + t*dy1;
            }
            out[outn++] = It;
        }
        S = E; Sc = Ec;
    }
    return outn;
}

static int convex_intersection(const Pt* A, int na, const Pt* B, int nb, Pt* workbuf, Pt* out) {
    // workbuf must have capacity at least na+nb
    int cn = na;
    for (int i=0;i<na;i++) workbuf[i] = A[i];
    int wn = cn;
    Pt* src = workbuf;
    Pt* dst = out;
    for (int i=0;i<nb;i++) {
        Pt a = B[i];
        Pt b = B[(i+1==nb)?0:(i+1)];
        wn = clip_polygon_against_edge(src, cn, a, b, dst);
        // swap buffers
        Pt* tmp = src; src = dst; dst = tmp;
        cn = wn;
        if (cn == 0) break;
    }
    // If last result in src, ensure it's in out
    if (cn > 0 && src != out) {
        for (int i=0;i<cn;i++) out[i] = src[i];
    }
    return cn;
}

static int ensure_double_2d(PyObject* obj, PyArrayObject** out, int* rows, int* cols) {
    PyArrayObject* arr = (PyArrayObject*)PyArray_FROM_OTF(obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!arr) return -1;
    if (PyArray_NDIM(arr) != 2) { Py_DECREF(arr); PyErr_SetString(PyExc_AssertionError, "Expected 2D array"); return -1; }
    int r = (int)PyArray_DIM(arr, 0);
    int c = (int)PyArray_DIM(arr, 1);
    if (cols && *cols > 0 && c != *cols) {
        Py_DECREF(arr);
        PyErr_SetString(PyExc_AssertionError, "Invalid second dimension");
        return -1;
    }
    if (cols) *cols = c; if (rows) *rows = r;
    *out = arr;
    return 0;
}

static int load_points_from_array(PyArrayObject* arr, Pt** out_pts, int* out_n, int drop_last_if_equal) {
    int r = (int)PyArray_DIM(arr,0);
    int c = (int)PyArray_DIM(arr,1);
    if (c != 2) { PyErr_SetString(PyExc_AssertionError, "Array must have 2 columns"); return -1; }
    double* p = (double*)PyArray_DATA(arr);
    npy_intp s0 = PyArray_STRIDE(arr,0) / sizeof(double);
    npy_intp s1 = PyArray_STRIDE(arr,1) / sizeof(double);
    int n = r;
    if (drop_last_if_equal && r >= 2) {
        double x0 = *(p + 0*s0 + 0*s1);
        double y0 = *(p + 0*s0 + 1*s1);
        double xl = *(p + (r-1)*s0 + 0*s1);
        double yl = *(p + (r-1)*s0 + 1*s1);
        if (x0 == xl && y0 == yl) n = r - 1;
    }
    Pt* pts = (Pt*)PyMem_Malloc(sizeof(Pt) * (size_t)n);
    if (!pts) { PyErr_NoMemory(); return -1; }
    for (int i=0;i<n;i++) {
        pts[i].x = *(p + i*s0 + 0*s1);
        pts[i].y = *(p + i*s0 + 1*s1);
    }
    *out_pts = pts; *out_n = n;
    return 0;
}

// -------- Douglas–Peucker for closed polygon (treat as open polyline, then close) --------

static double point_segment_distance(Pt p, Pt a, Pt b) {
    double vx = b.x - a.x, vy = b.y - a.y;
    double wx = p.x - a.x, wy = p.y - a.y;
    double c1 = vx*wx + vy*wy;
    if (c1 <= 0.0) {
        double dx = p.x - a.x, dy = p.y - a.y; return sqrt(dx*dx + dy*dy);
    }
    double c2 = vx*vx + vy*vy;
    if (c2 <= c1) {
        double dx = p.x - b.x, dy = p.y - b.y; return sqrt(dx*dx + dy*dy);
    }
    double t = c1 / c2;
    double projx = a.x + t*vx, projy = a.y + t*vy;
    double dx = p.x - projx, dy = p.y - projy; return sqrt(dx*dx + dy*dy);
}

typedef struct { int i, j; } IntPair;

static void dp_simplify(const Pt* pts, int n, double eps, char* keep) {
    for (int i=0;i<n;i++) keep[i]=0;
    keep[0]=1; keep[n-1]=1;
    // simple stack
    int cap = 64; int top = 0;
    IntPair* st = (IntPair*)PyMem_Malloc(sizeof(IntPair)* (size_t)cap);
    if (!st) return; // if OOM, keep endpoints only
    st[top++] = (IntPair){0, n-1};
    while (top>0) {
        IntPair pr = st[--top];
        int i = pr.i, j = pr.j;
        double maxd = -1.0; int idx = -1;
        Pt a = pts[i]; Pt b = pts[j];
        for (int k=i+1;k<j;k++) {
            double d = point_segment_distance(pts[k], a, b);
            if (d > maxd) { maxd = d; idx = k; }
        }
        if (maxd > eps && idx > i && idx < j) {
            keep[idx]=1;
            if (top+2 > cap) {
                int ncap = cap*2; IntPair* nst = (IntPair*)PyMem_Realloc(st, sizeof(IntPair)*(size_t)ncap);
                if (!nst) continue; st = nst; cap = ncap;
            }
            st[top++] = (IntPair){i, idx};
            st[top++] = (IntPair){idx, j};
        }
    }
    PyMem_Free(st);
}

// simplify_polygon_dp(points, max_sides, initial_epsilon, max_epsilon, epsilon_increment, iou_threshold)
static PyObject* mod_simplify_polygon_dp(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"points", "max_sides", "initial_epsilon", "max_epsilon", "epsilon_increment", "iou_threshold", NULL};
    PyObject* points_obj=NULL; int max_sides=10; double initial_epsilon=0.1, max_epsilon=0.5, epsilon_increment=0.02, iou_threshold=0.8;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|idddd:simplify_polygon_dp", kwlist,
        &points_obj, &max_sides, &initial_epsilon, &max_epsilon, &epsilon_increment, &iou_threshold)) return NULL;

    PyArrayObject* arr=NULL; int r=0, c=2;
    if (ensure_double_2d(points_obj, &arr, &r, &c) < 0) return NULL;
    Pt* P=NULL; int N=0; if (load_points_from_array(arr, &P, &N, 1) < 0) { Py_DECREF(arr); return NULL; }
    Py_DECREF(arr);
    // No simplification if None-like behavior (negative or zero) or already <= max_sides
    if (max_sides <= 0 || N <= max_sides) {
        // return original with closing point duplicated
        npy_intp dims[2] = {N+1, 2};
        PyArrayObject* out = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
        if (!out) { PyMem_Free(P); return NULL; }
        double* o = (double*)PyArray_DATA(out);
        for (int i=0;i<N;i++) { o[2*i]=P[i].x; o[2*i+1]=P[i].y; }
        o[2*N]=P[0].x; o[2*N+1]=P[0].y; PyMem_Free(P);
        return (PyObject*)out;
    }

    // Original area for IoU denom
    double orig_area = polygon_area_abs(P, N);
    // Work buffers
    char* keep = (char*)PyMem_Malloc((size_t)N);
    if (!keep) { PyMem_Free(P); PyErr_NoMemory(); return NULL; }

    // Start with previous = original
    Pt* prev_pts = (Pt*)PyMem_Malloc(sizeof(Pt)*(size_t)N);
    if (!prev_pts) { PyMem_Free(P); PyMem_Free(keep); PyErr_NoMemory(); return NULL; }
    for (int i=0;i<N;i++) prev_pts[i]=P[i];
    int prev_n = N;

    double eps = initial_epsilon;
    while (eps <= max_epsilon + 1e-12) {
        for (int i=0;i<N;i++) keep[i]=0;
        dp_simplify(P, N, eps, keep);
        // Build candidate
        int cnt = 0; for (int i=0;i<N;i++) if (keep[i]) cnt++;
        if (cnt < 4) break; // too few sides -> stop and return prev
        Pt* cand = (Pt*)PyMem_Malloc(sizeof(Pt)*(size_t)cnt);
        if (!cand) { PyMem_Free(P); PyMem_Free(keep); PyMem_Free(prev_pts); PyErr_NoMemory(); return NULL; }
        int idx=0; for (int i=0;i<N;i++) if (keep[i]) { cand[idx++]=P[i]; }
        if (cnt <= max_sides) {
            double a = polygon_area_abs(cand, cnt);
            double iou = (orig_area>0.0) ? (a / orig_area) : 0.0;
            if (iou > iou_threshold) {
                // accept and return
                npy_intp dims[2] = {cnt+1, 2};
                PyArrayObject* out = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
                if (!out) { PyMem_Free(P); PyMem_Free(keep); PyMem_Free(prev_pts); PyMem_Free(cand); return NULL; }
                double* o = (double*)PyArray_DATA(out);
                for (int i=0;i<cnt;i++) { o[2*i]=cand[i].x; o[2*i+1]=cand[i].y; }
                o[2*cnt]=cand[0].x; o[2*cnt+1]=cand[0].y;
                PyMem_Free(P); PyMem_Free(keep); PyMem_Free(prev_pts); PyMem_Free(cand);
                return (PyObject*)out;
            }
            // else return previous
            PyMem_Free(cand);
            break;
        } else {
            // keep as best-so-far, continue
            PyMem_Free(prev_pts);
            prev_pts = cand; prev_n = cnt;
            eps += epsilon_increment;
            continue;
        }
    }

    // Return previous (best so far) closed ring
    npy_intp dims[2] = {prev_n+1, 2};
    PyArrayObject* out = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!out) { PyMem_Free(P); PyMem_Free(keep); PyMem_Free(prev_pts); return NULL; }
    double* o = (double*)PyArray_DATA(out);
    for (int i=0;i<prev_n;i++) { o[2*i]=prev_pts[i].x; o[2*i+1]=prev_pts[i].y; }
    o[2*prev_n]=prev_pts[0].x; o[2*prev_n+1]=prev_pts[0].y;
    PyMem_Free(P); PyMem_Free(keep); PyMem_Free(prev_pts);
    return (PyObject*)out;
}

static PyObject* make_line_from_abc(double A, double B, double C) {
    PyObject* args = PyTuple_New(0);
    if (!args) return NULL;
    PyObject* kwargs = Py_BuildValue("{s:d,s:d,s:d}", "A", A, "B", B, "C", C);
    if (!kwargs) { Py_DECREF(args); return NULL; }
    PyObject* obj = PyObject_Call((PyObject*)&LineType, args, kwargs);
    Py_DECREF(args); Py_DECREF(kwargs);
    return obj;
}

// best_iou_quadrilateral(points, max_combinations=300, seed=None)
static PyObject* mod_best_iou_quadrilateral(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"points", "max_combinations", "seed", NULL};
    PyObject* points_obj = NULL; int max_comb = 300; PyObject* seed_obj = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|iO:best_iou_quadrilateral", kwlist, &points_obj, &max_comb, &seed_obj))
        return NULL;

    PyArrayObject* arr = NULL; int rows=0, cols=2;
    if (ensure_double_2d(points_obj, &arr, &rows, &cols) < 0) return NULL;
    Pt* hull = NULL; int nh = 0;
    if (load_points_from_array(arr, &hull, &nh, 1) < 0) { Py_DECREF(arr); return NULL; }
    Py_DECREF(arr);
    if (nh < 4) { PyMem_Free(hull); PyErr_SetString(PyExc_ValueError, "Need at least 4 hull points"); return NULL; }

    // Enforce CCW orientation for hull (clip polygon assumes CCW: inside = left of edge)
    ensure_ccw(hull, nh);

    if (seed_obj && seed_obj != Py_None) {
        long sd = PyLong_AsLong(seed_obj);
        if (!PyErr_Occurred()) srand((unsigned int)sd);
        else PyErr_Clear();
    }

    // Precompute hull area
    double hull_area = polygon_area_abs(hull, nh);

    // Work buffers for intersection
    Pt* work = (Pt*)PyMem_Malloc(sizeof(Pt) * (size_t)(nh + 4));
    Pt* inter = (Pt*)PyMem_Malloc(sizeof(Pt) * (size_t)(nh + 4));
    if (!work || !inter) { PyMem_Free(hull); if(work)PyMem_Free(work); if(inter)PyMem_Free(inter); PyErr_NoMemory(); return NULL; }

    // Count combinations
    long long N = nh;
    long long total = (N*(N-1)*(N-2)*(N-3))/24; // C(N,4)

    // Simple container for sampled combos (avoid duplicates by linear search)
    int sample_all = (max_comb <= 0 || (long long)max_comb >= total);
    int samples = sample_all ? (int)total : max_comb;

    // Allocate best result
    double best_iou = -1.0; Pt best_pts[4];

    if (sample_all) {
        for (int i=0;i<nh-3;i++)
        for (int j=i+1;j<nh-2;j++)
        for (int k=j+1;k<nh-1;k++)
        for (int l=k+1;l<nh;l++) {
            Pt quad[4] = {hull[i], hull[j], hull[k], hull[l]};
            // Ensure quad is CCW for consistent area and clipping
            ensure_ccw(quad, 4);
            double quad_area = polygon_area_abs(quad, 4);
            int in_n = convex_intersection(quad, 4, hull, nh, work, inter);
            double inter_area = (in_n>0) ? polygon_area_abs(inter, in_n) : 0.0;
            double denom = hull_area + quad_area - inter_area;
            double iou = (denom > 0.0) ? (inter_area / denom) : 0.0;
            if (iou > best_iou) { best_iou = iou; best_pts[0]=quad[0]; best_pts[1]=quad[1]; best_pts[2]=quad[2]; best_pts[3]=quad[3]; if (iou >= 1.0) goto done_full; }
        }
    } else {
        // Random sampling without strict dedup
        int picked = 0; int attempts = 0; int max_attempts = samples * 10;
        // store picked combos in an array for dedup (linear scan)
        int (*combos)[4] = (int(*)[4])PyMem_Malloc(sizeof(int)*4*(size_t)samples);
        if (!combos) { PyMem_Free(hull); PyMem_Free(work); PyMem_Free(inter); PyErr_NoMemory(); return NULL; }
        while (picked < samples && attempts < max_attempts) {
            int idx[4];
            // draw 4 distinct indices
            for (;;) {
                idx[0] = rand()%nh;
                idx[1] = rand()%nh;
                idx[2] = rand()%nh;
                idx[3] = rand()%nh;
                // ensure distinct
                int ok=1; for (int a=0;a<4;a++) for (int b=a+1;b<4;b++) if (idx[a]==idx[b]) { ok=0; break; }
                if (!ok) continue;
                // sort ascending
                for (int a=0;a<3;a++) for (int b=a+1;b<4;b++) if (idx[a]>idx[b]) { int t=idx[a]; idx[a]=idx[b]; idx[b]=t; }
                // check duplicate
                ok=1; for (int m=0;m<picked;m++) {
                    if (combos[m][0]==idx[0] && combos[m][1]==idx[1] && combos[m][2]==idx[2] && combos[m][3]==idx[3]) { ok=0; break; }
                }
                if (ok) break;
            }
            combos[picked][0]=idx[0]; combos[picked][1]=idx[1]; combos[picked][2]=idx[2]; combos[picked][3]=idx[3];
            picked++; attempts++;
        }
        for (int m=0;m<picked;m++) {
            int i=combos[m][0], j=combos[m][1], k=combos[m][2], l=combos[m][3];
            Pt quad[4] = {hull[i], hull[j], hull[k], hull[l]};
            ensure_ccw(quad, 4);
            double quad_area = polygon_area_abs(quad, 4);
            int in_n = convex_intersection(quad, 4, hull, nh, work, inter);
            double inter_area = (in_n>0) ? polygon_area_abs(inter, in_n) : 0.0;
            double denom = hull_area + quad_area - inter_area;
            double iou = (denom > 0.0) ? (inter_area / denom) : 0.0;
            if (iou > best_iou) { best_iou = iou; best_pts[0]=quad[0]; best_pts[1]=quad[1]; best_pts[2]=quad[2]; best_pts[3]=quad[3]; if (iou >= 1.0) { PyMem_Free(combos); goto done_full; } }
        }
        PyMem_Free(combos);
    }
done_full:
    {
        PyObject* out = PyTuple_New(4);
        if (!out) { PyMem_Free(hull); PyMem_Free(work); PyMem_Free(inter); return NULL; }
        for (int i=0;i<4;i++) {
            PyObject* pt = Py_BuildValue("(dd)", best_pts[i].x, best_pts[i].y);
            if (!pt) { Py_DECREF(out); PyMem_Free(hull); PyMem_Free(work); PyMem_Free(inter); return NULL; }
            PyTuple_SET_ITEM(out, i, pt);
        }
        PyMem_Free(hull); PyMem_Free(work); PyMem_Free(inter);
        return out;
    }
}

// finetune_quadrilateral(points, initial_vertices)
static PyObject* mod_finetune_quadrilateral(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"points", "initial_vertices", NULL};
    PyObject* points_obj=NULL; PyObject* initv_obj=NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO:finetune_quadrilateral", kwlist, &points_obj, &initv_obj))
        return NULL;
    PyArrayObject *parr=NULL, *ivar=NULL; int pr=0, pc=2, ivr=0, ivc=2;
    if (ensure_double_2d(points_obj, &parr, &pr, &pc) < 0) return NULL;
    if (ensure_double_2d(initv_obj, &ivar, &ivr, &ivc) < 0) { Py_DECREF(parr); return NULL; }
    if (ivr < 4) { Py_DECREF(parr); Py_DECREF(ivar); PyErr_SetString(PyExc_AssertionError, "initial_vertices must be shape (4,2)"); return NULL; }
    // When polygon coords include a duplicate last point equal to first, drop it
    Pt* P=NULL; int NP=0; if (load_points_from_array(parr, &P, &NP, 1) < 0) { Py_DECREF(parr); Py_DECREF(ivar); return NULL; }
    Pt* V=NULL; int NV=0; if (load_points_from_array(ivar, &V, &NV, 0) < 0) { Py_DECREF(parr); Py_DECREF(ivar); PyMem_Free(P); return NULL; }
    Py_DECREF(parr); Py_DECREF(ivar);
    if (NV != 4) { PyMem_Free(P); PyMem_Free(V); PyErr_SetString(PyExc_AssertionError, "initial_vertices must have 4 rows"); return NULL; }

    // Build initial lines L0: V0->V1, L1: V1->V2, L2: V2->V3, L3: V3->V0
    double A[4], B[4], Cc[4], Nn[4];
    for (int i=0;i<4;i++) {
        Pt p0 = V[i]; Pt p1 = V[(i+1)%4];
        double a = p1.y - p0.y;
        double b = p0.x - p1.x;
        double c = p1.x*p0.y - p0.x*p1.y;
        double norm = sqrt(a*a + b*b); if (norm == 0.0) norm = 1.0;
        A[i]=a; B[i]=b; Cc[i]=c; Nn[i]=norm;
    }

    // A small helper lambda-like macro for one assign+fit round
    for (int iter=0; iter<2; ++iter) {
        int counts[4] = {0,0,0,0};
        double sumx[4] = {0,0,0,0}, sumy[4] = {0,0,0,0};
        double sumxx[4] = {0,0,0,0}, sumyy[4] = {0,0,0,0}, sumxy[4] = {0,0,0,0};
        for (int i=0;i<NP;i++) {
            double x = P[i].x, y = P[i].y;
            double bestd = 1e300; int bestk = 0;
            for (int k=0;k<4;k++) {
                double val = fabs(A[k]*x + B[k]*y + Cc[k]) / (Nn[k] > 0.0 ? Nn[k] : 1.0);
                if (val < bestd) { bestd = val; bestk = k; }
            }
            counts[bestk]++;
            sumx[bestk]+=x; sumy[bestk]+=y;
            sumxx[bestk]+=x*x; sumyy[bestk]+=y*y; sumxy[bestk]+=x*y;
        }
        for (int k=0;k<4;k++) {
            if (counts[k] >= 2) {
                double n = (double)counts[k];
                double mx = sumx[k]/n, my = sumy[k]/n;
                double cxx = sumxx[k]/n - mx*mx;
                double cyy = sumyy[k]/n - my*my;
                double cxy = sumxy[k]/n - mx*my;
                double theta = 0.5 * atan2(2.0*cxy, (cxx - cyy));
                double ct = cos(theta), st = sin(theta);
                double a = -st, b = ct;
                double norm = sqrt(a*a + b*b);
                if (!(norm > 0.0 && isfinite(norm))) {
                    a = A[k]; b = B[k]; norm = sqrt(a*a + b*b);
                    if (!(norm > 0.0)) norm = 1.0;
                }
                double c = -(a*mx + b*my);
                if (!isfinite(a) || !isfinite(b) || !isfinite(c)) {
                    a = A[k]; b = B[k]; c = Cc[k]; norm = sqrt(a*a + b*b);
                    if (!(norm > 0.0)) norm = 1.0;
                }
                A[k]=a; B[k]=b; Cc[k]=c; Nn[k]=norm;
            }
        }
    }

    // Build Python Lines and intersection vertices
    PyObject* out_lines = PyTuple_New(4);
    if (!out_lines) { PyMem_Free(P); PyMem_Free(V); return NULL; }
    for (int k=0;k<4;k++) {
        PyObject* L = make_line_from_abc(A[k], B[k], Cc[k]);
        if (!L) { Py_DECREF(out_lines); PyMem_Free(P); PyMem_Free(V); return NULL; }
        PyTuple_SET_ITEM(out_lines, k, L);
    }

    // Intersections of consecutive lines (assuming 4 lines make a polygon)
    Pt pts[4]; double cx=0.0, cy=0.0;
    for (int i=0;i<4;i++) {
        // intersect Li and L(i+1)
        double A1=A[i], B1=B[i], C1=Cc[i];
        double A2=A[(i+1)%4], B2=B[(i+1)%4], C2=Cc[(i+1)%4];
        double det = A1*B2 - A2*B1;
        if (det == 0.0) { pts[i].x = V[i].x; pts[i].y = V[i].y; }
        else {
            pts[i].x = (-C1*B2 + C2*B1)/det;
            pts[i].y = (-A1*C2 + A2*C1)/det;
        }
        cx += pts[i].x; cy += pts[i].y;
    }
    cx /= 4.0; cy /= 4.0;
    // order by angle
    PtAng ord[4];
    for (int i=0;i<4;i++) { ord[i].x = pts[i].x; ord[i].y = pts[i].y; ord[i].ang = atan2(pts[i].y - cy, pts[i].x - cx); }
    qsort(ord, 4, sizeof(PtAng), cmp_ptang);
    PyObject* out_pts = PyTuple_New(4);
    if (!out_pts) { Py_DECREF(out_lines); PyMem_Free(P); PyMem_Free(V); return NULL; }
    for (int i=0;i<4;i++) {
        PyObject* pt = Py_BuildValue("(dd)", ord[i].x, ord[i].y);
        if (!pt) { Py_DECREF(out_lines); Py_DECREF(out_pts); PyMem_Free(P); PyMem_Free(V); return NULL; }
        PyTuple_SET_ITEM(out_pts, i, pt);
    }

    PyObject* ret = PyTuple_New(2);
    if (!ret) { Py_DECREF(out_lines); Py_DECREF(out_pts); PyMem_Free(P); PyMem_Free(V); return NULL; }
    PyTuple_SET_ITEM(ret, 0, out_lines);
    PyTuple_SET_ITEM(ret, 1, out_pts);
    PyMem_Free(P); PyMem_Free(V);
    return ret;
}

// expand_quadrilateral(lines, hull_points)
static PyObject* mod_expand_quadrilateral(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"lines", "hull_points", NULL};
    PyObject* seq=NULL; PyObject* pts_obj=NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO:expand_quadrilateral", kwlist, &seq, &pts_obj))
        return NULL;

    PyObject* fast = PySequence_Fast(seq, "Expected a sequence of Line");
    if (!fast) return NULL;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    if (n != 4) { Py_DECREF(fast); PyErr_SetString(PyExc_AssertionError, "Expected 4 lines"); return NULL; }
    LineObject* Ls[4];
    for (int i=0;i<4;i++) {
        PyObject* obj = PySequence_Fast_GET_ITEM(fast, i);
        if (!PyObject_TypeCheck(obj, &LineType)) { Py_DECREF(fast); PyErr_SetString(PyExc_TypeError, "All items must be Line"); return NULL; }
        Ls[i] = (LineObject*)obj;
    }

    PyArrayObject* arr=NULL; int hr=0, hc=2;
    if (ensure_double_2d(pts_obj, &arr, &hr, &hc) < 0) { Py_DECREF(fast); return NULL; }
    Pt* H=NULL; int NH=0; if (load_points_from_array(arr, &H, &NH, 1) < 0) { Py_DECREF(arr); Py_DECREF(fast); return NULL; }
    Py_DECREF(arr);
    if (NH < 3) { Py_DECREF(fast); PyMem_Free(H); PyErr_SetString(PyExc_ValueError, "Need at least 3 hull points"); return NULL; }

    // Compute centroid of hull
    double cx=0.0, cy=0.0; polygon_centroid(H, NH, &cx, &cy);

    // Create new Lines as copies and move outward as needed
    double A[4], B[4], Cc[4];
    for (int i=0;i<4;i++) { A[i]=Ls[i]->A; B[i]=Ls[i]->B; Cc[i]=Ls[i]->C; }
    for (int i=0;i<4;i++) {
        double norm = sqrt(A[i]*A[i] + B[i]*B[i]); if (norm == 0.0) norm = 1.0;
        double vc = A[i]*cx + B[i]*cy + Cc[i];
        int signc = (vc > 0) ? 1 : ((vc < 0) ? -1 : 0);
        if (signc == 0) signc = 1; // fallback
        double bestd = -1.0; Pt best = {0,0};
        for (int j=0;j<NH;j++) {
            Pt p = H[j];
            double v = A[i]*p.x + B[i]*p.y + Cc[i];
            int s = (v > 0) ? 1 : ((v < 0) ? -1 : 0);
            if (s != 0 && s != signc) {
                double d = fabs(v)/norm;
                if (d > bestd) { bestd = d; best = p; }
            }
        }
        if (bestd > 0.0) {
            Cc[i] = -(A[i]*best.x + B[i]*best.y);
        }
    }

    // Build Python Lines and vertices
    PyObject* out_lines = PyTuple_New(4);
    if (!out_lines) { Py_DECREF(fast); PyMem_Free(H); return NULL; }
    for (int k=0;k<4;k++) {
        PyObject* L = make_line_from_abc(A[k], B[k], Cc[k]);
        if (!L) { Py_DECREF(out_lines); Py_DECREF(fast); PyMem_Free(H); return NULL; }
        PyTuple_SET_ITEM(out_lines, k, L);
    }

    Pt pts[4]; double ccx=0.0, ccy=0.0;
    for (int i=0;i<4;i++) {
        double A1=A[i], B1=B[i], C1=Cc[i];
        double A2=A[(i+1)%4], B2=B[(i+1)%4], C2=Cc[(i+1)%4];
        double det = A1*B2 - A2*B1;
        if (det == 0.0) { pts[i].x = 0.0; pts[i].y = 0.0; }
        else {
            pts[i].x = (-C1*B2 + C2*B1)/det;
            pts[i].y = (-A1*C2 + A2*C1)/det;
        }
        ccx += pts[i].x; ccy += pts[i].y;
    }
    ccx /= 4.0; ccy /= 4.0;
    PtAng ord[4];
    for (int i=0;i<4;i++) { ord[i].x = pts[i].x; ord[i].y = pts[i].y; ord[i].ang = atan2(pts[i].y - ccy, pts[i].x - ccx); }
    qsort(ord, 4, sizeof(PtAng), cmp_ptang);
    PyObject* out_pts = PyTuple_New(4);
    if (!out_pts) { Py_DECREF(out_lines); Py_DECREF(fast); PyMem_Free(H); return NULL; }
    for (int i=0;i<4;i++) {
        PyObject* pt = Py_BuildValue("(dd)", ord[i].x, ord[i].y);
        if (!pt) { Py_DECREF(out_lines); Py_DECREF(out_pts); Py_DECREF(fast); PyMem_Free(H); return NULL; }
        PyTuple_SET_ITEM(out_pts, i, pt);
    }
    PyObject* ret = PyTuple_New(2);
    if (!ret) { Py_DECREF(out_lines); Py_DECREF(out_pts); Py_DECREF(fast); PyMem_Free(H); return NULL; }
    PyTuple_SET_ITEM(ret, 0, out_lines);
    PyTuple_SET_ITEM(ret, 1, out_pts);
    Py_DECREF(fast); PyMem_Free(H);
    return ret;
}

// convex_hull_monotone(points) -> np.ndarray[H+1,2] (closed ring)
typedef struct { double x, y; } Pt2;
static int cmp_pt2(const void* a, const void* b) {
    const Pt2* p = (const Pt2*)a; const Pt2* q = (const Pt2*)b;
    if (p->x < q->x) return -1; if (p->x > q->x) return 1;
    if (p->y < q->y) return -1; if (p->y > q->y) return 1;
    return 0;
}
static double cross_o(Pt2 o, Pt2 a, Pt2 b) {
    return (a.x - o.x)*(b.y - o.y) - (a.y - o.y)*(b.x - o.x);
}
static PyObject* mod_convex_hull_monotone(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"points", NULL};
    PyObject* points_obj=NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O:convex_hull_monotone", kwlist, &points_obj)) return NULL;
    PyArrayObject* arr=NULL; int r=0, c=2;
    if (ensure_double_2d(points_obj, &arr, &r, &c) < 0) return NULL;
    Pt2* pts = (Pt2*)PyMem_Malloc(sizeof(Pt2)*(size_t)r);
    if (!pts) { Py_DECREF(arr); PyErr_NoMemory(); return NULL; }
    // Load points, drop last if duplicates first
    double* p = (double*)PyArray_DATA(arr);
    npy_intp s0 = PyArray_STRIDE(arr,0)/sizeof(double);
    npy_intp s1 = PyArray_STRIDE(arr,1)/sizeof(double);
    int n = r;
    if (r>=2) {
        double x0 = *(p + 0*s0 + 0*s1), y0 = *(p + 0*s0 + 1*s1);
        double xl = *(p + (r-1)*s0 + 0*s1), yl = *(p + (r-1)*s0 + 1*s1);
        if (x0==xl && y0==yl) n = r-1;
    }
    for (int i=0;i<n;i++) { pts[i].x = *(p + i*s0 + 0*s1); pts[i].y = *(p + i*s0 + 1*s1); }
    Py_DECREF(arr);
    if (n == 0) {
        PyMem_Free(pts);
        npy_intp dims[2] = {0,2};
        return (PyObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    }
    // Sort and unique
    qsort(pts, (size_t)n, sizeof(Pt2), cmp_pt2);
    int m = 0; for (int i=1;i<n;i++) if (!(pts[i].x==pts[m].x && pts[i].y==pts[m].y)) pts[++m]=pts[i];
    int uniq = m+1;
    if (uniq == 1) {
        npy_intp dims[2] = {2,2};
        PyArrayObject* out = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
        if (!out) { PyMem_Free(pts); return NULL; }
        double* o=(double*)PyArray_DATA(out);
        o[0]=pts[0].x; o[1]=pts[0].y; o[2]=pts[0].x; o[3]=pts[0].y;
        PyMem_Free(pts); return (PyObject*)out;
    }
    // Build lower and upper
    Pt2* H = (Pt2*)PyMem_Malloc(sizeof(Pt2)*(size_t)(2*uniq));
    if (!H) { PyMem_Free(pts); PyErr_NoMemory(); return NULL; }
    int k=0;
    for (int i=0;i<uniq;i++) {
        while (k>=2 && cross_o(H[k-2], H[k-1], pts[i]) <= 0.0) k--;
        H[k++] = pts[i];
    }
    int t = k+1;
    for (int i=uniq-2;i>=0;i--) {
        while (k>=t && cross_o(H[k-2], H[k-1], pts[i]) <= 0.0) k--;
        H[k++] = pts[i];
    }
    // k is hull size+1 (last duplicates first). Ensure at least 2.
    int hs = (k<2)?2:k;
    npy_intp dims[2] = {hs, 2};
    PyArrayObject* out = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!out) { PyMem_Free(pts); PyMem_Free(H); return NULL; }
    double* o = (double*)PyArray_DATA(out);
    for (int i=0;i<hs;i++) { o[2*i]=H[i].x; o[2*i+1]=H[i].y; }
    PyMem_Free(pts); PyMem_Free(H);
    return (PyObject*)out;
}
