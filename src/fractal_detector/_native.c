#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <string.h>

static double *copy_matrix(PyObject *obj, Py_ssize_t *rows, Py_ssize_t *cols) {
    Py_buffer view;
    if (PyObject_GetBuffer(obj, &view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0)
        return NULL;
    if (view.ndim != 2 || view.itemsize != (Py_ssize_t)sizeof(double) || view.format == NULL ||
        strcmp(view.format, "d") != 0 || view.shape[0] <= 0 || view.shape[1] <= 0 ||
        view.shape[0] > PY_SSIZE_T_MAX / (Py_ssize_t)sizeof(double) / view.shape[1] ||
        view.len != view.shape[0] * view.shape[1] * (Py_ssize_t)sizeof(double)) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "expected a nonempty native float64 C-contiguous matrix");
        return NULL;
    }
    double *copy = PyMem_Malloc((size_t)view.len);
    if (copy == NULL) {
        PyBuffer_Release(&view);
        PyErr_NoMemory();
        return NULL;
    }
    *rows = view.shape[0];
    *cols = view.shape[1];
    memcpy(copy, view.buf, (size_t)view.len);
    PyBuffer_Release(&view);
    for (Py_ssize_t i = 0; i < *rows * *cols; i++) {
        if (!isfinite(copy[i]) || fabs(copy[i]) > 1.0) {
            PyMem_Free(copy);
            PyErr_SetString(PyExc_ValueError, "native inputs must be finite and in [-1,1]");
            return NULL;
        }
    }
    return copy;
}

static PyObject *tile_stats(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *obj;
    Py_ssize_t size, rows, cols;
    if (!PyArg_ParseTuple(args, "On", &obj, &size))
        return NULL;
    double *data = copy_matrix(obj, &rows, &cols);
    if (data == NULL)
        return NULL;
    if (size <= 0 || size > rows || size > cols || rows % size || cols % size) {
        PyMem_Free(data);
        PyErr_SetString(PyExc_ValueError, "tile size must be positive and divide both dimensions");
        return NULL;
    }
    Py_ssize_t count = (rows / size) * (cols / size);
    if (count > PY_SSIZE_T_MAX / (3 * (Py_ssize_t)sizeof(double))) {
        PyMem_Free(data);
        return PyErr_NoMemory();
    }
    double *out = PyMem_Malloc((size_t)count * 3 * sizeof(double));
    if (out == NULL) {
        PyMem_Free(data);
        return PyErr_NoMemory();
    }
    Py_BEGIN_ALLOW_THREADS
    Py_ssize_t n = 0;
    for (Py_ssize_t y = 0; y < rows; y += size) {
        for (Py_ssize_t x = 0; x < cols; x += size) {
            double lo = data[y * cols + x], hi = lo, sum = 0.0;
            for (Py_ssize_t dy = 0; dy < size; dy++) {
                for (Py_ssize_t dx = 0; dx < size; dx++) {
                    double value = data[(y + dy) * cols + x + dx];
                    if (value < lo)
                        lo = value;
                    if (value > hi)
                        hi = value;
                    sum += value;
                }
            }
            out[n++] = lo;
            out[n++] = hi;
            out[n++] = sum;
        }
    }
    Py_END_ALLOW_THREADS
    PyMem_Free(data);
    PyObject *result =
        PyBytes_FromStringAndSize((const char *)out, count * 3 * (Py_ssize_t)sizeof(double));
    PyMem_Free(out);
    return result;
}

static void moments(double *data, Py_ssize_t rows, Py_ssize_t cols, double *stats) {
    for (Py_ssize_t r = 0; r < rows; r++) {
        double mean = 0.0, var = 0.0;
        for (Py_ssize_t p = 0; p < cols; p++)
            mean += data[r * cols + p];
        mean /= (double)cols;
        for (Py_ssize_t p = 0; p < cols; p++) {
            data[r * cols + p] -= mean;
            var += data[r * cols + p] * data[r * cols + p];
        }
        stats[2 * r] = mean;
        stats[2 * r + 1] = var / (double)cols;
    }
}

static PyObject *affine_match(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *range_obj, *domain_obj;
    double clip;
    if (!PyArg_ParseTuple(args, "OOd", &range_obj, &domain_obj, &clip))
        return NULL;
    if (!isfinite(clip) || clip <= 0 || clip >= 1) {
        PyErr_SetString(PyExc_ValueError, "contrast clip must be in (0,1)");
        return NULL;
    }
    Py_ssize_t nr, nd, pixels, dpixels;
    double *ranges = copy_matrix(range_obj, &nr, &pixels);
    if (ranges == NULL)
        return NULL;
    double *domains = copy_matrix(domain_obj, &nd, &dpixels);
    if (domains == NULL) {
        PyMem_Free(ranges);
        return NULL;
    }
    if (pixels != dpixels) {
        PyMem_Free(ranges);
        PyMem_Free(domains);
        PyErr_SetString(PyExc_ValueError, "range and domain widths differ");
        return NULL;
    }
    if (nr > PY_SSIZE_T_MAX / (4 * (Py_ssize_t)sizeof(double)) ||
        nd > PY_SSIZE_T_MAX / (2 * (Py_ssize_t)sizeof(double))) {
        PyMem_Free(ranges);
        PyMem_Free(domains);
        return PyErr_NoMemory();
    }
    double *rs = PyMem_Malloc((size_t)nr * 2 * sizeof(double));
    double *ds = PyMem_Malloc((size_t)nd * 2 * sizeof(double));
    double *out = PyMem_Malloc((size_t)nr * 4 * sizeof(double));
    if (rs == NULL || ds == NULL || out == NULL) {
        PyMem_Free(ranges);
        PyMem_Free(domains);
        PyMem_Free(rs);
        PyMem_Free(ds);
        PyMem_Free(out);
        return PyErr_NoMemory();
    }
    Py_BEGIN_ALLOW_THREADS
    moments(ranges, nr, pixels, rs);
    moments(domains, nd, pixels, ds);
    for (Py_ssize_t r = 0; r < nr; r++) {
        double best = HUGE_VAL, best_s = 0, best_o = 0;
        Py_ssize_t best_d = 0;
        for (Py_ssize_t d = 0; d < nd; d++) {
            double cov = 0.0;
            for (Py_ssize_t p = 0; p < pixels; p++)
                cov += ranges[r * pixels + p] * domains[d * pixels + p];
            cov /= (double)pixels;
            double s = ds[2 * d + 1] > 1e-15 ? cov / ds[2 * d + 1] : 0;
            if (s > clip)
                s = clip;
            if (s < -clip)
                s = -clip;
            double err = fmax(0.0, rs[2 * r + 1] + s * s * ds[2 * d + 1] - 2 * s * cov);
            if (err < best) {
                best = err;
                best_s = s;
                best_o = rs[2 * r] - s * ds[2 * d];
                best_d = d;
            }
        }
        out[4 * r] = best;
        out[4 * r + 1] = best_s;
        out[4 * r + 2] = best_o;
        out[4 * r + 3] = (double)best_d;
    }
    Py_END_ALLOW_THREADS
    PyMem_Free(ranges);
    PyMem_Free(domains);
    PyMem_Free(rs);
    PyMem_Free(ds);
    PyObject *result =
        PyBytes_FromStringAndSize((const char *)out, nr * 4 * (Py_ssize_t)sizeof(double));
    PyMem_Free(out);
    return result;
}

static PyMethodDef methods[] = {{"tile_stats", tile_stats, METH_VARARGS, NULL},
                                {"affine_match", affine_match, METH_VARARGS, NULL},
                                {NULL, NULL, 0, NULL}};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_native", NULL, -1, methods, NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit__native(void) { return PyModule_Create(&module); }
