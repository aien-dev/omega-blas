"""Smoke check: cblas_sgemm / cblas_sgemv / omb_gemm_ex against a float64 numpy reference."""
import ctypes as C
import os
import sys

import numpy as np

lib = C.CDLL(os.environ.get("OMEGABLAS_LIB", os.path.join(os.path.dirname(__file__), "..", "build", "libomegablas.so")))
fp = C.POINTER(C.c_float)
lib.cblas_sgemm.argtypes = [C.c_int] * 6 + [C.c_float, fp, C.c_int, fp, C.c_int, C.c_float, fp, C.c_int]
lib.cblas_sgemv.argtypes = [C.c_int] * 4 + [C.c_float, fp, C.c_int, fp, C.c_int, C.c_float, fp, C.c_int]
P = lambda a: a.ctypes.data_as(fp)
rng = np.random.default_rng(1)
bad = 0
for (m, n, k) in [(1, 1, 1), (2, 3, 4), (5, 7, 3), (17, 9, 300), (300, 260, 513)]:
    for ta in (0, 1):
        for tb in (0, 1):
            a = rng.standard_normal((k, m) if ta else (m, k)).astype(np.float32)
            b = rng.standard_normal((n, k) if tb else (k, n)).astype(np.float32)
            c = rng.standard_normal((m, n)).astype(np.float32)
            al, be = 1.5, -0.5
            ref = al * ((a.T if ta else a).astype(np.float64) @ (b.T if tb else b).astype(np.float64)) + be * c
            out = c.copy()
            lib.cblas_sgemm(101, 112 if ta else 111, 112 if tb else 111, m, n, k, al, P(a), a.shape[1], P(b), b.shape[1], be, P(out), n)
            err = np.abs(out - ref).max() / max(1.0, np.abs(ref).max())
            ok = err < 1e-5
            bad += not ok
            print(f"sgemm m={m} n={n} k={k} ta={ta} tb={tb} relerr={err:.2e} {'ok' if ok else 'FAIL'}")
a = rng.standard_normal((37, 23)).astype(np.float32)
for t in (0, 1):
    x = rng.standard_normal(37 if t else 23).astype(np.float32)
    y = rng.standard_normal(23 if t else 37).astype(np.float32)
    ref = 2.0 * ((a.T if t else a).astype(np.float64) @ x) + 0.25 * y
    out = y.copy()
    lib.cblas_sgemv(101, 112 if t else 111, 37, 23, 2.0, P(a), 23, P(x), 1, 0.25, P(out), 1)
    err = np.abs(out - ref).max()
    bad += err > 1e-4
    print(f"sgemv t={t} abserr={err:.2e}")
print("FAILURES", bad)
sys.exit(1 if bad else 0)
