"""omegablas: Python binding of libomegablas (aien-dev/omega-blas) over DLPack.

Operands are any object with ``__dlpack__`` (numpy, torch, jax, cupy-on-host, ...) on the CPU
device, or a numpy array (which also covers the ml_dtypes bfloat16 / float8 dtypes that numpy
cannot export through DLPack). Nothing is copied: the library reads the producer's memory through
the DLPack pointer, shape and strides, and writes C in place.

    import numpy as np, omegablas as ob
    c = ob.gemm(a, b, alpha=1.0, bias=v, bias_axis="col", act="gelu_tanh")

Element types: float32, float16, bfloat16, float8_e4m3fn, float8_e5m2 for A and B; float32,
float16, bfloat16 for C. Accumulation is f32 (see docs/TOLERANCE.md).
"""
import ctypes as _C
import os as _os

__all__ = ["gemm", "gemv", "widen", "narrow", "ident", "lib", "Tensor", "tensor"]

F32, F16, BF16, E4M3, E5M2 = 0, 1, 2, 3, 4
_ROW, _NT, _T = 101, 111, 112
_ACT = {"none": 0, None: 0, "relu": 1, "gelu_tanh": 2}
_BIAS = {None: 0, "row": 1, "col": 2}
_SIZE = {F32: 4, F16: 2, BF16: 2, E4M3: 1, E5M2: 1}
# DLPack 1.1 DLDataTypeCode: (code, bits) -> omb_type
_DL = {(2, 32): F32, (2, 16): F16, (4, 16): BF16, (10, 8): E4M3, (12, 8): E5M2}
_NP = {"float32": F32, "float16": F16, "bfloat16": BF16, "float8_e4m3fn": E4M3, "float8_e5m2": E5M2}


def _load():
    p = _os.environ.get("OMEGABLAS_LIB")
    if not p:
        here = _os.path.dirname(_os.path.abspath(__file__))
        for cand in (_os.path.join(here, "libomegablas.so"),
                     _os.path.join(here, "..", "..", "build", "libomegablas.so")):
            if _os.path.exists(cand):
                p = cand
                break
    if not p:
        raise OSError("libomegablas.so not found: build it (make) or set OMEGABLAS_LIB")
    lib = _C.CDLL(p)
    i, f, v, i64 = _C.c_int, _C.c_float, _C.c_void_p, _C.c_int64
    lib.omb_gemm_ex.argtypes = [i, i, i, i, i, i, f, v, i, i, v, i, i, f, v, i, i, v, i, i]
    lib.omb_gemm_ex.restype = i
    lib.omb_gemm_strided_batched.argtypes = [i, i, i, i, i, i, f, v, i, i, i64, v, i, i, i64, f,
                                             v, i, i, i64, i]
    lib.omb_gemm_strided_batched.restype = i
    lib.omb_gemm_batched.argtypes = [i, i, i, i, i, i, f, v, i, i, v, i, i, f, v, i, i, i]
    lib.omb_gemm_batched.restype = i
    lib.cblas_sgemv.argtypes = [i, i, i, i, f, v, i, v, i, f, v, i]
    lib.cblas_sgemv.restype = None
    lib.cblas_sgemm.argtypes = [i, i, i, i, i, i, f, v, i, v, i, f, v, i]
    lib.cblas_sgemm.restype = None
    lib.omb_widen.argtypes = [i, v, v, _C.c_size_t]
    lib.omb_narrow.argtypes = [i, v, v, _C.c_size_t]
    lib.omb_ident.restype = _C.c_char_p
    return lib


lib = _load()


class _DLDevice(_C.Structure):
    _fields_ = [("device_type", _C.c_int32), ("device_id", _C.c_int32)]


class _DLDataType(_C.Structure):
    _fields_ = [("code", _C.c_uint8), ("bits", _C.c_uint8), ("lanes", _C.c_uint16)]


class _DLTensor(_C.Structure):
    _fields_ = [("data", _C.c_void_p), ("device", _DLDevice), ("ndim", _C.c_int32),
                ("dtype", _DLDataType), ("shape", _C.POINTER(_C.c_int64)),
                ("strides", _C.POINTER(_C.c_int64)), ("byte_offset", _C.c_uint64)]


class _DLManagedTensor(_C.Structure):
    _fields_ = [("dl_tensor", _DLTensor), ("manager_ctx", _C.c_void_p), ("deleter", _C.c_void_p)]


_capsule_ptr = _C.pythonapi.PyCapsule_GetPointer
_capsule_ptr.restype = _C.c_void_p
_capsule_ptr.argtypes = [_C.py_object, _C.c_char_p]


class Tensor:
    """A borrowed view: address of element 0, omb_type, shape and strides (in elements).

    ``keep`` holds the producer object and its DLPack capsule (unconsumed, so the producer's
    capsule destructor releases it) for the life of the view."""
    __slots__ = ("ptr", "dtype", "shape", "strides", "keep")

    def __init__(self, ptr, dtype, shape, strides, keep):
        self.ptr, self.dtype, self.shape, self.strides, self.keep = ptr, dtype, shape, strides, keep


def tensor(x):
    """Borrow x through DLPack (CPU device), or through the numpy array interface."""
    if isinstance(x, Tensor):
        return x
    try:
        import numpy as np
        if isinstance(x, np.ndarray) and x.dtype.name in _NP and x.dtype.name not in ("float32", "float16"):
            ai = x.__array_interface__
            st = tuple(s // x.itemsize for s in x.strides)
            if any(s * x.itemsize != b for s, b in zip(st, x.strides)):
                raise ValueError("omegablas: strides must be whole elements")
            return Tensor(ai["data"][0], _NP[x.dtype.name], tuple(x.shape), st, x)
    except ImportError:
        pass
    if not hasattr(x, "__dlpack__"):
        raise TypeError("omegablas: operand has no __dlpack__ and is not a numpy array")
    cap = x.__dlpack__()
    mt = _DLManagedTensor.from_address(_capsule_ptr(cap, b"dltensor"))
    t = mt.dl_tensor
    if t.device.device_type != 1:
        raise ValueError(f"omegablas: device type {t.device.device_type} is not kDLCPU "
                         "(GPU operands go through CUDApp; not wired in this binding)")
    key = (t.dtype.code, t.dtype.bits)
    if t.dtype.lanes != 1 or key not in _DL:
        raise TypeError(f"omegablas: unsupported DLPack dtype code={t.dtype.code} bits={t.dtype.bits}"
                        f" lanes={t.dtype.lanes}")
    nd = t.ndim
    shape = tuple(t.shape[i] for i in range(nd))
    if t.strides:
        strides = tuple(t.strides[i] for i in range(nd))
    else:
        strides, acc = [], 1
        for d in reversed(shape):
            strides.append(acc)
            acc *= max(d, 1)
        strides = tuple(reversed(strides))
    return Tensor((t.data or 0) + t.byte_offset, _DL[key], shape, strides, (x, cap))


def _mat(t, name):
    """(trans, ld) of a 2-D view for a row-major call; strides of length-1 axes are free."""
    r, c = t.shape[-2], t.shape[-1]
    s0, s1 = t.strides[-2], t.strides[-1]
    if c == 1 or s1 == 1:
        if r == 1 or s0 >= max(c, 1):
            return _NT, max(s0 if r > 1 else c, c, 1)
    if r == 1 or s0 == 1:
        if c == 1 or s1 >= max(r, 1):
            return _T, max(s1 if c > 1 else r, r, 1)
    raise ValueError(f"omegablas: {name} needs one unit stride and a leading stride >= its extent "
                     f"(shape {t.shape[-2:]}, strides {t.strides[-2:]})")


def _out(c, shape, out_dtype):
    if c is not None:
        return c, tensor(c)
    import numpy as np
    if out_dtype == "bfloat16":
        import ml_dtypes
        dt = ml_dtypes.bfloat16
    else:
        dt = np.dtype(out_dtype)
    arr = np.zeros(shape, dtype=dt)
    return arr, tensor(arr)


def gemm(a, b, c=None, *, alpha=1.0, beta=0.0, bias=None, bias_axis=None, act="none",
         out_dtype="float32"):
    """C = act(alpha * a @ b + beta * C + bias); a (M,K) or (batch,M,K), b (K,N) or (batch,K,N).

    bias_axis "row" adds bias[i] (length M), "col" adds bias[j] (length N). C is written in place
    when given (it must have a unit last stride); otherwise a new numpy array is returned."""
    ta_, tb_ = tensor(a), tensor(b)
    if len(ta_.shape) != len(tb_.shape) or len(ta_.shape) not in (2, 3):
        raise ValueError("omegablas.gemm: a and b must both be 2-D or both 3-D")
    m, k = ta_.shape[-2:]
    k2, n = tb_.shape[-2:]
    if k != k2:
        raise ValueError(f"omegablas.gemm: inner dimensions {k} and {k2} differ")
    batched = len(ta_.shape) == 3
    nb = ta_.shape[0] if batched else 1
    if batched and tb_.shape[0] != nb:
        raise ValueError("omegablas.gemm: batch sizes differ")
    res, tc = _out(c, (nb, m, n) if batched else (m, n), out_dtype)
    if tuple(tc.shape) != ((nb, m, n) if batched else (m, n)):
        raise ValueError(f"omegablas.gemm: C has shape {tc.shape}")
    if tc.dtype not in (F32, F16, BF16):
        raise TypeError("omegablas.gemm: C must be float32, float16 or bfloat16")
    if n > 1 and tc.strides[-1] != 1:
        raise ValueError("omegablas.gemm: C needs a unit last stride")
    ldc = max(tc.strides[-2] if m > 1 else n, n, 1)
    tra, lda = _mat(ta_, "a")
    trb, ldb = _mat(tb_, "b")
    bptr, bmode = None, _BIAS[bias_axis]
    if bias is not None:
        tv = tensor(bias)
        want = m if bias_axis == "row" else n
        if bias_axis is None or tv.dtype != F32 or tv.shape != (want,) or (want > 1 and tv.strides[0] != 1):
            raise ValueError(f"omegablas.gemm: bias must be contiguous float32 of length {want} "
                             "with bias_axis 'row' or 'col'")
        bptr = tv.ptr
    elif bmode:
        raise ValueError("omegablas.gemm: bias_axis given without bias")
    av = _ACT[act]
    if batched and not bptr and not av:
        rc = lib.omb_gemm_strided_batched(_ROW, tra, trb, m, n, k, alpha, ta_.ptr, ta_.dtype, lda,
                                          ta_.strides[0], tb_.ptr, tb_.dtype, ldb, tb_.strides[0],
                                          beta, tc.ptr, tc.dtype, ldc, tc.strides[0], nb)
    else:
        rc = 0
        for i in range(nb):
            oa = ta_.strides[0] * i * _SIZE[ta_.dtype] if batched else 0
            ob = tb_.strides[0] * i * _SIZE[tb_.dtype] if batched else 0
            oc = tc.strides[0] * i * _SIZE[tc.dtype] if batched else 0
            rc = lib.omb_gemm_ex(_ROW, tra, trb, m, n, k, alpha, (ta_.ptr or 0) + oa, ta_.dtype, lda,
                                 (tb_.ptr or 0) + ob, tb_.dtype, ldb, beta, (tc.ptr or 0) + oc, tc.dtype,
                                 ldc, bptr, bmode, av)
            if rc:
                break
    if rc:
        raise ValueError(f"omegablas.gemm: library rejected argument {rc}")
    return res


def gemv(a, x, y=None, *, alpha=1.0, beta=0.0, trans=False):
    """y = alpha * op(a) @ x + beta * y in float32 (cblas_sgemv); x and y may have any stride."""
    ta_, tx = tensor(a), tensor(x)
    if len(ta_.shape) != 2 or len(tx.shape) != 1 or ta_.dtype != F32 or tx.dtype != F32:
        raise ValueError("omegablas.gemv: a must be 2-D float32 and x 1-D float32")
    m, n = ta_.shape
    lx, ly = (m, n) if trans else (n, m)
    if tx.shape[0] != lx:
        raise ValueError(f"omegablas.gemv: x has length {tx.shape[0]}, expected {lx}")
    res, ty = _out(y, (ly,), "float32")
    if ty.shape != (ly,) or ty.dtype != F32:
        raise ValueError(f"omegablas.gemv: y must be float32 of length {ly}")
    tr, lda = _mat(ta_, "a")
    if tr == _T:  # a stored as a^T in row-major: swap the operation instead
        trans = not trans
        m, n = n, m
    ix = tx.strides[0] if lx > 1 else 1
    iy = ty.strides[0] if ly > 1 else 1
    if ix == 0 or iy == 0:
        raise ValueError("omegablas.gemv: zero stride")
    # BLAS addresses a negative increment from the lowest element
    px = tx.ptr + (min(ix, 0) * (lx - 1)) * 4
    py = ty.ptr + (min(iy, 0) * (ly - 1)) * 4
    lib.cblas_sgemv(_ROW, _T if trans else _NT, m, n, alpha, ta_.ptr, lda, px, ix, beta, py, iy)
    return res


def widen(x):
    """float32 copy of a contiguous float16/bfloat16/float8 array (exact)."""
    import numpy as np
    t = tensor(x)
    n = 1
    for d in t.shape:
        n *= d
    out = np.empty(t.shape, np.float32)
    lib.omb_widen(t.dtype, t.ptr, out.ctypes.data, n)
    return out


def narrow(x, dtype):
    """Round a contiguous float32 array to float16 or bfloat16 (round to nearest even)."""
    import numpy as np
    src = np.ascontiguousarray(x, np.float32)
    res, t = _out(None, src.shape, dtype)
    lib.omb_narrow(t.dtype, src.ctypes.data, t.ptr, src.size)
    return res


def ident():
    """Identity of the bound Omega unit (om/blas.om source and compiler sha256)."""
    return lib.omb_ident().decode()
