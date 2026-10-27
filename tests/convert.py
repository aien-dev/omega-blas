"""Exhaustive format-conversion check of the Omega converters against ml_dtypes.

widen: every f16 and bf16 bit pattern (65536 each) and every e4m3fn / e5m2 byte (256 each) must
give the exact f32 (NaN for NaN). narrow: every f16 / bf16 pattern round-trips, plus the f32
midpoints between neighbouring 16-bit values (ties to even), their neighbours one f32 ulp either
side, overflow and underflow edges, and 2**20 random f32 bit patterns, all against ml_dtypes RNE.
"""
import os
import sys

import ml_dtypes as md
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import omegablas as ob  # noqa: E402

bad = 0


def same(got, want, what):
    global bad
    g = np.asarray(got).view(np.uint32 if got.dtype.itemsize == 4 else np.uint16)
    w = np.asarray(want).view(g.dtype)
    gn = np.isnan(np.asarray(got, np.float32))
    wn = np.isnan(np.asarray(want, np.float32))
    diff = np.flatnonzero(((g != w) & ~(gn & wn)) | (gn != wn))
    print(f"{what}: {got.size} values, {diff.size} mismatches")
    if diff.size:
        bad += 1
        for i in diff[:5]:
            print(f"   index {i}: got {g.flat[i]:#x} want {w.flat[i]:#x}")


for name, dt, bits in (("f16", np.float16, np.uint16), ("bf16", md.bfloat16, np.uint16),
                       ("e4m3fn", md.float8_e4m3fn, np.uint8), ("e5m2", md.float8_e5m2, np.uint8)):
    src = np.arange(1 << (8 * np.dtype(bits).itemsize), dtype=np.uint64).astype(bits).view(dt)
    same(ob.widen(src), src.astype(np.float32), f"widen {name} (all patterns)")

rng = np.random.default_rng(7)
for name, dt in (("float16", np.float16), ("bfloat16", md.bfloat16)):
    every = np.arange(65536, dtype=np.uint32).astype(np.uint16).view(dt)
    f = every.astype(np.float32)
    fin = f[np.isfinite(f)]
    srt = np.unique(fin)
    mids = ((srt[:-1].astype(np.float64) + srt[1:]) / 2).astype(np.float32)  # exact in f32
    near = np.concatenate([np.nextafter(mids, np.float32(np.inf)), np.nextafter(mids, np.float32(-np.inf))])
    edges = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 65504.0, 65519.99, 65520.0, 6.0e-8, 2.9e-8,
                      3.0e-8, 3.4e38, -3.4e38, 1e-45, 1.17549435e-38], np.float32)
    rnd = rng.integers(0, 1 << 32, 1 << 20, dtype=np.uint64).astype(np.uint32).view(np.float32)
    for label, x in (("all patterns", f), ("ties", mids), ("tie +-1ulp", near), ("edges", edges),
                     ("random f32", rnd)):
        same(ob.narrow(x, name), x.astype(dt), f"narrow {name} {label}")

print("FAILURES", bad)
sys.exit(1 if bad else 0)
