# omega-blas

BLAS in Omega: the open replacement track for layer #6 of the NVIDIA stack (cuBLAS).

- **Parent:** [aien-dev/omega-mv](https://github.com/aien-dev/omega-mv) at `e5b32567a7aff3ce594e3254a622a846a6cd190b` (language, compiler, gates). Exact pins: [PINS](PINS).
- **Binding:** [aien-dev/omega-bridge](https://github.com/aien-dev/omega-bridge) `tools/native`, ABI v2.
- **GPU execution:** [aien-dev/CUDApp](https://github.com/aien-dev/CUDApp) (ADR 0042: Omega owns compute, CUDApp owns GPU execution, omega-bridge owns compatibility).

## What is here

| Path | Role | Status |
|---|---|---|
| `om/blas.om` | the compute unit: f16/bf16/e4m3/e5m2 widening and RNE narrowing, panel packing, the 2x2 f32 gemm macro-kernel, the fused epilogue (alpha, beta, bias by row or column, relu, gelu-tanh), gemv, strided copy, reference argument checks | Omega (OM-2) |
| `src/omegablas.c`, `include/omegablas.h` | **interim** C layer: CBLAS and Fortran 77 entry points, xerbla routing, workspace, cache blocking, batch loops | C, migration pending; the Omega extensions that retire it are not yet written down |
| `python/omegablas` | DLPack binding (numpy, torch, any `__dlpack__` producer) | Python over ctypes |
| `tests/` | `smoke.py` (cblas_sgemm and cblas_sgemv against a float64 numpy reference), `convert.py` (exhaustive f16/bf16/e4m3fn/e5m2 widening and f32 to f16/bf16 RNE narrowing against ml_dtypes; fp8 narrowing not covered), `netlib/run.sh` (netlib sblat2/3 and c_sblat2/3 drivers, SGEMV and SGEMM only) | no fuzz, mutant, A/B or gemm_ex tests yet |

Every floating-point operation runs in Omega. The C layer moves no numbers except the bias copy
into the workspace and the batch pointer arithmetic.

## Routines

- `cblas_sgemm`, `sgemm_`, `cblas_sgemv`, `sgemv_` (standard CBLAS / F77 ABI, LP64).
- `omb_gemm_ex`: A and B in f32, f16, bf16, OCP FP8 e4m3 or e5m2; C in f32, f16 or bf16; f32
  accumulation; fused `C = act(alpha op(A) op(B) + beta C + bias)` with bias per row or column and
  act none, relu or gelu-tanh.
- `omb_hgemm`, `omb_bfgemm`, `omb_gemm_batched`, `omb_gemm_strided_batched`, `omb_widen`, `omb_narrow`.

Numerics: no FMA (one rounding per operation). The bounds the tests enforce: `tests/smoke.py` requires
sgemm max error / max(1, max |reference|) < 1e-5 and sgemv max absolute error at most 1e-4; the netlib
drivers apply their own test-ratio threshold from the input decks. No derived error bound is written down yet.

## Build

```sh
make BRIDGE=/path/to/omega-bridge OMC=/path/to/seeded/omc OMC2=/path/to/om2/omc
python3 tests/smoke.py
```

Limits: each dimension below 1,000,000 (the seeded compiler's loop bound); calls are serialised by
one lock (the bound Omega runtime has one stack).

Licence: Apache-2.0 WITH LLVM-exception.
