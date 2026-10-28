# omega-blas

BLAS in Omega: the open replacement track for layer #6 of the NVIDIA stack (cuBLAS).

- **Parent:** [aien-dev/omega-mv](https://github.com/aien-dev/omega-mv) at `e5b32567a7aff3ce594e3254a622a846a6cd190b` (language, compiler, gates). Exact pins: [PINS](PINS).
- **Binding:** [aien-dev/omega-bridge](https://github.com/aien-dev/omega-bridge) `tools/native`, ABI v2.
- **GPU execution:** [aien-dev/CUDApp](https://github.com/aien-dev/CUDApp) (ADR 0042: Omega owns compute, CUDApp owns GPU execution, omega-bridge owns compatibility).

## What is here

| Path | Role | Status |
|---|---|---|
| `om/blas.om` | the compute unit: f16/bf16/e4m3/e5m2 widening and RNE narrowing, panel packing, the 2x2 f32 gemm macro-kernel, the fused epilogue (alpha, beta, bias by row or column, relu, gelu-tanh), gemv, strided copy, reference argument checks | Omega (OM-2) |
| `src/omegablas.c`, `include/omegablas.h` | **interim** C layer: CBLAS and Fortran 77 entry points, xerbla routing, workspace, cache blocking, batch loops | C, retired by the extensions in [docs/EXTENSIONS.md](docs/EXTENSIONS.md) |
| `python/omegablas` | DLPack binding (numpy, torch, any `__dlpack__` producer) | Python over ctypes |
| `tests/` | netlib BLAS/CBLAS test drivers, fuzz vs numpy, mutant control, A/B bench | |

Every floating-point operation runs in Omega. The C layer moves no numbers except the bias copy
into the workspace and the batch pointer arithmetic.

## Routines

- `cblas_sgemm`, `sgemm_`, `cblas_sgemv`, `sgemv_` (standard CBLAS / F77 ABI, LP64).
- `omb_gemm_ex`: A and B in f32, f16, bf16, OCP FP8 e4m3 or e5m2; C in f32, f16 or bf16; f32
  accumulation; fused `C = act(alpha op(A) op(B) + beta C + bias)` with bias per row or column and
  act none, relu or gelu-tanh.
- `omb_hgemm`, `omb_bfgemm`, `omb_gemm_batched`, `omb_gemm_strided_batched`, `omb_widen`, `omb_narrow`.

Numerics: no FMA (one rounding per operation); see [docs/TOLERANCE.md](docs/TOLERANCE.md) for the
bound the tests enforce.

## Build

```sh
make BRIDGE=/path/to/omega-bridge OMC=/path/to/seeded/omc OMC2=/path/to/om2/omc
python3 tests/smoke.py
```

Limits: each dimension below 1,000,000 (the seeded compiler's loop bound); calls are serialised by
one lock (the bound Omega runtime has one stack).

Licence: Apache-2.0 WITH LLVM-exception.

## Human-Directed Agentic Engineering

Development follows Human-Directed Agentic Engineering (HDAE): humans define goals,
architecture, constraints, acceptance criteria, and stopping points; agents implement,
execute checks, diagnose failures, and revise within those boundaries. Human review
retains responsibility for claims, acceptance, and release decisions.

For the documented method and a systems-software case study, see Gregory M. Kurtzer,
[*Ginux: Building an Independent Kernel That Runs Linux Userspace through
Human-Directed Agentic Engineering* (2026)](https://doi.org/10.5281/zenodo.23682548).
