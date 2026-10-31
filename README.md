# omega-blas

BLAS in Omega: the open replacement track for layer #6 of the NVIDIA stack (cuBLAS).

- **Parent:** [aien-dev/omega-mv](https://github.com/aien-dev/omega-mv) at `e5b32567a7aff3ce594e3254a622a846a6cd190b` (language, compiler, gates). Exact pins: [PINS](PINS).
- **Binding:** [aien-dev/omega-bridge](https://github.com/aien-dev/omega-bridge) `tools/native`, ABI v2.
- **GPU execution:** [aien-dev/CUDApp](https://github.com/aien-dev/CUDApp) (ADR 0042: Omega owns compute, CUDApp owns GPU execution, omega-bridge owns compatibility).

## What is here

| Path | Role | Status |
|---|---|---|
| `om/blas.om` | the compute unit: f16/bf16/e4m3/e5m2 widening and RNE narrowing, panel packing, the 2x2 f32 gemm macro-kernel, the fused epilogue (alpha, beta, bias by row or column, relu, gelu-tanh), gemv, strided copy, reference argument checks | Omega (OM-2) |
| `src/omegablas.c`, `include/omegablas.h` | **interim** C layer: CBLAS and Fortran 77 entry points, xerbla routing, workspace, cache blocking, batch loops, and (built only with `GB10=1`) the GB10 device dispatch | C, retired by the extensions in [docs/EXTENSIONS.md](docs/EXTENSIONS.md) |
| `python/omegablas` | DLPack binding (numpy, torch, any `__dlpack__` producer) | Python over ctypes |
| `tests/smoke.py`, `tests/convert.py` | numpy checks of cblas_sgemm and cblas_sgemv; exhaustive format-conversion check against ml_dtypes | Python |
| `tests/netlib/` | `run.sh`: the unmodified netlib BLAS/CBLAS SGEMV/SGEMM test drivers (sblat2, sblat3, c_sblat2, c_sblat3) against a build, on the host or the GB10 path; `fetch.sh`: the pinned LAPACK source | shell |
| `tests/gb10/` | `batched.c` + `run.sh`: strided, pointer-array and single products, route predictions and untouched padding, on either device; `hooks.c`: test-only HMMA-model backend and CUDApp mutants | C test drivers (migration-pending, docs/EXTENSIONS.md) |
| `receipts/` | dated run receipts with their logs | |

On the host path every floating-point operation runs in Omega; the C layer moves no numbers except
the bias copy into the workspace and the batch pointer arithmetic. On the GB10 path the f32
products run in CUDApp (see below), not in Omega.

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

## GB10 device path

Status: EXERCISED on the GB10 chip (netlib SGEMV/SGEMM, 2026-10-27,
[receipts/netlib-gb10-20261027.md](receipts/netlib-gb10-20261027.md)). Not QUALIFIED. It depends on
CUDApp `omega_gpu_gemm_ex` (aien-dev/CUDApp#139, EXPERIMENTAL, pinned in [PINS](PINS)), which is
not merged yet.

```sh
make GB10=1 CUDAPP=/path/to/CUDApp BRIDGE=... OMC=... OMC2=...   # builds into build-gb10/
OMEGABLAS_DEVICE=gb10 ./your_program                              # unset or host: the Omega host path
```

- Scope: f32 A, B and C with no bias and no activation, through `sgemm_`, `cblas_sgemm`, `sgemv_`,
  `cblas_sgemv`, `omb_gemm_ex`, `omb_gemm_batched` and `omb_gemm_strided_batched`. A strided batch
  with disjoint C is one `omega_gpu_gemm_ex` call. Every other call stays on the host path and is
  counted as such. `omb_set_device` / `omb_get_device` switch at run time.
- Numerics: CUDApp's bf16x3 route on the tensor cores, not bit-exact fp32. The derived netlib
  ratio bound is 11.2 for the decks' k <= 9 ([docs/TOLERANCE.md](docs/TOLERANCE.md)).
- No silent fallback: a device error aborts the process with the gemm_ex code. So does
  `OMEGABLAS_DEVICE=gb10` on a library built without `GB10=1`, and any value other than `host` or
  `gb10`.
- `omb_get_route_counts` and `OMEGABLAS_ROUTE_REPORT=1` (one stderr line at unload) report host
  products, gemm_ex calls, tensor-core calls, and outputs CUDApp recomputed on the host, plus the
  CUDApp revision linked in.
- Tests: `OMB_BUILD=build-gb10 OMEGABLAS_DEVICE=gb10 LAPACK=... sh tests/netlib/run.sh` and
  `OMEGABLAS_DEVICE=gb10 sh tests/gb10/run.sh`. With `GB10_HOOKS=/path/to/CUDApp
  OMB_TEST_BACKEND=model`, both run the device glue on CUDApp's chip-exact HMMA model without a
  GPU.

Licence: Apache-2.0 WITH LLVM-exception.

## Human-Directed Agentic Engineering

Development follows Human-Directed Agentic Engineering (HDAE): humans define goals,
architecture, constraints, acceptance criteria, and stopping points; agents implement,
execute checks, diagnose failures, and revise within those boundaries. Human review
retains responsibility for claims, acceptance, and release decisions.

For the documented method and a systems-software case study, see Gregory M. Kurtzer,
[*Ginux: Building an Independent Kernel That Runs Linux Userspace through
Human-Directed Agentic Engineering* (2026)](https://doi.org/10.5281/zenodo.23682548).
