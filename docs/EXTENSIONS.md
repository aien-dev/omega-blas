# Retiring the C layer

`src/omegablas.c` is interim C. All host arithmetic already runs in `om/blas.om`. This file lists
what the C layer still does, why that work is not yet in Omega, and the extension that moves each
piece. Every limit below is a limit of the pinned compiler (PINS), checked as stated. None of them is
a property of the language.

## Limits checked at the pin

With OM-2 omc sha256 `2c82480596bbdc50cda9626866209ef662d5834f278f839ff4ce0250c495029f`, 2026-10-27:

| Probe | Result |
|---|---|
| `fn f` with 6 int parameters | rc 0 |
| `fn f` with 7 int parameters | rc 9 (`omc: line 1: error 9`) |
| `global q int 4` in a unit without `fn main` (function mode, the mode omega-bridge binds) | rc 9 |

## What the C layer does, and the successor for each

| C layer duty | Blocker at the pin | Successor |
|---|---|---|
| CBLAS and F77 entry points (`sgemm_` takes 13 by-reference arguments, `cblas_sgemm` takes 14) | 6-parameter limit; omega-bridge exports its own ABI v2 slice signatures, not arbitrary C signatures | omega-bridge export of a declared foreign signature, generated as checked C from the `.om` declaration |
| `xerbla_` routing (a user can override `xerbla_`) | an Omega unit cannot call a foreign symbol | bridge-generated callback trampoline |
| workspace, the single call lock, the device choice and the route counters | no `global` in function mode | function-mode globals, or an explicit context argument carried by the bridge |
| **GB10 device path**: the device choice, the column-major to row-major mapping, batched and strided calls into CUDApp `omega_gpu_gemm_ex` | an Omega unit cannot call C, and `omega_gpu_gemm_ex` is itself migration-pending C | a unit-to-unit Omega call into `src/omega_gpu_gemm_ex.om`, CUDApp's named successor (CUDApp `mk/gemm-ex.mk`) |

## The GB10 device path is migration-pending C

The device code in `src/omegablas.c` sits inside `#ifdef OMB_GB10`. The build switch is `GB10`
(Makefile), and its default is 0. The default build compiles none of this code and does not link
CUDApp. When `OMEGABLAS_DEVICE=gb10` is set on that build, the library aborts and says it was built
without the path. The device code adds no arithmetic: every number on this path is computed by
CUDApp. It is retired by the last row of the table above. When CUDApp's gemm_ex `.om` unit exists,
the dispatch, which is pure index arithmetic plus one call, moves into `om/blas.om`.

The test-only hook (`tests/gb10/hooks.c`) is C for the same reason. CUDApp's backend hook is a C
function pointer. The hook is linked only into the test drivers, never into the library.
