# Tolerance

What the tests enforce, and the bound each enforced number rests on. All arithmetic was done with
the Wolfram Language evaluator. Each query is quoted as `query -> result`.

## The netlib test ratio

The netlib drivers (`tests/netlib/run.sh`: sblat2, sblat3, c_sblat2, c_sblat3, LAPACK 3.12.1)
recompute every result as a fp32 reference `CT` (SMMCH / SMVCH: sequential fp32 sum over k, then
`ALPHA*CT + BETA*C`). They also form `G = |alpha| sum_k |a_k||b_k| + |beta| |c|` and report
`ratio = |CT - CC| / (EPS * G)` with `EPS = 2^-23`. A result is FATAL when
`ERR * sqrt(EPS) >= 1`, and a routine is SUSPECT when its largest ratio reaches the deck's
`THRESHOLD VALUE` (16.0 in all four decks). The decks use `n, m, k` in {0, 1, 2, 3, 5, 9}, so the
inner length satisfies `k <= 9`. The gate is the stock decks with THRESH 16.0. `run.sh` refuses to
run if a deck's threshold line is not the expected value. `THRESH=<x>` replaces 16.0 only for a
measurement run. That run prints every routine's largest ratio and is not a gate.

Write `u = 2^-24` (fp32 unit roundoff) and `gamma_n = n u / (1 - n u)`. The fp32 reference has
`|CT - exact| <= gamma_(k+2) G`: k products and k-1 additions in the sum, then the alpha and beta
roundings.

## Host path

`om/blas.om` rounds once per operation and uses no FMA. It meets the same `gamma_(k+2) G` bound in
any summation order. The two errors add, so the ratio is at most `2 gamma_(k+2) / EPS`:
`N[2 gam[9+2]/2^-23]` with `gam[n_] := n 2^-24/(1 - n 2^-24)` `-> 11.0000072` for `k = 9`. That is
below 16.

Measured (THRESH 0.0 measurement run, 2026-10-27, receipt `receipts/netlib-gb10-20261027.md`):
largest ratio SGEMV 1.48 (cblas_sgemv row-major 1.51), SGEMM 0.00. The host k-loop for these shapes
matches the reference's order exactly.

## GB10 path (fp32 through bf16x3)

The device path calls CUDApp `omega_gpu_gemm_ex` with compute DEFAULT. For f32 inputs that is the
bf16x3 route, which is not bit-exact. Its mechanism, from CUDApp `src/omega_gpu_gemm_ex.c` at the
pin in [PINS](../PINS):

1. Every f32 value splits exactly into `hi + mid + lo` bf16 planes by truncation:
   `|mid| < 2^-7 |x|` and `|lo| < 2^-14 |x|`.
2. One BF16 tensor-core call computes the nine plane products, smallest first:
   (lo,lo) (lo,mid) (mid,lo) (lo,hi) (mid,mid) (hi,lo) (mid,hi) (hi,mid) (hi,hi). Their magnitude
   weights relative to `|a||b|` are 2^-28, 2^-21, 2^-21, 2^-14, 2^-14, 2^-14, 2^-7, 2^-7, 1. Each
   plane pair is a segment of `s = ceil(k/16)` HMMA steps, and the nine segments sum exactly to
   the true product.
3. Every HMMA step follows the chip-exact model of CUDApp `tests/bw_hmma_model.h`
   (`{16, 2, RZ, nominal 2}`, evidence OWN30B-T3-HMMA-MODEL). The step adds the accumulator and 16
   exact products. Each term is truncated toward zero to a multiple of `q = 2^(emax-26)`, where
   `emax` is the largest nominal exponent. The terms are summed exactly and the sum is converted to
   f32 by RZ.
4. The epilogue `alpha*acc + beta*c` is formed in double and stored to f32 with one rounding.

The bound for one step: the term with exponent `emax` has magnitude at least `2^(emax-1)` and is
exact on the grid. The other 16 terms each lose less than `q`, so they lose at most
`16 q <= 2^-21 T` in total, where `T` is the sum of the step's term magnitudes. The RZ conversion
loses less than `2^-23 T`. The step error is therefore at most `5 eps T` with `eps = 2^-23`. In
segment j, `T` is at most `C_j S`, where `S = sum_k |a_k||b_k|` and `C_j` is the running sum of the
weights. Summing over all steps:

    |CC - exact| <= d + (2^-24 + 2^-53) (|v| + d) + 2^-53 |v|,   d = 5 eps s (sum_j C_j) |alpha| S

where `v = alpha*exact + beta*c` and `|v| <= G`.

`Total[Accumulate[{2^-28,2^-21,2^-21,2^-14,2^-14,2^-14,2^-7,2^-7,1}]] -> 279168905/268435456`
(1.03999), and `N[5*279168905/268435456] -> 5.19993`, so the device term contributes `5.19993 s`
ratio units.

Netlib ratio bound, device result against the fp32 reference, including the second-order terms:

    ratio <= gamma_(k+2)/EPS + 5 s (sum_j C_j) + 1/2 + O(2^-29)

- `full[9, 1] -> 11.19992999` (k = 9, the deck maximum). Below THRESH 16, so the stock decks are
  a sound gate for this path: a correct device result cannot be SUSPECT, and a SUSPECT or FATAL is
  a defect.
- `Reduce[full[k, Ceiling[k/16]] < 16 && 1 <= k <= 1808, k] -> k = 1 .. 16`. At these decks'
  threshold the argument holds only for `k <= 16`. Above that, the HMMA truncation term grows with
  the number of steps (k = 17: 20.40; k = 1024: 846.3). A caller with large k compares against a
  k-scaled tolerance such as CUDApp's own predeclared check
  (`(S+4) 2^-22 sum|ab|`, `S = 9 ceil(k/16) + parts`), not a fixed 16.
- k = 0 and alpha = 0 never reach the tensor core. The device computes `beta*c` in double, rounded
  once.

Measured on the GB10 (THRESH 0.0 measurement run, 2026-10-27, `receipts/netlib-gb10-20261027.md`):
largest ratio SGEMV 1.61, SGEMM 1.87. Those are 0.144 and 0.167 of the 11.2 bound. The host model
backend (`tests/gb10/hooks.c`, `OMB_TEST_BACKEND=model`) reports the same 1.61 and 1.87.

Inputs this argument does not cover: underflow (plane products below the f32 normal range, where
the relative step argument does not hold; [INFERENCE] the netlib generators' small rationals stay
far above it), and non-finite inputs. CUDApp recomputes the rows and
columns with non-finite inputs on the host; the route report counts them as `gb10_host_fixed`.

## tests/gb10/batched.c limits

`batched.c` compares against a double reference. The reference is exact to far below `2^-23 G`, so
the reference's own `gamma` term drops out and each route gets a tighter limit. Every case has
`K <= 16`, so `s = 1`:

- device route: `5 c + (2^-24 + 2^-53)(1/EPS + 5 c) + 2^-53/EPS` with `c` = 279168905/268435456
  `-> 5.69992638`; the limit is 5.70.
- host route (bias adds one rounding): `gamma_(16+3)/EPS -> 9.50001076`; the limit is 9.51.

`tests/smoke.py` keeps its own fixed host checks (relative 1e-5 for sgemm, absolute 1e-4 for
sgemv). Those limits are not derived for the device path. Its k = 513 case lies beyond the
`k <= 16` range above. It has not been run on the device path.
