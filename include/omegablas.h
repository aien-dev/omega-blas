/* omegablas.h: the C interface of libomegablas (aien-dev/omega-blas).
 *
 * The arithmetic is the Omega unit om/blas.om, bound through omega-bridge native ABI v2. This
 * header and src/omegablas.c are the INTERIM C orchestration layer (argument mapping, workspace,
 * blocking, batching); see docs/EXTENSIONS.md for the Omega features that retire it.
 *
 * Standard entry points follow the public CBLAS and Fortran 77 BLAS calling conventions (LP64:
 * 32-bit integers). The omb_* entry points are omega-blas extensions.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
#ifndef OMEGABLAS_H
#define OMEGABLAS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };
typedef enum CBLAS_ORDER CBLAS_LAYOUT;

/* CBLAS level 2 and 3 (single precision real) */
void cblas_sgemv(CBLAS_LAYOUT layout, enum CBLAS_TRANSPOSE TransA, int M, int N, float alpha,
                 const float *A, int lda, const float *X, int incX, float beta, float *Y, int incY);
void cblas_sgemm(CBLAS_LAYOUT layout, enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
                 int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb,
                 float beta, float *C, int ldc);
void cblas_xerbla(int p, const char *rout, const char *form, ...);

/* Fortran 77 entries (hidden string lengths last, ignored) */
void sgemv_(const char *trans, const int *m, const int *n, const float *alpha, const float *a,
            const int *lda, const float *x, const int *incx, const float *beta, float *y,
            const int *incy, size_t ltrans);
void sgemm_(const char *transa, const char *transb, const int *m, const int *n, const int *k,
            const float *alpha, const float *a, const int *lda, const float *b, const int *ldb,
            const float *beta, float *c, const int *ldc, size_t lta, size_t ltb);
void xerbla_(const char *srname, const int *info, size_t len);

/* ---- omega-blas extensions ---- */

/* Element types. Products are f32 x f32 with an f32 accumulator; the narrow input formats are
 * widened exactly, and a 16-bit C is written with round-to-nearest-even. */
typedef enum {
    OMB_F32 = 0,
    OMB_F16 = 1,
    OMB_BF16 = 2,
    OMB_E4M3 = 3, /* OCP FP8 E4M3 (finite-only, NaN = S.1111.111) */
    OMB_E5M2 = 4  /* OCP FP8 E5M2 (IEEE-like) */
} omb_type;

/* Fused epilogue: C = act(alpha * op(A) op(B) + beta * C + bias). */
typedef enum { OMB_ACT_NONE = 0, OMB_ACT_RELU = 1, OMB_ACT_GELU_TANH = 2 } omb_act;
typedef enum { OMB_BIAS_NONE = 0, OMB_BIAS_ROW = 1 /* bias[i], length M */,
               OMB_BIAS_COL = 2 /* bias[j], length N */ } omb_bias;

/* Returns 0, or the CBLAS argument number of the first invalid argument (nothing is written).
 * A and B: OMB_F32 .. OMB_E5M2; C: OMB_F32, OMB_F16 or OMB_BF16. */
int omb_gemm_ex(CBLAS_LAYOUT layout, enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
                int M, int N, int K, float alpha, const void *A, omb_type atype, int lda,
                const void *B, omb_type btype, int ldb, float beta, void *C, omb_type ctype,
                int ldc, const float *bias, omb_bias bias_mode, omb_act act);

/* hgemm and bf16 gemm: 16-bit A, B and C, f32 accumulation. */
int omb_hgemm(CBLAS_LAYOUT layout, enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
              int M, int N, int K, float alpha, const uint16_t *A, int lda, const uint16_t *B,
              int ldb, float beta, uint16_t *C, int ldc);
int omb_bfgemm(CBLAS_LAYOUT layout, enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
               int M, int N, int K, float alpha, const uint16_t *A, int lda, const uint16_t *B,
               int ldb, float beta, uint16_t *C, int ldc);

/* Pointer-array batch: count independent problems of one shape. */
int omb_gemm_batched(CBLAS_LAYOUT layout, enum CBLAS_TRANSPOSE TransA,
                     enum CBLAS_TRANSPOSE TransB, int M, int N, int K, float alpha,
                     const void *const *A, omb_type atype, int lda, const void *const *B,
                     omb_type btype, int ldb, float beta, void *const *C, omb_type ctype, int ldc,
                     int count);
/* Strided batch: problem b uses A + b*strideA, B + b*strideB, C + b*strideC (in elements). */
int omb_gemm_strided_batched(CBLAS_LAYOUT layout, enum CBLAS_TRANSPOSE TransA,
                             enum CBLAS_TRANSPOSE TransB, int M, int N, int K, float alpha,
                             const void *A, omb_type atype, int lda, int64_t strideA,
                             const void *B, omb_type btype, int ldb, int64_t strideB, float beta,
                             void *C, omb_type ctype, int ldc, int64_t strideC, int count);

/* Format conversion through the Omega unit (exact widening, RNE narrowing). */
void omb_widen(omb_type src, const void *s, float *d, size_t n);
void omb_narrow(omb_type dst, const float *s, void *d, size_t n);

/* The identity string of the bound Omega unit (source and compiler sha256). */
const char *omb_ident(void);

#ifdef __cplusplus
}
#endif
#endif
