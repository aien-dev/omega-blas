/* omegablas.c: INTERIM C orchestration for libomegablas (aien-dev/omega-blas).
 *
 * Every floating-point operation, format conversion, argument check and packed-panel move is done
 * by the Omega unit om/blas.om (bound by omega-bridge tools/native into omega_native.c). This file
 * only maps the CBLAS / Fortran calling conventions onto that unit: argument translation, xerbla
 * routing, workspace allocation, cache blocking and batch loops. docs/EXTENSIONS.md lists the Omega
 * features (more than 6 parameter slots, f32 scalar parameters, a C-callable variadic ABI, global
 * state) that let this layer move into Omega.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
#include "omegablas.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "omega_api.h"
#ifdef OMB_GB10
#include "omega_gpu_gemm_ex.h"
#include "omega_gpu_session.h"
#endif

/* The CBLAS reference test drivers read and write these two globals (c_xerbla.c). */
int RowMajorStrg = 0;
int CBLAS_CallFromC = 0;

/* The bound Omega runtime is one per library (one stack, one reentry flag): calls are serialised. */
static pthread_mutex_t omb_lock = PTHREAD_MUTEX_INITIALIZER;

/* Cache blocks (rows of op(A), columns of op(B), depth). Each keeps every workspace offset far
 * below 2^32 (the Omega kernel carries panel offsets as uint). */
enum { MC = 256, NC = 256, KC = 256 };

static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

static int is16(omb_type t) { return t == OMB_F16 || t == OMB_BF16; }
static int is8(omb_type t) { return t == OMB_E4M3 || t == OMB_E5M2; }

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) { fputs("omegablas: out of memory\n", stderr); abort(); }
    return p;
}

/* Element count a view must cover for an r x c column-major block with leading dimension ld. */
static size_t span(int64_t r, int64_t c, int64_t ld)
{
    if (r <= 0 || c <= 0) return 0;
    return (size_t)(ld * (c - 1) + r);
}

/* Pack rows x cols of X (element (i, q) at x[off + i*si + q*sq]) into 2-row panels at d[dst]. */
static void pack(const void *x, omb_type t, size_t xlen, uint64_t rows, uint64_t cols,
                 uint64_t off, uint64_t si, uint64_t sq, uint32_t *d, size_t dlen, uint64_t dst)
{
    pk p = { .rows = rows, .cols = cols, .si = si, .off = off, .sq = sq, .w = 2 * cols - 2,
             .dst = dst, .dt = (t == OMB_BF16 || t == OMB_E5M2) ? 1 : 0 };
    if (t == OMB_F32) pack32(&p, x, xlen, d, dlen);
    else if (is16(t)) pack16(&p, x, xlen, d, dlen);
    else pack8(&p, x, xlen, d, dlen);
}

/* The column-major core. ta, tb: 0 = no transpose, 1 = transpose. Arguments already checked. */
static void core_gemm(int ta, int tb, int64_t m, int64_t n, int64_t k, float alpha, const void *A,
                      omb_type at, int64_t lda, const void *B, omb_type bt, int64_t ldb, float beta,
                      void *C, omb_type ct, int64_t ldc, const float *bias, omb_bias bmode,
                      omb_act act)
{
    if (m == 0 || n == 0) return;
    int epi = act != OMB_ACT_NONE || (bias && bmode != OMB_BIAS_NONE);
    if ((alpha == 0.0f || k == 0) && beta == 1.0f && !epi) return;
    int nok = (alpha == 0.0f || k == 0); /* the reference reads no A or B */
    size_t la = ta ? span(k, m, lda) : span(m, k, lda);
    size_t lb = tb ? span(n, k, ldb) : span(k, n, ldb);
    size_t lc = span(m, n, ldc);
    int64_t blen = epi && bias && bmode != OMB_BIAS_NONE ? (bmode == OMB_BIAS_ROW ? m : n) : 0;
    size_t awords = 2 * (size_t)((MC + 1) / 2) * KC, bwords = 2 * (size_t)((NC + 1) / 2) * KC;
    size_t wlen = awords + bwords + (size_t)blen;
    uint32_t *w = xmalloc(wlen * 4);
    if (blen) memcpy(w + awords + bwords, bias, (size_t)blen * 4);
    uint32_t *stg = NULL;
    size_t slen = 0;
    if (ct != OMB_F32) { slen = (size_t)m * NC; stg = xmalloc(slen * 4); }
    uint64_t dt = ct == OMB_BF16 ? 1 : 0;
    uint64_t actw = (uint64_t)act | (blen ? (bmode == OMB_BIAS_ROW ? 16u : 48u) : 0u);

    for (int64_t jc = 0; jc < n; jc += NC) {
        int64_t nb = n - jc < NC ? n - jc : NC;
        if (ct != OMB_F32) {
            pk sp = { .rows = (uint64_t)m, .cols = (uint64_t)nb, .si = 1, .sq = (uint64_t)ldc,
                      .off = (uint64_t)(jc * ldc), .dst = 0, .dt = dt };
            if (beta != 0.0f) stage16(&sp, C, lc, stg, slen);
            else memset(stg, 0, slen * 4);
        }
        int64_t kstep = nok ? 1 : KC;
        int64_t kend = nok ? 1 : k;
        for (int64_t pc = 0; pc < kend; pc += kstep) {
            int64_t kb = nok ? 0 : (k - pc < KC ? k - pc : KC);
            int last = pc + kstep >= kend;
            if (kb) {
                if (tb) pack(B, bt, lb, nb, kb, jc + pc * ldb, 1, ldb, w, wlen, awords);
                else pack(B, bt, lb, nb, kb, pc + jc * ldb, ldb, 1, w, wlen, awords);
            }
            for (int64_t ic = 0; ic < m; ic += MC) {
                int64_t mb = m - ic < MC ? m - ic : MC;
                if (kb) {
                    if (ta) pack(A, at, la, mb, kb, pc + ic * lda, lda, 1, w, wlen, 0);
                    else pack(A, at, la, mb, kb, ic + pc * lda, 1, lda, w, wlen, 0);
                }
                gp g = { .m = (uint64_t)mb, .n = (uint64_t)nb, .k = (uint64_t)kb,
                         .alpha = f2u(nok ? 0.0f : alpha), .beta = f2u(pc == 0 ? beta : 1.0f),
                         .act = last ? actw : 0, .wsa = 0, .wsb = awords,
                         .wsbias = awords + bwords + (uint64_t)(bmode == OMB_BIAS_ROW ? ic : jc),
                         .mp = (uint64_t)(mb + 1) / 2, .np = (uint64_t)(nb + 1) / 2 };
                if (ct == OMB_F32) {
                    g.ldc = (uint64_t)ldc;
                    g.offc = (uint64_t)(ic + jc * ldc);
                    gemm_k(&g, w, wlen, C, lc);
                } else {
                    g.ldc = (uint64_t)m;
                    g.offc = (uint64_t)ic;
                    gemm_k(&g, w, wlen, stg, slen);
                }
            }
        }
        if (ct != OMB_F32) {
            pk up = { .rows = (uint64_t)m, .cols = (uint64_t)nb, .si = 1, .sq = (uint64_t)ldc,
                      .off = (uint64_t)(jc * ldc), .dst = 0, .dt = dt };
            unpack16(&up, stg, slen, C, lc);
        }
    }
    free(stg);
    free(w);
}

static int64_t chk_gemm(char ta, char tb, int64_t m, int64_t n, int64_t k, int64_t lda,
                        int64_t ldb, int64_t ldc)
{
    ck c = { .ta = (unsigned char)ta, .tb = (unsigned char)tb, .m = m, .n = n, .k = k,
             .lda = lda, .ldb = ldb, .ldc = ldc };
    return (int64_t)gemm_info(&c);
}

/* ---- device routing (README.md, "GB10 device path") ----
 * MIGRATION-PENDING C, build switch OMB_GB10 (make GB10=1; the default build has no device code).
 * The dispatch to CUDApp omega_gpu_gemm_ex stays C because an Omega unit cannot call a C entry
 * point; its named successor is a unit-to-unit Omega call into CUDApp's src/omega_gpu_gemm_ex.om
 * (docs/EXTENSIONS.md). Argument checks and the gemv gather/scatter stay in the Omega unit. */

static int omb_dev = -1; /* -1: OMEGABLAS_DEVICE not read yet */
static omb_route_counts omb_counts;

static void omb_die(const char *what, const char *detail)
{
    fprintf(stderr, "omega-blas: %s%s\n", what, detail ? detail : "");
    abort();
}

/* lock held */
static int dev_gb10(void)
{
    if (omb_dev < 0) {
        const char *e = getenv("OMEGABLAS_DEVICE");
        if (!e || !*e || !strcmp(e, "host")) omb_dev = OMB_DEVICE_HOST;
#ifdef OMB_GB10
        else if (!strcmp(e, "gb10")) omb_dev = OMB_DEVICE_GB10;
#else
        else if (!strcmp(e, "gb10")) omb_die("OMEGABLAS_DEVICE=gb10, but this library was built without the GB10 path (make GB10=1)", NULL);
#endif
        else omb_die("OMEGABLAS_DEVICE must be host or gb10, not ", e);
    }
    return omb_dev == OMB_DEVICE_GB10;
}

#ifdef OMB_GB10
/* count column-major products C_b (m x n, ldc) = alpha op(A_b) op(B_b) + beta C_b at element strides
 * sa, sb, sc (all >= 0, C batches disjoint), as gemm_ex's row-major C^T = op(B)^T op(A)^T: column-major
 * storage of B, A and C is row-major storage of their transposes, so A' = B (op tb, lda' = ldb),
 * B' = A (op ta, ldb' = lda), M' = n, N' = m. k = 0 runs as alpha 0, k 1 (C = beta C, A and B unread).
 * Arguments already checked, m, n, count >= 1. Any error aborts. lock held */
static void gb10_gemm(int ta, int tb, int64_t m, int64_t n, int64_t k, float alpha, const float *A,
                      int64_t lda, const float *B, int64_t ldb, float beta, float *C, int64_t ldc,
                      int64_t count, int64_t sa, int64_t sb, int64_t sc)
{
    int nok = alpha == 0.0f || k == 0;
    OmegaGemmExInfo inf;
    OmegaGemmExArgs x;
    memset(&x, 0, sizeof x);
    memset(&inf, 0, sizeof inf);
    x.op_a = tb ? OMEGA_GEMM_OP_T : OMEGA_GEMM_OP_N;
    x.op_b = ta ? OMEGA_GEMM_OP_T : OMEGA_GEMM_OP_N;
    x.m = (uint64_t)n;
    x.n = (uint64_t)m;
    x.k = nok ? 1u : (uint64_t)k;
    x.batch = (uint64_t)count;
    x.a_type = x.b_type = x.c_type = OMEGA_GEMM_DT_F32;
    x.a = nok ? NULL : B;
    x.lda = (uint64_t)ldb;
    x.stride_a = (uint64_t)sb;
    x.b = nok ? NULL : A;
    x.ldb = (uint64_t)lda;
    x.stride_b = (uint64_t)sa;
    x.c = C;
    x.ldc = (uint64_t)ldc;
    x.stride_c = (uint64_t)sc;
    x.alpha = nok ? 0.0f : alpha;
    x.beta = beta;
    x.compute = OMEGA_GEMM_EX_DEFAULT;
    x.scale = OMEGA_GEMM_SCALE_NONE;
    x.info = &inf;
    int rc = omega_gpu_gemm_ex(&x);
    if (rc != OMEGA_GPU_GEMM_OK) {
        const char *why = omega_gpu_session_last_error();
        fprintf(stderr, "omega-blas: omega_gpu_gemm_ex %s (%d) on m %lld n %lld k %lld batch %lld: %s\n",
                omega_gpu_gemm_rc_name(rc), rc, (long long)m, (long long)n, (long long)k,
                (long long)count, why && *why ? why : "-");
        abort();
    }
    omb_counts.gb10++;
    omb_counts.gb10_core_calls += inf.core_calls;
    omb_counts.gb10_host_fixed += inf.host_fixed;
}
#endif

static size_t esize(omb_type t) { return t == OMB_F32 ? 4 : is16(t) ? 2 : 1; }
static const void *at_elem(const void *p, int64_t i, omb_type t) { return (const char *)p + (ptrdiff_t)i * (ptrdiff_t)esize(t); }

/* count checked column-major products at element strides sa, sb, sc: on the GB10 for f32 without an
 * epilogue when that device is selected, else in the host core. lock held */
static void run_gemm(int ta, int tb, int64_t m, int64_t n, int64_t k, float alpha, const void *A,
                     omb_type at, int64_t lda, const void *B, omb_type bt, int64_t ldb, float beta,
                     void *C, omb_type ct, int64_t ldc, const float *bias, omb_bias bmode,
                     omb_act act, int64_t count, int64_t sa, int64_t sb, int64_t sc)
{
    int epi = act != OMB_ACT_NONE || (bias && bmode != OMB_BIAS_NONE);
    if (m == 0 || n == 0 || count == 0 || ((alpha == 0.0f || k == 0) && beta == 1.0f && !epi)) return;
    int gb10 = dev_gb10();
#ifdef OMB_GB10
    if (gb10 && at == OMB_F32 && bt == OMB_F32 && ct == OMB_F32 && !epi) {
        if (sa >= 0 && sb >= 0 && (count == 1 || sc >= (int64_t)span(m, n, ldc)))
            gb10_gemm(ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, count, sa, sb, sc);
        else
            for (int64_t b = 0; b < count; b++)
                gb10_gemm(ta, tb, m, n, k, alpha, at_elem(A, b * sa, at), lda, at_elem(B, b * sb, bt),
                          ldb, beta, (void *)at_elem(C, b * sc, ct), ldc, 1, 0, 0, 0);
        return;
    }
#else
    (void)gb10;
#endif
    for (int64_t b = 0; b < count; b++) {
        core_gemm(ta, tb, m, n, k, alpha, at_elem(A, b * sa, at), at, lda, at_elem(B, b * sb, bt), bt,
                  ldb, beta, (void *)at_elem(C, b * sc, ct), ct, ldc, bias, bmode, act);
        omb_counts.host++;
    }
}

int omb_set_device(omb_device d)
{
#ifdef OMB_GB10
    if (d != OMB_DEVICE_HOST && d != OMB_DEVICE_GB10) return -1;
#else
    if (d != OMB_DEVICE_HOST) return -1;
#endif
    pthread_mutex_lock(&omb_lock);
    omb_dev = (int)d;
    pthread_mutex_unlock(&omb_lock);
    return 0;
}

omb_device omb_get_device(void)
{
    pthread_mutex_lock(&omb_lock);
    omb_device d = dev_gb10() ? OMB_DEVICE_GB10 : OMB_DEVICE_HOST;
    pthread_mutex_unlock(&omb_lock);
    return d;
}

void omb_get_route_counts(omb_route_counts *out)
{
    pthread_mutex_lock(&omb_lock);
    *out = omb_counts;
    pthread_mutex_unlock(&omb_lock);
}

#ifndef OMB_CUDAPP_REV
#define OMB_CUDAPP_REV "none"
#endif
/* OMEGABLAS_ROUTE_REPORT=1: one line on stderr at unload, the receipt that calls reached the device. */
__attribute__((destructor)) static void omb_route_report(void)
{
    const char *e = getenv("OMEGABLAS_ROUTE_REPORT");
    if (!e || strcmp(e, "1")) return;
    fprintf(stderr, "omega-blas routes: device %s host %llu gb10 %llu gb10_core_calls %llu gb10_host_fixed %llu cudapp %s\n",
            omb_dev == OMB_DEVICE_GB10 ? "gb10" : omb_dev == OMB_DEVICE_HOST ? "host" : "unread",
            (unsigned long long)omb_counts.host, (unsigned long long)omb_counts.gb10,
            (unsigned long long)omb_counts.gb10_core_calls,
            (unsigned long long)omb_counts.gb10_host_fixed, OMB_CUDAPP_REV);
}

static int trn(char c) { return !(c == 'N' || c == 'n'); }

/* ---- Fortran 77 ---- */

__attribute__((weak)) void xerbla_(const char *srname, const int *info, size_t len)
{
    char name[8] = { 0 };
    size_t i;
    for (i = 0; i < len && i < 7 && srname[i] != ' '; i++) name[i] = srname[i];
    if (CBLAS_CallFromC) {
        char rout[16] = "cblas_";
        for (i = 0; name[i]; i++) rout[6 + i] = (char)(name[i] | 0x20);
        cblas_xerbla(*info + 1, rout, "");
        return;
    }
    fprintf(stderr, " ** On entry to %s parameter number %d had an illegal value\n", name, *info);
    exit(1);
}

__attribute__((weak)) void cblas_xerbla(int p, const char *rout, const char *form, ...)
{
    (void)form;
    if (RowMajorStrg && strstr(rout, "gemm")) {
        if (p == 4) p = 5; else if (p == 5) p = 4; else if (p == 9) p = 11; else if (p == 11) p = 9;
    } else if (RowMajorStrg && strstr(rout, "gemv")) {
        if (p == 3) p = 4; else if (p == 4) p = 3;
    }
    fprintf(stderr, "Parameter %d to routine %s was incorrect\n", p, rout);
}

void sgemm_(const char *transa, const char *transb, const int *m, const int *n, const int *k,
            const float *alpha, const float *a, const int *lda, const float *b, const int *ldb,
            const float *beta, float *c, const int *ldc, size_t lta, size_t ltb)
{
    (void)lta; (void)ltb;
    pthread_mutex_lock(&omb_lock);
    int64_t info = chk_gemm(*transa, *transb, *m, *n, *k, *lda, *ldb, *ldc);
    if (info) {
        pthread_mutex_unlock(&omb_lock);
        int i = (int)info;
        xerbla_("SGEMM ", &i, 6);
        return;
    }
    run_gemm(trn(*transa), trn(*transb), *m, *n, *k, *alpha, a, OMB_F32, *lda, b, OMB_F32, *ldb,
             *beta, c, OMB_F32, *ldc, NULL, OMB_BIAS_NONE, OMB_ACT_NONE, 1, 0, 0, 0);
    pthread_mutex_unlock(&omb_lock);
}

/* x of length nx with increment inc as a view: its covering length and the index of element 0. */
static size_t vlen(int64_t nx, int64_t inc) { return nx ? (size_t)(1 + (nx - 1) * (inc < 0 ? -inc : inc)) : 0; }
static uint64_t vstart(int64_t nx, int64_t inc) { return inc < 0 ? (uint64_t)((nx - 1) * -inc) : 0; }

void sgemv_(const char *trans, const int *m, const int *n, const float *alpha, const float *a,
            const int *lda, const float *x, const int *incx, const float *beta, float *y,
            const int *incy, size_t ltrans)
{
    (void)ltrans;
    pthread_mutex_lock(&omb_lock);
    ck c = { .ta = (unsigned char)*trans, .m = *m, .n = *n, .lda = *lda, .incx = *incx,
             .incy = *incy };
    int info = (int)gemv_info(&c);
    if (info) {
        pthread_mutex_unlock(&omb_lock);
        xerbla_("SGEMV ", &info, 6);
        return;
    }
    int64_t M = *m, N = *n, ix = *incx, iy = *incy;
    if (M == 0 || N == 0 || (*alpha == 0.0f && *beta == 1.0f)) {
        pthread_mutex_unlock(&omb_lock);
        return;
    }
    int t = trn(*trans);
    int64_t lx = t ? M : N, ly = t ? N : M;
    size_t wl = (size_t)(lx + ly);
    uint32_t *w = xmalloc(wl * 4);
    size_t xl = vlen(lx, ix), yl = vlen(ly, iy);
    cp gx = { .n = (uint64_t)lx, .os = vstart(lx, ix), .is = ix, .od = 0, .id = 1 };
    copy32(&gx, (const uint32_t *)x, xl, w, wl);
    cp gy = { .n = (uint64_t)ly, .os = vstart(ly, iy), .is = iy, .od = (uint64_t)lx, .id = 1 };
    copy32(&gy, (const uint32_t *)y, yl, w, wl);
    if (dev_gb10()) {
#ifdef OMB_GB10
        /* y (ly) = alpha op(A) x + beta y as the column-major product with x a k = lx by 1 B and
         * y a ly by 1 C, both in the workspace */
        gb10_gemm(t, 0, ly, 1, lx, *alpha, a, *lda, (const float *)w, lx, *beta, (float *)(w + lx),
                  ly, 1, 0, 0, 0);
#endif
    } else {
        gv g = { .m = (uint64_t)M, .n = (uint64_t)N, .lda = (uint64_t)*lda, .offa = 0, .ta = (uint64_t)t,
                 .alpha = f2u(*alpha), .beta = f2u(*beta), .wx = 0, .wy = (uint64_t)lx };
        gemv_k(&g, (const uint32_t *)a, span(M, N, *lda), w, wl);
        omb_counts.host++;
    }
    cp sy = { .n = (uint64_t)ly, .os = (uint64_t)lx, .is = 1, .od = vstart(ly, iy), .id = iy };
    copy32(&sy, w, wl, (uint32_t *)y, yl);
    free(w);
    pthread_mutex_unlock(&omb_lock);
}

/* ---- CBLAS ---- */

static char tchar(enum CBLAS_TRANSPOSE t)
{
    return t == CblasNoTrans ? 'N' : t == CblasTrans ? 'T' : t == CblasConjTrans ? 'C' : 0;
}

void cblas_sgemm(CBLAS_LAYOUT layout, enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
                 int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb,
                 float beta, float *C, int ldc)
{
    char ta = tchar(TransA), tb = tchar(TransB);
    RowMajorStrg = 0;
    CBLAS_CallFromC = 1;
    if (layout != CblasColMajor && layout != CblasRowMajor) {
        cblas_xerbla(1, "cblas_sgemm", "Illegal layout setting, %d\n", layout);
    } else {
        if (layout == CblasRowMajor) RowMajorStrg = 1;
        if (!ta) cblas_xerbla(2, "cblas_sgemm", "Illegal TransA setting, %d\n", TransA);
        else if (!tb) cblas_xerbla(3, "cblas_sgemm", "Illegal TransB setting, %d\n", TransB);
        else if (layout == CblasColMajor)
            sgemm_(&ta, &tb, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);
        else
            sgemm_(&tb, &ta, &N, &M, &K, &alpha, B, &ldb, A, &lda, &beta, C, &ldc, 1, 1);
    }
    CBLAS_CallFromC = 0;
    RowMajorStrg = 0;
}

void cblas_sgemv(CBLAS_LAYOUT layout, enum CBLAS_TRANSPOSE TransA, int M, int N, float alpha,
                 const float *A, int lda, const float *X, int incX, float beta, float *Y, int incY)
{
    char ta = tchar(TransA);
    RowMajorStrg = 0;
    CBLAS_CallFromC = 1;
    if (layout == CblasColMajor) {
        if (!ta) cblas_xerbla(2, "cblas_sgemv", "Illegal TransA setting, %d\n", TransA);
        else sgemv_(&ta, &M, &N, &alpha, A, &lda, X, &incX, &beta, Y, &incY, 1);
    } else if (layout == CblasRowMajor) {
        RowMajorStrg = 1;
        char tr = ta == 'N' ? 'T' : 'N';
        if (!ta) cblas_xerbla(2, "cblas_sgemv", "Illegal TransA setting, %d\n", TransA);
        else sgemv_(&tr, &N, &M, &alpha, A, &lda, X, &incX, &beta, Y, &incY, 1);
    } else {
        cblas_xerbla(1, "cblas_sgemv", "Illegal layout setting, %d\n", layout);
    }
    CBLAS_CallFromC = 0;
    RowMajorStrg = 0;
}

/* ---- extensions ---- */

/* omb_gemm_ex over count problems at element strides sA, sB, sC (count >= 1; batches share every
 * argument but the pointers, so one check covers all). */
static int gemm_exb(CBLAS_LAYOUT layout, enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
                    int M, int N, int K, float alpha, const void *A, omb_type atype, int lda,
                    const void *B, omb_type btype, int ldb, float beta, void *C, omb_type ctype,
                    int ldc, const float *bias, omb_bias bias_mode, omb_act act, int64_t count,
                    int64_t sA, int64_t sB, int64_t sC)
{
    char ta = tchar(TransA), tb = tchar(TransB);
    if (layout != CblasColMajor && layout != CblasRowMajor) return 1;
    if (!ta) return 2;
    if (!tb) return 3;
    if ((unsigned)atype > OMB_E5M2) return 9;
    if ((unsigned)btype > OMB_E5M2) return 12;
    if (ctype != OMB_F32 && !is16(ctype)) return 16;
    if ((unsigned)bias_mode > OMB_BIAS_COL || (bias_mode != OMB_BIAS_NONE && !bias)) return 19;
    if ((unsigned)act > OMB_ACT_GELU_TANH) return 20;
    pthread_mutex_lock(&omb_lock);
    int64_t info;
    if (layout == CblasColMajor) {
        info = chk_gemm(ta, tb, M, N, K, lda, ldb, ldc);
        if (!info)
            run_gemm(trn(ta), trn(tb), M, N, K, alpha, A, atype, lda, B, btype, ldb, beta, C,
                     ctype, ldc, bias, bias_mode, act, count, sA, sB, sC);
        else info += 1;
    } else {
        /* row major C (M x N) is column major C^T = op(B)^T op(A)^T (N x M): a row bias of C is
         * a column bias of C^T. */
        omb_bias bm = bias_mode == OMB_BIAS_ROW ? OMB_BIAS_COL
                    : bias_mode == OMB_BIAS_COL ? OMB_BIAS_ROW : OMB_BIAS_NONE;
        info = chk_gemm(tb, ta, N, M, K, ldb, lda, ldc);
        if (!info)
            run_gemm(trn(tb), trn(ta), N, M, K, alpha, B, btype, ldb, A, atype, lda, beta, C,
                     ctype, ldc, bias, bm, act, count, sB, sA, sC);
        else {
            info += 1;
            if (info == 4) info = 5; else if (info == 5) info = 4;
            else if (info == 9) info = 11; else if (info == 11) info = 9;
        }
    }
    pthread_mutex_unlock(&omb_lock);
    return (int)info;
}

int omb_gemm_ex(CBLAS_LAYOUT layout, enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
                int M, int N, int K, float alpha, const void *A, omb_type atype, int lda,
                const void *B, omb_type btype, int ldb, float beta, void *C, omb_type ctype,
                int ldc, const float *bias, omb_bias bias_mode, omb_act act)
{
    return gemm_exb(layout, TransA, TransB, M, N, K, alpha, A, atype, lda, B, btype, ldb, beta, C,
                    ctype, ldc, bias, bias_mode, act, 1, 0, 0, 0);
}

int omb_hgemm(CBLAS_LAYOUT layout, enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
              int M, int N, int K, float alpha, const uint16_t *A, int lda, const uint16_t *B,
              int ldb, float beta, uint16_t *C, int ldc)
{
    return omb_gemm_ex(layout, TransA, TransB, M, N, K, alpha, A, OMB_F16, lda, B, OMB_F16, ldb,
                       beta, C, OMB_F16, ldc, NULL, OMB_BIAS_NONE, OMB_ACT_NONE);
}

int omb_bfgemm(CBLAS_LAYOUT layout, enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
               int M, int N, int K, float alpha, const uint16_t *A, int lda, const uint16_t *B,
               int ldb, float beta, uint16_t *C, int ldc)
{
    return omb_gemm_ex(layout, TransA, TransB, M, N, K, alpha, A, OMB_BF16, lda, B, OMB_BF16, ldb,
                       beta, C, OMB_BF16, ldc, NULL, OMB_BIAS_NONE, OMB_ACT_NONE);
}

int omb_gemm_batched(CBLAS_LAYOUT layout, enum CBLAS_TRANSPOSE TransA,
                     enum CBLAS_TRANSPOSE TransB, int M, int N, int K, float alpha,
                     const void *const *A, omb_type atype, int lda, const void *const *B,
                     omb_type btype, int ldb, float beta, void *const *C, omb_type ctype, int ldc,
                     int count)
{
    if (count < 0) return 21;
    for (int b = 0; b < count; b++) {
        int r = omb_gemm_ex(layout, TransA, TransB, M, N, K, alpha, A[b], atype, lda, B[b], btype,
                            ldb, beta, C[b], ctype, ldc, NULL, OMB_BIAS_NONE, OMB_ACT_NONE);
        if (r) return r;
    }
    return 0;
}

int omb_gemm_strided_batched(CBLAS_LAYOUT layout, enum CBLAS_TRANSPOSE TransA,
                             enum CBLAS_TRANSPOSE TransB, int M, int N, int K, float alpha,
                             const void *A, omb_type atype, int lda, int64_t strideA,
                             const void *B, omb_type btype, int ldb, int64_t strideB, float beta,
                             void *C, omb_type ctype, int ldc, int64_t strideC, int count)
{
    if (count < 0) return 21;
    if (count == 0) return 0;
    return gemm_exb(layout, TransA, TransB, M, N, K, alpha, A, atype, lda, B, btype, ldb, beta, C,
                    ctype, ldc, NULL, OMB_BIAS_NONE, OMB_ACT_NONE, count, strideA, strideB, strideC);
}

/* The seeded Omega loop bound is 1,000,000 iterations; convert in chunks below it. */
#define OMB_CHUNK ((size_t)1 << 19)

void omb_widen(omb_type src, const void *s, float *d, size_t n)
{
    pthread_mutex_lock(&omb_lock);
    if (src == OMB_F32) memmove(d, s, n * 4);
    else
        for (size_t o = 0; o < n; o += OMB_CHUNK) {
            size_t c = n - o < OMB_CHUNK ? n - o : OMB_CHUNK;
            if (is16(src)) widen16(src == OMB_BF16, (const uint16_t *)s + o, c, (uint32_t *)d + o, c);
            else if (is8(src)) widen8(src == OMB_E5M2, (const uint8_t *)s + o, c, (uint32_t *)d + o, c);
        }
    pthread_mutex_unlock(&omb_lock);
}

void omb_narrow(omb_type dst, const float *s, void *d, size_t n)
{
    pthread_mutex_lock(&omb_lock);
    if (dst == OMB_F32) memmove(d, s, n * 4);
    else if (is16(dst))
        for (size_t o = 0; o < n; o += OMB_CHUNK) {
            size_t c = n - o < OMB_CHUNK ? n - o : OMB_CHUNK;
            narrow16(dst == OMB_BF16, (const uint32_t *)s + o, c, (uint16_t *)d + o, c);
        }
    pthread_mutex_unlock(&omb_lock);
}

const char *omb_ident(void) { return omega2_ident_blas(); }
