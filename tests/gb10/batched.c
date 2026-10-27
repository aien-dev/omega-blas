/* tests/gb10/batched.c: the f32 entry points netlib does not call (omb_gemm_strided_batched,
 * omb_gemm_batched, omb_gemm_ex), checked on whichever device OMEGABLAS_DEVICE selects.
 * Run by tests/gb10/run.sh. A case passes when all three hold:
 *  - every C element: ratio = |C - ref| / (2^-23 G) < limit, with G = |alpha| sum|a||b| + |beta||c0|
 *    + |bias| and ref computed in double (exact to far below 2^-23 G). Every case has K <= 16, one
 *    HMMA step per plane pair, so docs/TOLERANCE.md gives, against the exact product: device route
 *    5 c + 1/2 + O(2^-23) = 5.69993 (limit 5.70), host route gamma_(K+3)/2^-23 = 9.50001 (limit 9.51);
 *  - every element outside the products (ld padding, gaps between batches) is unchanged;
 *  - the route counters move as predicted. One gemm_ex call per disjoint strided batch, one per
 *    problem otherwise; epilogue and non-f32 calls stay on the host.
 * Prints one line per failure, the largest ratio and the case count; exits 0 only if every case
 * passes.
 * MIGRATION-PENDING C test driver: it drives the C ABI from outside, which an Omega unit cannot do
 * yet. Successor: an Omega driver once omega-bridge carries Omega-to-C calls (docs/EXTENSIONS.md).
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
#include "omegablas.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SENT 1234.5f
#define LIM_DEV 5.70
#define LIM_HOST 9.51

enum mode { STRIDED, OVERLAP, PTR, EX_BIAS, EX_BF16 };
static const char *mname[] = { "strided", "overlap", "pointer", "ex+bias", "ex-bf16" };

static uint64_t rs = 0x9e3779b97f4a7c15u;
static float rnd(void) /* [-1, 1), 24 significant bits */
{
    rs = rs * 6364136223846793005u + 1442695040888963407u;
    return (float)((double)(rs >> 40) * 0x1p-23 - 1.0);
}

static float bf16val(uint16_t h)
{
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static int fails, cases, gdev;
static double worst_dev, worst_host;

/* element (i, j) of a stored matrix in the layout */
static size_t ix(int row, int ld, int i, int j) { return row ? (size_t)i * ld + j : i + (size_t)j * ld; }

static void run_case(int row, int ta, int tb, int M, int N, int K, int count, enum mode md)
{
    char name[96];
    snprintf(name, sizeof name, "%s %s ta=%d tb=%d M=%d N=%d K=%d count=%d", mname[md],
             row ? "row" : "col", ta, tb, M, N, K, count);
    cases++;
    CBLAS_LAYOUT lay = row ? CblasRowMajor : CblasColMajor;
    enum CBLAS_TRANSPOSE TA = ta ? CblasTrans : CblasNoTrans, TB = tb ? CblasTrans : CblasNoTrans;
    /* stored shapes: A is (ta ? K x M : M x K), B is (tb ? N x K : K x N), C is M x N */
    int ar = ta ? K : M, ac = ta ? M : K, br = tb ? N : K, bc = tb ? K : N;
    int lda = (row ? ac : ar) + 2, ldb = (row ? bc : br) + 2, ldc = (row ? N : M) + 2;
    if (lda < 1) lda = 1;
    if (ldb < 1) ldb = 1;
    size_t sa = (size_t)lda * (row ? ar : ac) + 5, sb = (size_t)ldb * (row ? br : bc) + 5;
    size_t sc = (size_t)ldc * (row ? M : N) + 5;
    size_t nc = md == OVERLAP ? sc : sc * count;
    float *A = malloc(sa * count * 4), *B = malloc(sb * count * 4), *C = malloc(nc * 4), *C0 = malloc(nc * 4);
    uint16_t *Ah = malloc(sa * count * 2), *Bh = malloc(sb * count * 2);
    float bias[16];
    for (size_t i = 0; i < sa * count; i++) {
        A[i] = rnd();
        if (md == EX_BF16) { uint32_t u; memcpy(&u, &A[i], 4); Ah[i] = u >> 16; A[i] = bf16val(Ah[i]); }
    }
    for (size_t i = 0; i < sb * count; i++) {
        B[i] = rnd();
        if (md == EX_BF16) { uint32_t u; memcpy(&u, &B[i], 4); Bh[i] = u >> 16; B[i] = bf16val(Bh[i]); }
    }
    for (size_t i = 0; i < nc; i++) C[i] = SENT;
    int nb = md == OVERLAP ? 1 : count;
    for (int b = 0; b < nb; b++)
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) C[b * sc + ix(row, ldc, i, j)] = rnd();
    for (int i = 0; i < M; i++) bias[i] = rnd();
    memcpy(C0, C, nc * 4);
    const float al = 1.25f, be = md == OVERLAP ? 0.0f : -0.75f;

    omb_route_counts r0, r1;
    omb_get_route_counts(&r0);
    int info = 0;
    if (md == STRIDED || md == OVERLAP)
        info = omb_gemm_strided_batched(lay, TA, TB, M, N, K, al, A, OMB_F32, lda, (int64_t)sa, B,
                                        OMB_F32, ldb, (int64_t)sb, be, C, OMB_F32, ldc,
                                        md == OVERLAP ? 0 : (int64_t)sc, count);
    else if (md == PTR) {
        const void *pa[8], *pb[8];
        void *pc[8];
        for (int b = 0; b < count; b++) { pa[b] = A + b * sa; pb[b] = B + b * sb; pc[b] = C + b * sc; }
        info = omb_gemm_batched(lay, TA, TB, M, N, K, al, pa, OMB_F32, lda, pb, OMB_F32, ldb, be, pc,
                                OMB_F32, ldc, count);
    } else if (md == EX_BIAS)
        info = omb_gemm_ex(lay, TA, TB, M, N, K, al, A, OMB_F32, lda, B, OMB_F32, ldb, be, C, OMB_F32,
                           ldc, bias, OMB_BIAS_ROW, OMB_ACT_NONE);
    else
        info = omb_gemm_ex(lay, TA, TB, M, N, K, al, Ah, OMB_BF16, lda, Bh, OMB_BF16, ldb, be, C,
                           OMB_F32, ldc, NULL, OMB_BIAS_NONE, OMB_ACT_NONE);
    omb_get_route_counts(&r1);

    int bad = 0;
    if (info) { printf("FAIL %s: info %d\n", name, info); bad = 1; }
    /* predicted routes */
    uint64_t eg = 0, eh = (uint64_t)count;
    if (gdev && (md == STRIDED || md == OVERLAP || md == PTR))
        eg = md == STRIDED ? 1 : (uint64_t)count, eh = 0;
    if (r1.gb10 - r0.gb10 != eg || r1.host - r0.host != eh) {
        printf("FAIL %s: routes gb10 +%llu host +%llu, predicted gb10 +%llu host +%llu\n", name,
               (unsigned long long)(r1.gb10 - r0.gb10), (unsigned long long)(r1.host - r0.host),
               (unsigned long long)eg, (unsigned long long)eh);
        bad = 1;
    }
    /* values: in OVERLAP (stride C 0, beta 0) the last problem wins */
    int dr = gdev && (md == STRIDED || md == OVERLAP || md == PTR);
    double lim = dr ? LIM_DEV : LIM_HOST, *wr = dr ? &worst_dev : &worst_host;
    char *in = calloc(nc, 1);
    for (int b = 0; b < count; b++) {
        int cb = md == OVERLAP ? 0 : b;
        if (md == OVERLAP && b != count - 1) continue;
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                double s = 0, g = 0;
                for (int p = 0; p < K; p++) {
                    double a = A[b * sa + (ta ? ix(row, lda, p, i) : ix(row, lda, i, p))];
                    double v = B[b * sb + (tb ? ix(row, ldb, j, p) : ix(row, ldb, p, j))];
                    s += a * v;
                    g += fabs(a * v);
                }
                size_t e = cb * sc + ix(row, ldc, i, j);
                double ref = (double)al * s + (double)be * C0[e];
                g = fabs((double)al) * g + fabs((double)be) * fabs((double)C0[e]);
                if (md == EX_BIAS) { ref += bias[i]; g += fabs((double)bias[i]); }
                double ratio = g > 0 ? fabs((double)C[e] - ref) / (0x1p-23 * g) : (C[e] == ref ? 0 : INFINITY);
                if (ratio > *wr) *wr = ratio;
                if (!(ratio < lim)) {
                    if (!bad) printf("FAIL %s: C(%d,%d) batch %d = %.9g, ref %.17g, ratio %.3g\n", name, i, j, b, C[e], ref, ratio);
                    bad = 1;
                }
                in[e] = 1;
            }
    }
    for (size_t e = 0; e < nc; e++)
        if (!in[e] && C[e] != C0[e]) {
            printf("FAIL %s: element %zu outside the products changed (%.9g -> %.9g)\n", name, e, C0[e], C[e]);
            bad = 1;
            break;
        }
    fails += bad;
    free(in); free(A); free(B); free(C); free(C0); free(Ah); free(Bh);
}

int main(void)
{
    setvbuf(stdout, NULL, _IOLBF, 0); /* a device error aborts: keep the lines printed before it */
    gdev = omb_get_device() == OMB_DEVICE_GB10;
    static const int sh[][3] = { { 1, 1, 1 }, { 3, 5, 2 }, { 9, 7, 9 }, { 16, 13, 16 }, { 4, 6, 0 } };
    for (int row = 0; row < 2; row++)
        for (int ta = 0; ta < 2; ta++)
            for (int tb = 0; tb < 2; tb++)
                for (int s = 0; s < 5; s++) run_case(row, ta, tb, sh[s][0], sh[s][1], sh[s][2], 3, STRIDED);
    run_case(0, 1, 0, 9, 7, 9, 3, OVERLAP);
    run_case(1, 0, 1, 5, 3, 16, 3, OVERLAP);
    run_case(0, 1, 0, 9, 7, 9, 3, PTR);
    run_case(1, 0, 1, 5, 3, 16, 3, PTR);
    run_case(0, 0, 1, 9, 7, 9, 1, EX_BIAS);
    run_case(1, 1, 0, 16, 13, 16, 1, EX_BF16);
    omb_route_counts r;
    omb_get_route_counts(&r);
    printf("batched: device %s, %d cases, %d failed, largest ratio device route %.3f (limit %.2f) host route %.3f (limit %.2f), routes host %llu gb10 %llu gb10_core_calls %llu\n",
           gdev ? "gb10" : "host", cases, fails, worst_dev, LIM_DEV, worst_host, LIM_HOST, (unsigned long long)r.host,
           (unsigned long long)r.gb10, (unsigned long long)r.gb10_core_calls);
    return fails != 0;
}
