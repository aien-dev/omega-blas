/* tests/gb10/hooks.c: test-only constructor linked into the netlib drivers of a GB10=1 build
 * (tests/netlib/run.sh with GB10_HOOKS=1). libomegablas itself has neither knob.
 *   OMB_TEST_BACKEND=model  replace CUDApp's chip core with the chip-exact HMMA model of CUDApp
 *                           tests/bw_hmma_model.h ({16, 2, RZ, nominal 2}, evidence
 *                           OWN30B-T3-HMMA-MODEL): the device glue runs on the host, no GPU opened
 *   OMB_TEST_MUTANT=<n>     plant CUDApp gemm_ex mutant n (1: the last k element is dropped), the
 *                           negative control; the drivers must report failures
 * MIGRATION-PENDING C test glue: it exists because the backend hook is a C function pointer; its
 * successor is the same model as an Omega unit once CUDApp's gemm_ex core is one (docs/EXTENSIONS.md).
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
#include "omega_gpu_gemm_ex.h"
#include "omega_gpu_gemm.h"
#include "bw_hmma_model.h"

#include <stdio.h>
#include <stdlib.h>

static const HmmaModel HM = { 16, 2, 0, 2, 0 };

static float bf(const void *p, uint64_t i)
{
    uint32_t u = (uint32_t)((const uint16_t *)p)[i] << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

/* the bf16 core only: the bf16x3 route of an f32 gemm_ex call is the one omega-blas sends */
static int model_core(void *ctx, OmegaGpuGemmType t, unsigned lay, uint32_t M, uint32_t N, uint32_t K,
                      const void *A, uint64_t lda, const void *B, uint64_t ldb, float *C)
{
    (void)ctx;
    if (t != OMEGA_GPU_GEMM_BF16) {
        fprintf(stderr, "omega-blas test model: core type %d is not modelled\n", (int)t);
        return OMEGA_GPU_GEMM_UNSUPPORTED;
    }
    const int acol = (lay & OMEGA_GPU_GEMM_A_COL) != 0, bcol = (lay & OMEGA_GPU_GEMM_B_COL) != 0;
    const uint64_t ea = lda ? lda : acol ? M : K, eb = ldb ? ldb : bcol ? K : N;
    for (uint32_t m = 0; m < M; m++)
        for (uint32_t n = 0; n < N; n++) {
            float acc = 0.0f;
            for (uint32_t k0 = 0; k0 < K; k0 += 16) {
                float av[16], bv[16];
                for (uint32_t k = 0; k < 16; k++) {
                    uint64_t kk = k0 + k;
                    if (kk >= K) { av[k] = bv[k] = 0.0f; continue; }
                    av[k] = bf(A, acol ? kk * ea + m : (uint64_t)m * ea + kk);
                    bv[k] = bf(B, bcol ? (uint64_t)n * eb + kk : kk * eb + n);
                }
                acc = hm_dot16(&HM, av, bv, acc);
            }
            C[(uint64_t)m * N + n] = acc;
        }
    return OMEGA_GPU_GEMM_OK;
}

__attribute__((constructor)) static void omb_test_hooks(void)
{
    const char *b = getenv("OMB_TEST_BACKEND"), *m = getenv("OMB_TEST_MUTANT");
    if (b && *b) {
        if (strcmp(b, "model")) { fprintf(stderr, "omega-blas test hooks: OMB_TEST_BACKEND must be model\n"); abort(); }
        omega_gpu_gemm_ex_set_backend(model_core, NULL);
        fprintf(stderr, "omega-blas test hooks: backend model (bw_hmma_model {16,2,RZ,nominal 2})\n");
    }
    if (m && *m) {
        omega_gpu_gemm_ex_test_set_mutant(atoi(m));
        fprintf(stderr, "omega-blas test hooks: gemm_ex mutant %d planted\n", atoi(m));
    }
}
