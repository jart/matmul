// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c++ ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2024 Mozilla Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifdef __aarch64__

#include "linalg.h"
#include <algorithm>
#include <arm_neon.h>

#define PRECISION 8e-5
#define VECTOR_REGISTERS 32

#define BEGIN_KERNEL(RM, RN) \
    int ytiles = (m - m0) / RM; \
    int xtiles = (n - n0) / RN; \
    int tiles = ytiles * xtiles; \
    int duty = (tiles + nth - 1) / nth; \
    if (duty < 1) \
        duty = 1; \
    int start = duty * ith; \
    int end = start + duty; \
    if (end > tiles) \
        end = tiles; \
    for (int job = start; job < end; ++job) { \
        int i = m0 + job / xtiles * RM; \
        int j = n0 + job % xtiles * RN;

#define END_KERNEL() }

#define TINYBLAS_ANALYSIS
#ifndef TINYBLAS_ANALYSIS
#define BEGIN_CRITICAL(i) (void)0
#define END_CRITICAL(i) (void)0
#elif defined(__x86_64__)
#define BEGIN_CRITICAL(i) asm("sysenter # %0" : : "r"(i))
#define END_CRITICAL(i) asm("cpuid # %0" : : "r"(i))
#else
#define BEGIN_CRITICAL(i) asm("brk #0x123 // %0" : : "r"(i))
#define END_CRITICAL(i) asm("brk #0x456 // %0" : : "r"(i))
#endif

class GEMMER {
  public:
    GEMMER(int k, const float *A, int lda, const float *B, int ldb, float *C, int ldc, int ith,
           int nth)
        : k(k), A(A), lda(lda), B(B), ldb(ldb), C(C), ldc(ldc), ith(ith), nth(nth) {
        ASSERT(A != nullptr);
        ASSERT(B != nullptr);
        ASSERT(C != nullptr);
        ASSERT(k >= 0 && k % 8 == 0);
        ASSERT(ith >= 0 && ith < nth);
    }

    void gemm(int m, int n) {
        ASSERT(m >= 0);
        ASSERT(n >= 0);
        ASSERT(lda >= k);
        ASSERT(ldb >= k);
        ASSERT(ldc >= m);
        mnpack(0, m, 0, n);
    }

  private:
    void mnpack(int m0, int m, int n0, int n) {
        if (m - m0 <= 0 || n - n0 <= 0)
            return;
        int mc, nc, mp, np;
        if (m - m0 >= 8 && n - n0 >= 3) {
            mc = 8;
            nc = 3;
            gemm8x3(m0, m, n0, n);
        } else if (m - m0 >= 5 && n - n0 >= 5) {
            mc = 5;
            nc = 5;
            gemm5x5(m0, m, n0, n);
        } else if (m - m0 >= 3 && n - n0 >= 4) {
            mc = 3;
            nc = 4;
            gemm3x4(m0, m, n0, n);
        } else if (m - m0 >= 4 && n - n0 >= 1) {
            mc = 4;
            nc = 1;
            gemm4x1(m0, m, n0, n);
        } else if (m - m0 >= 1 && n - n0 >= 4) {
            mc = 1;
            nc = 4;
            gemm1x4(m0, m, n0, n);
        } else {
            mc = 1;
            nc = 1;
            gemm1x1(m0, m, n0, n);
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, mp, np, n);
        mnpack(mp, m, np, n);
    }

    dontinline void gemm8x3(int m0, int m, int n0, int n) {
        BEGIN_KERNEL(8, 3)
        V c00 = {0};
        V c01 = {0};
        V c02 = {0};
        V c10 = {0};
        V c11 = {0};
        V c12 = {0};
        V c20 = {0};
        V c21 = {0};
        V c22 = {0};
        V c30 = {0};
        V c31 = {0};
        V c32 = {0};
        V c40 = {0};
        V c41 = {0};
        V c42 = {0};
        V c50 = {0};
        V c51 = {0};
        V c52 = {0};
        V c60 = {0};
        V c61 = {0};
        V c62 = {0};
        V c70 = {0};
        V c71 = {0};
        V c72 = {0};
        for (int l = 0; l < k; l += KN) {
            V k0 = load(B + ldb * (j + 0) + l);
            V k1 = load(B + ldb * (j + 1) + l);
            V k2 = load(B + ldb * (j + 2) + l);
            V a0 = load(A + lda * (i + 0) + l);
            c00 = madd(a0, k0, c00);
            c01 = madd(a0, k1, c01);
            c02 = madd(a0, k2, c02);
            V a1 = load(A + lda * (i + 1) + l);
            c10 = madd(a1, k0, c10);
            c11 = madd(a1, k1, c11);
            c12 = madd(a1, k2, c12);
            V a2 = load(A + lda * (i + 2) + l);
            c20 = madd(a2, k0, c20);
            c21 = madd(a2, k1, c21);
            c22 = madd(a2, k2, c22);
            V a3 = load(A + lda * (i + 3) + l);
            c30 = madd(a3, k0, c30);
            c31 = madd(a3, k1, c31);
            c32 = madd(a3, k2, c32);
            V a4 = load(A + lda * (i + 4) + l);
            c40 = madd(a4, k0, c40);
            c41 = madd(a4, k1, c41);
            c42 = madd(a4, k2, c42);
            V a5 = load(A + lda * (i + 5) + l);
            c50 = madd(a5, k0, c50);
            c51 = madd(a5, k1, c51);
            c52 = madd(a5, k2, c52);
            V a6 = load(A + lda * (i + 6) + l);
            c60 = madd(a6, k0, c60);
            c61 = madd(a6, k1, c61);
            c62 = madd(a6, k2, c62);
            V a7 = load(A + lda * (i + 7) + l);
            c70 = madd(a7, k0, c70);
            c71 = madd(a7, k1, c71);
            c72 = madd(a7, k2, c72);
        }
        C[ldc * (j + 0) + (i + 0)] = hsum(c00);
        C[ldc * (j + 0) + (i + 1)] = hsum(c10);
        C[ldc * (j + 0) + (i + 2)] = hsum(c20);
        C[ldc * (j + 0) + (i + 3)] = hsum(c30);
        C[ldc * (j + 0) + (i + 4)] = hsum(c40);
        C[ldc * (j + 0) + (i + 5)] = hsum(c50);
        C[ldc * (j + 0) + (i + 6)] = hsum(c60);
        C[ldc * (j + 0) + (i + 7)] = hsum(c70);
        C[ldc * (j + 1) + (i + 0)] = hsum(c01);
        C[ldc * (j + 1) + (i + 1)] = hsum(c11);
        C[ldc * (j + 1) + (i + 2)] = hsum(c21);
        C[ldc * (j + 1) + (i + 3)] = hsum(c31);
        C[ldc * (j + 1) + (i + 4)] = hsum(c41);
        C[ldc * (j + 1) + (i + 5)] = hsum(c51);
        C[ldc * (j + 1) + (i + 6)] = hsum(c61);
        C[ldc * (j + 1) + (i + 7)] = hsum(c71);
        C[ldc * (j + 2) + (i + 0)] = hsum(c02);
        C[ldc * (j + 2) + (i + 1)] = hsum(c12);
        C[ldc * (j + 2) + (i + 2)] = hsum(c22);
        C[ldc * (j + 2) + (i + 3)] = hsum(c32);
        C[ldc * (j + 2) + (i + 4)] = hsum(c42);
        C[ldc * (j + 2) + (i + 5)] = hsum(c52);
        C[ldc * (j + 2) + (i + 6)] = hsum(c62);
        C[ldc * (j + 2) + (i + 7)] = hsum(c72);
        END_KERNEL()
    }

    dontinline void gemm5x5(int m0, int m, int n0, int n) {
        BEGIN_KERNEL(5, 5)
        V c00 = zero();
        V c01 = zero();
        V c02 = zero();
        V c03 = zero();
        V c04 = zero();
        V c10 = zero();
        V c11 = zero();
        V c12 = zero();
        V c13 = zero();
        V c14 = zero();
        V c20 = zero();
        V c21 = zero();
        V c22 = zero();
        V c23 = zero();
        V c24 = zero();
        V c30 = zero();
        V c31 = zero();
        V c32 = zero();
        V c33 = zero();
        V c34 = zero();
        V c40 = zero();
        V c41 = zero();
        V c42 = zero();
        V c43 = zero();
        V c44 = zero();
        for (int l = 0; l < k; l += KN) {
            BEGIN_CRITICAL(i);
            V k0 = load(B + ldb * (j + 0) + l);
            V k1 = load(B + ldb * (j + 1) + l);
            V k2 = load(B + ldb * (j + 2) + l);
            V k3 = load(B + ldb * (j + 3) + l);
            V k4 = load(B + ldb * (j + 4) + l);
            V a0 = load(A + lda * (i + 0) + l);
            c00 = madd(a0, k0, c00);
            c01 = madd(a0, k1, c01);
            c02 = madd(a0, k2, c02);
            c03 = madd(a0, k3, c03);
            c04 = madd(a0, k4, c04);
            V a1 = load(A + lda * (i + 1) + l);
            c10 = madd(a1, k0, c10);
            c11 = madd(a1, k1, c11);
            c12 = madd(a1, k2, c12);
            c13 = madd(a1, k3, c13);
            c14 = madd(a1, k4, c14);
            V a2 = load(A + lda * (i + 2) + l);
            c20 = madd(a2, k0, c20);
            c21 = madd(a2, k1, c21);
            c22 = madd(a2, k2, c22);
            c23 = madd(a2, k3, c23);
            c24 = madd(a2, k4, c24);
            V a3 = load(A + lda * (i + 3) + l);
            c30 = madd(a3, k0, c30);
            c31 = madd(a3, k1, c31);
            c32 = madd(a3, k2, c32);
            c33 = madd(a3, k3, c33);
            c34 = madd(a3, k4, c34);
            V a4 = load(A + lda * (i + 4) + l);
            c40 = madd(a4, k0, c40);
            c41 = madd(a4, k1, c41);
            c42 = madd(a4, k2, c42);
            c43 = madd(a4, k3, c43);
            c44 = madd(a4, k4, c44);
            END_CRITICAL(i);
        }
        C[ldc * (j + 0) + (i + 0)] = hsum(c00);
        C[ldc * (j + 0) + (i + 1)] = hsum(c10);
        C[ldc * (j + 0) + (i + 2)] = hsum(c20);
        C[ldc * (j + 0) + (i + 3)] = hsum(c30);
        C[ldc * (j + 0) + (i + 4)] = hsum(c40);
        C[ldc * (j + 1) + (i + 0)] = hsum(c01);
        C[ldc * (j + 1) + (i + 1)] = hsum(c11);
        C[ldc * (j + 1) + (i + 2)] = hsum(c21);
        C[ldc * (j + 1) + (i + 3)] = hsum(c31);
        C[ldc * (j + 1) + (i + 4)] = hsum(c41);
        C[ldc * (j + 2) + (i + 0)] = hsum(c02);
        C[ldc * (j + 2) + (i + 1)] = hsum(c12);
        C[ldc * (j + 2) + (i + 2)] = hsum(c22);
        C[ldc * (j + 2) + (i + 3)] = hsum(c32);
        C[ldc * (j + 2) + (i + 4)] = hsum(c42);
        C[ldc * (j + 3) + (i + 0)] = hsum(c03);
        C[ldc * (j + 3) + (i + 1)] = hsum(c13);
        C[ldc * (j + 3) + (i + 2)] = hsum(c23);
        C[ldc * (j + 3) + (i + 3)] = hsum(c33);
        C[ldc * (j + 3) + (i + 4)] = hsum(c43);
        C[ldc * (j + 4) + (i + 0)] = hsum(c04);
        C[ldc * (j + 4) + (i + 1)] = hsum(c14);
        C[ldc * (j + 4) + (i + 2)] = hsum(c24);
        C[ldc * (j + 4) + (i + 3)] = hsum(c34);
        C[ldc * (j + 4) + (i + 4)] = hsum(c44);
        END_KERNEL()
    }

    NOINLINE void gemm3x4(int m0, int m, int n0, int n) {
        BEGIN_KERNEL(3, 4)
        V c00 = zero();
        V c01 = zero();
        V c02 = zero();
        V c03 = zero();
        V c10 = zero();
        V c11 = zero();
        V c12 = zero();
        V c13 = zero();
        V c20 = zero();
        V c21 = zero();
        V c22 = zero();
        V c23 = zero();
        for (int l = 0; l < k; l += KN) {
            BEGIN_CRITICAL(i);
            V k0 = load(B + ldb * (j + 0) + l);
            V k1 = load(B + ldb * (j + 1) + l);
            V k2 = load(B + ldb * (j + 2) + l);
            V k3 = load(B + ldb * (j + 3) + l);
            V a0 = load(A + lda * (i + 0) + l);
            c00 = madd(a0, k0, c00);
            c01 = madd(a0, k1, c01);
            c02 = madd(a0, k2, c02);
            c03 = madd(a0, k3, c03);
            V a1 = load(A + lda * (i + 1) + l);
            c10 = madd(a1, k0, c10);
            c11 = madd(a1, k1, c11);
            c12 = madd(a1, k2, c12);
            c13 = madd(a1, k3, c13);
            V a2 = load(A + lda * (i + 2) + l);
            c20 = madd(a2, k0, c20);
            c21 = madd(a2, k1, c21);
            c22 = madd(a2, k2, c22);
            c23 = madd(a2, k3, c23);
            END_CRITICAL(i);
        }
        C[ldc * (j + 0) + (i + 0)] = hsum(c00);
        C[ldc * (j + 0) + (i + 1)] = hsum(c10);
        C[ldc * (j + 0) + (i + 2)] = hsum(c20);
        C[ldc * (j + 1) + (i + 0)] = hsum(c01);
        C[ldc * (j + 1) + (i + 1)] = hsum(c11);
        C[ldc * (j + 1) + (i + 2)] = hsum(c21);
        C[ldc * (j + 2) + (i + 0)] = hsum(c02);
        C[ldc * (j + 2) + (i + 1)] = hsum(c12);
        C[ldc * (j + 2) + (i + 2)] = hsum(c22);
        C[ldc * (j + 3) + (i + 0)] = hsum(c03);
        C[ldc * (j + 3) + (i + 1)] = hsum(c13);
        C[ldc * (j + 3) + (i + 2)] = hsum(c23);
        END_KERNEL()
    }

    NOINLINE void gemm1x4(int m0, int m, int n0, int n) {
        BEGIN_KERNEL(1, 4)
        V c00 = zero();
        V c01 = zero();
        V c02 = zero();
        V c03 = zero();
        for (int l = 0; l < k; l += KN) {
            V a0 = load(A + lda * (i + 0) + l);
            V k0 = load(B + ldb * (j + 0) + l);
            V k1 = load(B + ldb * (j + 1) + l);
            V k2 = load(B + ldb * (j + 2) + l);
            V k3 = load(B + ldb * (j + 3) + l);
            c00 = madd(a0, k0, c00);
            c01 = madd(a0, k1, c01);
            c02 = madd(a0, k2, c02);
            c03 = madd(a0, k3, c03);
        }
        C[ldc * (j + 0) + (i + 0)] = hsum(c00);
        C[ldc * (j + 1) + (i + 0)] = hsum(c01);
        C[ldc * (j + 2) + (i + 0)] = hsum(c02);
        C[ldc * (j + 3) + (i + 0)] = hsum(c03);
        END_KERNEL()
    }

    NOINLINE void gemm4x1(int m0, int m, int n0, int n) {
        constexpr int KC = VECTOR_REGISTERS / 6;
        BEGIN_KERNEL(4, 1)
        if (!(k % (KN * KC))) {
            V c00[KC] = {0};
            V c10[KC] = {0};
            V c20[KC] = {0};
            V c30[KC] = {0};
            for (int l = 0; l < k; l += KN * KC) {
                for (int z = 0; z < KC; ++z) {
                    V k0 = load(B + ldb * (j + 0) + (l + KN * z));
                    V a0 = load(A + lda * (i + 0) + (l + KN * z));
                    c00[z] = madd(a0, k0, c00[z]);
                    V a1 = load(A + lda * (i + 1) + (l + KN * z));
                    c10[z] = madd(a1, k0, c10[z]);
                    V a2 = load(A + lda * (i + 2) + (l + KN * z));
                    c20[z] = madd(a2, k0, c20[z]);
                    V a3 = load(A + lda * (i + 3) + (l + KN * z));
                    c30[z] = madd(a3, k0, c30[z]);
                }
            }
            C[ldc * (j + 0) + (i + 0)] = hsums(c00, KC);
            C[ldc * (j + 0) + (i + 1)] = hsums(c10, KC);
            C[ldc * (j + 0) + (i + 2)] = hsums(c20, KC);
            C[ldc * (j + 0) + (i + 3)] = hsums(c30, KC);
        } else {
            V c00 = zero();
            V c10 = zero();
            V c20 = zero();
            V c30 = zero();
            for (int l = 0; l < k; l += KN) {
                V k0 = load(B + ldb * (j + 0) + l);
                V a0 = load(A + lda * (i + 0) + l);
                c00 = madd(a0, k0, c00);
                V a1 = load(A + lda * (i + 1) + l);
                c10 = madd(a1, k0, c10);
                V a2 = load(A + lda * (i + 2) + l);
                c20 = madd(a2, k0, c20);
                V a3 = load(A + lda * (i + 3) + l);
                c30 = madd(a3, k0, c30);
            }
            C[ldc * (j + 0) + (i + 0)] = hsum(c00);
            C[ldc * (j + 0) + (i + 1)] = hsum(c10);
            C[ldc * (j + 0) + (i + 2)] = hsum(c20);
            C[ldc * (j + 0) + (i + 3)] = hsum(c30);
        }
        END_KERNEL()
    }

    NOINLINE void gemm1x1(int m0, int m, int n0, int n) {
        BEGIN_KERNEL(1, 1)
        V c = zero();
        for (int l = 0; l < k; l += KN) {
            c = madd(load(A + lda * i + l), load(B + ldb * j + l), c);
        }
        C[ldc * j + i] = hsum(c);
        END_KERNEL()
    }

    typedef float32x4_t V;
    static constexpr int KN = sizeof(V) / sizeof(float);
    static inline float32x4_t zero() {
        return vdupq_n_f32(0.0f);
    }
    static inline float32x4_t load(const float *p) {
        return vld1q_f32(p);
    }
    static inline float hsum(float32x4_t x) {
        return vaddvq_f32(x);
    }
    static inline float32x4_t madd(float32x4_t x, float32x4_t y, float32x4_t z) {
        return vfmaq_f32(z, x, y);
        // return vaddq_f32(vmulq_f32(x, y), z);
    }

    template <typename T> float hsums(const T *x, int n) {
        float sum = 0;
        for (int i = 0; i < n; ++i)
            sum += hsum(x[i]);
        return sum;
    }

    const int k;
    const float *const A;
    const int lda;
    const float *const B;
    const int ldb;
    float *const C;
    const int ldc;
    const int ith;
    const int nth;
};

void sgemm(int m, int n, int k, const float *A, int lda, const float *B, int ldb, float *C, int ldc,
           int ith, int nth) {
    if (nth) {
        GEMMER tb{k, A, lda, B, ldb, C, ldc, ith, nth};
        tb.gemm(m, n);
    } else if (!HAVE_OPENMP || n * m * k < THRESHOLD) {
        GEMMER tb{k, A, lda, B, ldb, C, ldc, 0, 1};
        tb.gemm(m, n);
    } else {
        ASSERT((nth = sysconf(_SC_NPROCESSORS_ONLN)) > 0);
#pragma omp parallel for
        for (ith = 0; ith < nth; ++ith) {
            GEMMER tb{k, A, lda, B, ldb, C, ldc, ith, nth};
            tb.gemm(m, n);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

void test_gemm(int m, int n, int k) {
    int lda = k;
    int ldb = k;
    int ldc = m;
    int ldg = m;
    float *A = new float[m * k];
    float *B = new float[n * k];
    float *C = new float[n * m];
    float *G = new float[n * m];
    fill(m, k, A, lda);
    fill(n, k, B, ldb);
    dgemm(m, n, k, 1, A, lda, B, ldb, 0, G, ldg);
    broadcast(n, m, C, ldc, 666.f);
    sgemm(m, n, k, A, lda, B, ldb, C, ldc, 0, 0);
    check(PRECISION, n, m, G, ldg, C, ldc);
    delete[] G;
    delete[] C;
    delete[] B;
    delete[] A;
}

void check_works(void) {
    static int kSizes[] = {5, 6, 7, 8, 9,  15, 1,  0,  2,  3,  4,  5,
                           6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64};
    std::atomic_llong t = 0;
    int N = sizeof(kSizes) / sizeof(kSizes[0]);
    int start = micros();
#pragma omp parallel for collapse(3)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                char name[128];
                int m = kSizes[i];
                int n = kSizes[N - 1 - i];
                int K = (kSizes[i] + 7) / 8 * 8;
                snprintf(name, sizeof(name), "testing %2lld %2lld %2lld", m, n, K);
                if (t++ % 13 == 0)
                    fprintf(stderr, "%s\r", name);
                is_self_testing = name;
                test_gemm(m, n, K);
                is_self_testing = 0;
            }
        }
    }
    int end = micros();
    fprintf(stderr, "\rself test completed successfully in %lld ms\n", (end - start) / 1000);
}

////////////////////////////////////////////////////////////////////////////////

void check_sgemm(void) {
    int m = 512;
    int n = 1;
    int k = 16384;
    printf("m=%lld n=%lld k=%lld\n", m, n, k);
    float *A = new float[m * k];
    float *B = new float[n * k];
    float *C = new float[n * m];
    float *Ct = new float[n * m];
    float *G = new float[n * m];
    fill(m, k, A, k);
    fill(n, k, B, k);
    bench(dgemm(m, n, k, 1, A, k, B, k, 0, G, m));
    bench(sgemm(m, n, k, A, k, B, k, C, m, 0, 0));
    check(PRECISION, n, m, G, m, C, m);
    delete[] G;
    delete[] Ct;
    delete[] C;
    delete[] B;
    delete[] A;
}

int main(int argc, char *argv[]) {
    run(check_works());
    run(check_sgemm());
}

#else
int main(int argc, char *argv[]) {
}
#endif
