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

#define MKL 0
#define BLIS 0
#define ATLAS 0

#include "linalg.h"
#include <immintrin.h>
#if MKL
#include <mkl_blas.h>
#endif
#if BLIS
#include <blis.h>
#endif
#if ATLAS
extern "C" {
#include <cblas-atlas.h>
}
#endif

#define PRECISION 8e-5

#define BEGIN_KERNEL(RM, RN)                                                   \
    i64 ytiles = (m - m0) / RM;                                                \
    i64 xtiles = (n - n0) / RN;                                                \
    i64 tiles = ytiles * xtiles;                                               \
    double duty = (double)tiles / nth;                                         \
    if (duty < 1)                                                              \
        duty = 1;                                                              \
    double spot = duty * ith + .5;                                             \
    i64 end = spot + duty;                                                     \
    i64 start = spot;                                                          \
    if (end > tiles)                                                           \
        end = tiles;                                                           \
    for (i64 job = start; job < end; ++job) {                                  \
        i64 i = m0 + job / xtiles * RM;                                        \
        i64 j = n0 + job % xtiles * RN;

#define END_KERNEL() }

typedef long long i64;
typedef unsigned long long u64;

class tinyBLAS_f32 {
  public:
    tinyBLAS_f32(i64 k, const float *A, i64 lda, const float *B, i64 ldb,
                 float *C, i64 ldc, i64 ith, i64 nth)
        : k(k), A(A), lda(lda), B(B), ldb(ldb), C(C), ldc(ldc), ith(ith),
          nth(nth) {
        ASSERT(A != nullptr);
        ASSERT(B != nullptr);
        ASSERT(C != nullptr);
        ASSERT(k >= 0 && k % 8 == 0);
        ASSERT(ith >= 0 && ith < nth);
    }

    void gemm(i64 m, i64 n) {
        ASSERT(m >= 0);
        ASSERT(n >= 0);
        ASSERT(lda >= k);
        ASSERT(ldb >= k);
        ASSERT(ldc >= m);
        mnpack(0, m, 0, n);
    }

  private:
    void mnpack(i64 m0, i64 m, i64 n0, i64 n) {
        if (m - m0 <= 0 || n - n0 <= 0)
            return;
        i64 mc, nc, mp, np;
        if (m - m0 >= 3 && n - n0 >= 4) {
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

    NOINLINE void gemm3x4(i64 m0, i64 m, i64 n0, i64 n) {
        BEGIN_KERNEL(3, 4)
        __m256 c00 = _mm256_setzero_ps();
        __m256 c01 = _mm256_setzero_ps();
        __m256 c02 = _mm256_setzero_ps();
        __m256 c03 = _mm256_setzero_ps();
        __m256 c10 = _mm256_setzero_ps();
        __m256 c11 = _mm256_setzero_ps();
        __m256 c12 = _mm256_setzero_ps();
        __m256 c13 = _mm256_setzero_ps();
        __m256 c20 = _mm256_setzero_ps();
        __m256 c21 = _mm256_setzero_ps();
        __m256 c22 = _mm256_setzero_ps();
        __m256 c23 = _mm256_setzero_ps();
        for (i64 l = 0; l < k; l += 8) {
            __m256 k0 = _mm256_loadu_ps(B + ldb * (j + 0) + l);
            __m256 k1 = _mm256_loadu_ps(B + ldb * (j + 1) + l);
            __m256 k2 = _mm256_loadu_ps(B + ldb * (j + 2) + l);
            __m256 k3 = _mm256_loadu_ps(B + ldb * (j + 3) + l);
            __m256 a0 = _mm256_loadu_ps(A + lda * (i + 0) + l);
            c00 = _mm256_fmadd_ps(a0, k0, c00);
            c01 = _mm256_fmadd_ps(a0, k1, c01);
            c02 = _mm256_fmadd_ps(a0, k2, c02);
            c03 = _mm256_fmadd_ps(a0, k3, c03);
            __m256 a1 = _mm256_loadu_ps(A + lda * (i + 1) + l);
            c10 = _mm256_fmadd_ps(a1, k0, c10);
            c11 = _mm256_fmadd_ps(a1, k1, c11);
            c12 = _mm256_fmadd_ps(a1, k2, c12);
            c13 = _mm256_fmadd_ps(a1, k3, c13);
            __m256 a2 = _mm256_loadu_ps(A + lda * (i + 2) + l);
            c20 = _mm256_fmadd_ps(a2, k0, c20);
            c21 = _mm256_fmadd_ps(a2, k1, c21);
            c22 = _mm256_fmadd_ps(a2, k2, c22);
            c23 = _mm256_fmadd_ps(a2, k3, c23);
        }
        C[ldc * (j + 0) + (i + 0)] = hsum_float_8(c00);
        C[ldc * (j + 0) + (i + 1)] = hsum_float_8(c10);
        C[ldc * (j + 0) + (i + 2)] = hsum_float_8(c20);
        C[ldc * (j + 1) + (i + 0)] = hsum_float_8(c01);
        C[ldc * (j + 1) + (i + 1)] = hsum_float_8(c11);
        C[ldc * (j + 1) + (i + 2)] = hsum_float_8(c21);
        C[ldc * (j + 2) + (i + 0)] = hsum_float_8(c02);
        C[ldc * (j + 2) + (i + 1)] = hsum_float_8(c12);
        C[ldc * (j + 2) + (i + 2)] = hsum_float_8(c22);
        C[ldc * (j + 3) + (i + 0)] = hsum_float_8(c03);
        C[ldc * (j + 3) + (i + 1)] = hsum_float_8(c13);
        C[ldc * (j + 3) + (i + 2)] = hsum_float_8(c23);
        END_KERNEL()
    }

    NOINLINE void gemm1x4(i64 m0, i64 m, i64 n0, i64 n) {
        BEGIN_KERNEL(1, 4)
        __m256 c00 = _mm256_setzero_ps();
        __m256 c01 = _mm256_setzero_ps();
        __m256 c02 = _mm256_setzero_ps();
        __m256 c03 = _mm256_setzero_ps();
        for (i64 l = 0; l < k; l += 8) {
            __m256 a0 = _mm256_loadu_ps(A + lda * (i + 0) + l);
            __m256 k0 = _mm256_loadu_ps(B + ldb * (j + 0) + l);
            __m256 k1 = _mm256_loadu_ps(B + ldb * (j + 1) + l);
            __m256 k2 = _mm256_loadu_ps(B + ldb * (j + 2) + l);
            __m256 k3 = _mm256_loadu_ps(B + ldb * (j + 3) + l);
            c00 = _mm256_fmadd_ps(a0, k0, c00);
            c01 = _mm256_fmadd_ps(a0, k1, c01);
            c02 = _mm256_fmadd_ps(a0, k2, c02);
            c03 = _mm256_fmadd_ps(a0, k3, c03);
        }
        C[ldc * (j + 0) + (i + 0)] = hsum_float_8(c00);
        C[ldc * (j + 1) + (i + 0)] = hsum_float_8(c01);
        C[ldc * (j + 2) + (i + 0)] = hsum_float_8(c02);
        C[ldc * (j + 3) + (i + 0)] = hsum_float_8(c03);
        END_KERNEL()
    }

    NOINLINE void gemm4x1(i64 m0, i64 m, i64 n0, i64 n) {
        BEGIN_KERNEL(4, 1)
        __m256 c00 = _mm256_setzero_ps();
        __m256 c10 = _mm256_setzero_ps();
        __m256 c20 = _mm256_setzero_ps();
        __m256 c30 = _mm256_setzero_ps();
        for (i64 l = 0; l < k; l += 8) {
            __m256 k0 = _mm256_loadu_ps(B + ldb * (j + 0) + l);
            __m256 a0 = _mm256_loadu_ps(A + lda * (i + 0) + l);
            c00 = _mm256_fmadd_ps(a0, k0, c00);
            __m256 a1 = _mm256_loadu_ps(A + lda * (i + 1) + l);
            c10 = _mm256_fmadd_ps(a1, k0, c10);
            __m256 a2 = _mm256_loadu_ps(A + lda * (i + 2) + l);
            c20 = _mm256_fmadd_ps(a2, k0, c20);
            __m256 a3 = _mm256_loadu_ps(A + lda * (i + 3) + l);
            c30 = _mm256_fmadd_ps(a3, k0, c30);
        }
        C[ldc * (j + 0) + (i + 0)] = hsum_float_8(c00);
        C[ldc * (j + 0) + (i + 1)] = hsum_float_8(c10);
        C[ldc * (j + 0) + (i + 2)] = hsum_float_8(c20);
        C[ldc * (j + 0) + (i + 3)] = hsum_float_8(c30);
        END_KERNEL()
    }

    NOINLINE void gemm1x1(i64 m0, i64 m, i64 n0, i64 n) {
        BEGIN_KERNEL(1, 1)
        __m256 c = _mm256_setzero_ps();
        for (i64 l = 0; l < k; l += 8) {
            c = _mm256_fmadd_ps(_mm256_loadu_ps(A + lda * i + l),
                                _mm256_loadu_ps(B + ldb * j + l), c);
        }
        C[ldc * j + i] = hsum_float_8(c);
        END_KERNEL()
    }

    NOINLINE static float hsum_float_8(__m256 x) {
        __m128 res = _mm256_extractf128_ps(x, 1);
        res = _mm_add_ps(res, _mm256_castps256_ps128(x));
        res = _mm_add_ps(res, _mm_movehl_ps(res, res));
        res = _mm_add_ss(res, _mm_movehdup_ps(res));
        return _mm_cvtss_f32(res);
    }

    const i64 k;
    const float *const A;
    const i64 lda;
    const float *const B;
    const i64 ldb;
    float *const C;
    const i64 ldc;
    const i64 ith;
    const i64 nth;
};

void sgemm(i64 m, i64 n, i64 k, const float *A, i64 lda, const float *B,
           i64 ldb, float *C, i64 ldc, i64 ith, i64 nth) {
    if (nth) {
        tinyBLAS_f32 tb{k, A, lda, B, ldb, C, ldc, ith, nth};
        tb.gemm(m, n);
    } else if (!HAVE_OPENMP || n * m * k < THRESHOLD) {
        tinyBLAS_f32 tb{k, A, lda, B, ldb, C, ldc, 0, 1};
        tb.gemm(m, n);
    } else {
        ASSERT((nth = sysconf(_SC_NPROCESSORS_ONLN)) > 0);
#pragma omp parallel for
        for (ith = 0; ith < nth; ++ith) {
            tinyBLAS_f32 tb{k, A, lda, B, ldb, C, ldc, ith, nth};
            tb.gemm(m, n);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

void test_gemm(i64 m, i64 n, i64 k) {
    i64 lda, ldb, ldc, ldg;
    float *A = new_matrix<float>(m, k, &lda);
    float *B = new_matrix<float>(n, k, &ldb);
    float *C = new_matrix<float>(n, m, &ldc);
    float *GOLD = new_matrix<float>(n, m, &ldg);
    fill(m, k, A, lda);
    fill(n, k, B, ldb);
    dgemm(m, n, k, 1, A, lda, B, ldb, 0, GOLD, ldg);
    broadcast(n, m, C, ldc, 666.f);
    sgemm(m, n, k, A, lda, B, ldb, C, ldc, 0, 0);
    check(PRECISION, n, m, GOLD, ldg, C, ldc);
    delete_matrix(GOLD);
    delete_matrix(C);
    delete_matrix(B);
    delete_matrix(A);
}

void check_works(void) {
    static i64 kSizes[] = {5, 6, 7, 8, 0,  9,  15, 1,  2,  3,  4,  5,
                           6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64};
    std::atomic_llong t = 0;
    i64 N = sizeof(kSizes) / sizeof(kSizes[0]);
    i64 start = micros();
#pragma omp parallel for collapse(3)
    for (i64 i = 0; i < N; ++i) {
        for (i64 j = 0; j < N; ++j) {
            for (i64 k = 0; k < N; ++k) {
                char name[128];
                i64 m = kSizes[i];
                i64 n = kSizes[N - 1 - i];
                i64 K = (kSizes[i] + 7) / 8 * 8;
                snprintf(name, sizeof(name), "testing %2lld %2lld %2lld", m, n,
                         K);
                if (t++ % 13 == 0)
                    fprintf(stderr, "%s\r", name);
                is_self_testing = name;
                test_gemm(m, n, K);
                is_self_testing = 0;
            }
        }
    }
    i64 end = micros();
    fprintf(stderr, "\rself test completed successfully in %lld ms\n",
            (end - start) / 1000);
}

////////////////////////////////////////////////////////////////////////////////

void check_sgemm(void) {
    // i64 m = 1024;
    // i64 n = 1;
    // i64 k = 4096;
    i64 m = 2333;
    i64 n = 713;
    i64 k = (577 + 7) / 8 * 8;
    // i64 m = 14336;
    // i64 n = 215;
    // i64 k = 4096;
    printf("m=%lld n=%lld k=%lld\n", m, n, k);
    float *A = new_matrix<float>(m, k);
    float *B = new_matrix<float>(n, k);
    float *C = new_matrix<float>(n, m);
    float *Ct = new_matrix<float>(n, m);
    float *G = new_matrix<float>(n, m);
    fill(m, k, A, k);
    fill(n, k, B, k);
    bench(dgemm(m, n, k, 1, A, k, B, k, 0, G, m));
    bench(sgemm(m, n, k, A, k, B, k, C, m, 0, 0));
    check(PRECISION, n, m, G, m, C, m);
#if MKL
    float beta = 0;
    float alpha = 1;
    bench(SGEMM("T", "N", &n, &m, &k, &alpha, B, &k, A, &k, &beta, Ct, &n));
    transpose(m, n, Ct, n, C, m);
    check(PRECISION, n, m, G, m, C, m);
#endif
#if ATLAS
    bench(cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, k, 1, B, k,
                      A, k, 0, C, m));
    check(PRECISION, n, m, G, m, C, m);
#endif
#if BLIS
    float beta = 0;
    float alpha = 1;
    bench(bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_TRANSPOSE, n, m, k, &alpha, B, k, 1,
                    A, k, 1, &beta, C, m, 1));
    check(PRECISION, n, m, G, m, C, m);
#endif
    delete_matrix(G);
    delete_matrix(Ct);
    delete_matrix(C);
    delete_matrix(B);
    delete_matrix(A);
}

int main(int argc, char *argv[]) {
    run(check_works());
    run(check_sgemm());
}
