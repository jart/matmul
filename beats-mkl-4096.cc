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

#define MKL 1
#define TEST 1
#define YOLO 0

#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wignored-attributes"

#include "linalg.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cinttypes>
#include <climits>
#include <immintrin.h>
#if MKL
#include <mkl_blas.h>
#endif

#if YOLO
#define PRECISION 1e-4
#else
#define PRECISION 1e-5
#endif

#if defined(__ARM_NEON) || defined(__AVX512F__)
#define VECTOR_REGISTERS 32
#else
#define VECTOR_REGISTERS 16
#endif

#if defined(__AVX512F__)
#define VECTOR_WIDTH 64
#elif defined(__AVX__) || defined(__AVX2__)
#define VECTOR_WIDTH 32
#else
#define VECTOR_WIDTH 16
#endif

namespace {

std::atomic_int gemms;

inline void atomicAdd(float *p, float f) {
#if YOLO
    *p += f;
#else
    union {
        float f;
        unsigned i;
    } desired, expected = {*p};
    do
        desired.f = expected.f + f;
    while (!std::atomic_compare_exchange_weak_explicit(
        reinterpret_cast<std::atomic_uint *>(p), &expected.i, desired.i, std::memory_order_relaxed,
        std::memory_order_relaxed));
#endif
}

template <int KN, typename T, typename TA, typename TB, typename TC> struct tinyBLAS {

    tinyBLAS(const TA *A, i64 lda, const TB *B, i64 ldb, TC *C, i64 ldc, int ith, int nth)
        : A(A), lda(lda), B(B), ldb(ldb), C(C), ldc(ldc), ith(ith), nth(nth) {
    }

    void matmul(int m, int n, int k) {
        mnpack(0, m, 0, n, k);
    }

    void mnpack(int m0, int m, int n0, int n, int k) {
        int mc, nc, mp, np;
        if (m - m0 <= 0 || n - n0 <= 0)
            return;
        if (VECTOR_REGISTERS >= 32 && m - m0 >= 8 && n - n0 >= 3) {
            mc = 8;
            nc = 3;
            gemm<8, 3>(m0, m, n0, n, k);
        } else if (m - m0 >= 4 && n - n0 >= 3) {
            mc = 4;
            nc = 3;
            gemm<4, 3>(m0, m, n0, n, k);
        } else if (n - n0 >= 9) {
            mc = 1;
            nc = 9;
            gemm<1, 9>(m0, m, n0, n, k);
        } else if (n - n0 >= 4) {
            mc = 1;
            nc = 4;
            gemm<1, 4>(m0, m, n0, n, k);
        } else if (m - m0 >= 4) {
            mc = 4;
            nc = 1;
            gemm<4, 1>(m0, m, n0, n, k);
        } else {
            mc = 1;
            nc = 1;
            gemm<1, 1>(m0, m, n0, n, k);
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np, k);
        mnpack(m0, mp, np, n, k);
        mnpack(mp, m, np, n, k);
    }

    template <int RM, int RN> void gemm(int m0, int m, int n0, int n, int k) {
        gemms++;
        int ytiles = (m - m0) / RM;
        int xtiles = (n - n0) / RN;
        int tiles = xtiles * ytiles;
        int duty = (tiles + nth - 1) / nth;
        int start = duty * ith;
        int end = start + duty;
        if (end > tiles)
            end = tiles;
        for (int job = start; job < end; ++job) {
            int ii = m0 + job / xtiles * RM;
            int jj = n0 + job % xtiles * RN;
            T Cv[RN][RM] = {0};
            for (int l = 0; l < k; l += KN)
                for (int j = 0; j < RN; ++j)
                    for (int i = 0; i < RM; ++i)
                        Cv[j][i] = madd(load(A + lda * (ii + i) + l), //
                                        load(B + ldb * (jj + j) + l), //
                                        Cv[j][i]);
            TC Cd[RN][RM];
            for (int j = 0; j < RN; ++j)
                for (int i = 0; i < RM; ++i)
                    Cd[j][i] = hsum(Cv[j][i]);
            for (int j = 0; j < RN; ++j)
                for (int i = 0; i < RM; ++i)
                    C[ldc * (jj + j) + (ii + i)] = Cd[j][i];
        }
    }

    inline __m512 load(const float *p) {
        return _mm512_loadu_ps((void *)p);
        // return (__m512)_mm512_stream_load_si512((void *)p);
    }

    inline __m512 madd(__m512 a, __m512 b, __m512 c) {
        return _mm512_add_ps(_mm512_mul_ps(a, b), c);
    }

    inline float hsum(__m512 x) {
        return _mm512_reduce_add_ps(x);
    }

    const TA *const A;
    const i64 lda;
    const TB *const B;
    const i64 ldb;
    TC *const C;
    const i64 ldc;
    const int ith;
    const int nth;
};

NOINLINE void sgemm(int m, int n, int k, const float *A, int lda, const float *B, int ldb, float *C,
                    int ldc) {
    int nth = sysconf(_SC_NPROCESSORS_ONLN);
#pragma omp parallel for
    for (int ith = 0; ith < nth; ++ith) {
        tinyBLAS<16, __m512, float, float, float> tb{A, lda, B, ldb, C, ldc, ith, nth};
        tb.matmul(m, n, k);
    }
}

} // namespace

void test_gemm(i64 m, i64 n, i64 k) {
    i64 lda, ldb, ldc, ldg;
    float *A = new_matrix<float>(m, k, &lda);
    float *B = new_matrix<float>(n, k, &ldb);
    float *C = new_matrix<float>(n, m, &ldc);
    float *G = new_matrix<float>(n, m, &ldg);
    fill(m, k, A, lda);
    fill(n, k, B, ldb);
    // broadcast(n, m, C, ldc, 666.f);
    sgemm(m, n, k, A, lda, B, ldb, C, ldc);
    dgemm(m, n, k, 1, A, lda, B, ldb, 0, G, ldg);
    check(PRECISION, n, m, G, ldg, C, ldc);
    delete_matrix(G);
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
                i64 K = (kSizes[i] + 15) / 16 * 16;
                snprintf(name, sizeof(name), "testing %2lld %2lld %2lld", m, n, K);
                if (t++ % 13 == 0)
                    fprintf(stderr, "%s\r", name);
                is_self_testing = name;
                test_gemm(m, n, K);
                is_self_testing = 0;
            }
        }
    }
    i64 end = micros();
    fprintf(stderr, "\rself test completed successfully in %lld ms\n", (end - start) / 1000);
}

i64 m = 4095;
i64 n = 4095;
i64 k = 4096;

void check_sgemm(void) {
    log_mb(m, n, k);
    i64 lda, ldb, ldc, ldg;
    float *A = new_matrix<float>(m, k, &lda);
    float *B = new_matrix<float>(n, k, &ldb);
    float *C = new_matrix<float>(n, m, &ldc);
    float *G = new_matrix<float>(n, m, &ldg);
    fill(m, k, A, lda);
    fill(n, k, B, ldb);
    bench(sgemm(m, n, k, A, lda, B, ldb, C, ldc));
    bench(sgemm(m, n, k, A, lda, B, ldb, C, ldc));
    if (0)
        bench(dgemm(m, n, k, 1, A, lda, B, ldb, 0, G, ldg));
    if (0)
        check(PRECISION, n, m, G, ldg, C, ldc);
    delete_matrix(G);
    delete_matrix(C);
    delete_matrix(B);
    delete_matrix(A);
    printf("%8d gemms\n", (int)gemms);
}

void check_mkl(void) {
#if MKL
    log_mb(m, n, k);
    i64 lda, ldb, ldc, ldg;
    float *A = new_matrix<float>(m, k, &lda);
    float *B = new_matrix<float>(n, k, &ldb);
    float *C = new_matrix<float>(n, m, &ldc);
    float *G = new_matrix<float>(n, m, &ldg);
    float beta = 0;
    float alpha = 1;
    fill(m, k, A, lda);
    fill(n, k, B, ldb);
    bench(SGEMM("T", "N", &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc));
    bench(SGEMM("T", "N", &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc));
    if (0)
        bench(dgemm(m, n, k, 1, A, lda, B, ldb, 0, G, ldg));
    if (0)
        check(PRECISION, n, m, G, ldg, C, ldc);
#endif
}

int main(int argc, char *argv[]) {
#ifdef __SANITIZE_ADDRESS__
    printf("===========================================================\n");
    printf("         IT'S NOT SLOW; IT'S RUNNING IN ASAN MODE\n");
    printf("===========================================================\n");
#endif
    // run(check_works());
    run(check_sgemm());
    run(check_mkl());
}
