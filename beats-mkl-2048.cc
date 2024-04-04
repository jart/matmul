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

#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wignored-attributes"

#include "hsum.h"
#include "linalg.h"
#include "load.h"
#include "varith.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cinttypes>
#include <climits>
#include <immintrin.h>

#if MKL
#include <mkl_blas.h>
#endif

#define PRECISION 1e-5

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

inline void atomicAdd(float *p, float f) {
    union {
        float f;
        unsigned i;
    } desired, expected = {*p};
    do
        desired.f = expected.f + f;
    while (!std::atomic_compare_exchange_weak_explicit(
        reinterpret_cast<std::atomic_uint *>(p), &expected.i, desired.i, std::memory_order_relaxed,
        std::memory_order_relaxed));
}

template <int KN, typename T, typename TA, typename TB, typename TC> struct tinyBLAS {
  public:
    tinyBLAS(const TA *A, i64 lda, const TB *B, i64 ldb, TC *C, i64 ldc, int nth)
        : A(A), lda(lda), B(B), ldb(ldb), C(C), ldc(ldc), nth(nth), barrier(ATOMIC_VAR_INIT(0)) {
    }

    NOINLINE void matmul(i64 m, i64 n, i64 k, int ith) {
        if (!m || !n)
            return;
        zeroify(m, n, ith);
        if (!k)
            return;
        mnpack(0, m, 0, n, k, ith);
    }

  private:
    NOINLINE void zeroify(i64 m, i64 n, int ith) {
        int duty = (n + nth - 1) / nth;
        if (duty < 1)
            duty = 1;
        int start = duty * ith;
        int end = start + duty;
        if (end > n)
            end = n;
        for (int j = start; j < end; ++j)
            memset(C + ldc * j, 0, sizeof(TC) * m);
        syncthreads();
    }

    NOINLINE void mnpack(i64 m0, i64 m, i64 n0, i64 n, i64 k, int ith) {
        int mc, nc, mp, np;
        if (m - m0 <= 0 || n - n0 <= 0)
            return;
        if (m - m0 >= (mc = 5) && n - n0 >= (nc = 5)) {
            mp = m0 + (m - m0) / mc * mc;
            np = n0 + (n - n0) / nc * nc;
            kpack<5, 5>(m0, mp, n0, np, 0, k, ith);
        } else {
            mc = 1;
            nc = 1;
            mp = m0 + (m - m0) / mc * mc;
            np = n0 + (n - n0) / nc * nc;
            kpack<1, 1>(m0, mp, n0, np, 0, k, ith);
        }
        mnpack(mp, m, n0, np, k, ith);
        mnpack(m0, mp, np, n, k, ith);
        mnpack(mp, m, np, n, k, ith);
    }

    template <int MC, int NC>
    NOINLINE void kpack(i64 m0, i64 m, i64 n0, i64 n, i64 k0, i64 k, int ith) {
        int kc, kp;
        if (k - k0 <= 0)
            return;
        constexpr int KC = 128;
        if (k - k0 >= (kc = KC) * KN) {
            kp = k0 + (k - k0) / (kc * KN) * (kc * KN);
            gemm<MC, NC, KC>(m0, m, n0, n, k0, kp, ith);
        } else {
            kc = KN;
            kp = k0 + (k - k0) / kc * kc;
            gemm<MC, NC, 1>(m0, m, n0, n, k0, kp, ith);
        }
        kpack<MC, NC>(m0, m, n0, n, kp, k, ith);
    }

    template <int MC, int NC, int KC>
    NOINLINE void gemm(i64 m0, i64 m, i64 n0, i64 n, i64 k0, i64 k, int ith) {
        i64 ytiles = (m - m0) / MC;
        i64 ztiles = (k - k0) / (KC * KN);
        i64 tiles = ytiles * ztiles;
        i64 duty = (tiles + nth - 1) / nth;
        i64 start = duty * ith;
        i64 end = start + duty;
        if (end > tiles)
            end = tiles;
        for (i64 job = start; job < end; ++job) {
            i64 ii = m0 + job / ztiles * MC;
            i64 ll = k0 + job % ztiles * (KC * KN);
            T Ac[KC][MC];
            for (i64 i = 0; i < MC; ++i)
                for (i64 l = 0; l < KC; ++l)
                    Ac[l][i] = load<T>(A + lda * (ii + i) + (ll + KN * l));
            for (i64 jj = n0; jj < n; jj += NC) {
                T Cc[NC][MC] = {0};
                for (i64 l = 0; l < KC; ++l)
                    for (i64 j = 0; j < NC; ++j) {
                        T b = load<T>(B + ldb * (jj + j) + (ll + KN * l));
                        for (i64 i = 0; i < MC; ++i)
                            Cc[j][i] = madd(Ac[l][i], b, Cc[j][i]);
                    }
                TC Ct[NC][MC];
                for (i64 j = 0; j < NC; ++j)
                    for (i64 i = 0; i < MC; ++i)
                        Ct[j][i] = hsum(Cc[j][i]);
                for (i64 j = 0; j < NC; ++j)
                    for (i64 i = 0; i < MC; ++i)
                        atomicAdd(C + ldc * (jj + j) + (ii + i), Ct[j][i]);
            }
        }
    }

    NOINLINE void syncthreads() {
        if (barrier.fetch_add(1) + 1 == nth) {
            barrier.store(0);
        } else {
            while (barrier.load()) {
            }
        }
    }

    const TA *const A;
    const i64 lda;
    const TB *const B;
    const i64 ldb;
    TC *const C;
    const i64 ldc;
    const int nth;
    std::atomic_int barrier;
};

NOINLINE void sgemm(i64 m, i64 n, i64 k, const float *A, i64 lda, const float *B, i64 ldb, float *C,
                    i64 ldc) {
    int nth = 1;
    if (m * n * k > 64 * 64 * 64)
        nth = sysconf(_SC_NPROCESSORS_ONLN);

#if defined(__AVX512F__)
    if (!(k % 16)) {
        tinyBLAS<16, __m512, float, float, float> tb{A, lda, B, ldb, C, ldc, nth};
#pragma omp parallel for if (nth > 1)
        for (int ith = 0; ith < nth; ++ith)
            tb.matmul(m, n, k, ith);
        return;
    }

#elif defined(__AVX2__)
    if (!(k % 8)) {
        tinyBLAS<8, __m256, float, float, float> tb{A, lda, B, ldb, C, ldc, nth};
#pragma omp parallel for if (nth > 1)
        for (int ith = 0; ith < nth; ++ith)
            tb.matmul(m, n, k, ith);
        return;
    }

#elif defined(__SSE__)
    if (!(k % 4)) {
        tinyBLAS<4, __m128, float, float, float> tb{A, lda, B, ldb, C, ldc, nth};
#pragma omp parallel for if (nth > 1)
        for (int ith = 0; ith < nth; ++ith)
            tb.matmul(m, n, k, ith);
        return;
    }
#endif

    tinyBLAS<1, float, float, float, float> tb{A, lda, B, ldb, C, ldc, nth};
#pragma omp parallel for if (nth > 1)
    for (int ith = 0; ith < nth; ++ith)
        tb.matmul(m, n, k, ith);
}

} // namespace

void test_gemm(i64 m, i64 n, i64 k) {
    i64 lda = k, ldb = k, ldc = m, ldg = m;
    float *A = new float[m * k];
    float *B = new float[n * k];
    float *C = new float[n * m];
    float *G = new float[n * m];
    fill(m, k, A, lda);
    fill(n, k, B, ldb);
    dgemm(m, n, k, 1, A, lda, B, ldb, 0, G, ldg);
    broadcast(n, m, C, ldc, 666.f);
    sgemm(m, n, k, A, lda, B, ldb, C, ldc);
    check(PRECISION, n, m, G, ldg, C, ldc);
    delete[] G;
    delete[] C;
    delete[] B;
    delete[] A;
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

i64 m = 2048;
i64 n = 2048;
i64 k = 2048;

void check_sgemm(void) {
    log_mb(m, n, k);
    i64 lda = k, ldb = k, ldc = m, ldg = m;
    float *A = new float[m * k];
    float *B = new float[n * k];
    float *C = new float[n * m];
    float *G = new float[n * m];
    fill(m, k, A, lda);
    fill(k, n, B, ldb);
    dgemm(m, n, k, 1, A, lda, B, ldb, 0, G, ldg);
    sgemm(m, n, k, A, lda, B, ldb, C, ldc);
    bench(sgemm(m, n, k, A, lda, B, ldb, C, ldc));
    check(1e-3, n, m, G, ldg, C, ldc);
    delete[] G;
    delete[] C;
    delete[] B;
    delete[] A;
}

void check_mkl(void) {
#if MKL
    log_mb(m, n, k);
    i64 lda = k, ldb = k, ldc = m, ldg = m;
    float *A = new float[m * k];
    float *B = new float[n * k];
    float *C = new float[n * m];
    float *G = new float[n * m];
    float beta = 0;
    float alpha = 1;
    fill(m, k, A, lda);
    fill(k, n, B, ldb);
    dgemm(m, n, k, 1, A, lda, B, ldb, 0, G, ldg);
    SGEMM("T", "N", &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
    bench(SGEMM("T", "N", &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc));
    check(PRECISION, n, m, G, ldg, C, ldc);
#else
    printf("put `#define MKL 1` in the source code for MKL\n");
#endif
}

int main(int argc, char *argv[]) {
#ifdef __SANITIZE_ADDRESS__
    printf("===========================================================\n");
    printf("         IT'S NOT SLOW; IT'S RUNNING IN ASAN MODE\n");
    printf("===========================================================\n");
#endif
    run(check_works());
    run(check_sgemm());
    run(check_mkl());
}
