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

// immintrin perf on threadripper 7995wx (-march=znver4)
//
//     362 µs 10x n=  511 m= 4609 k=  784 multiply_llamafile() 5100.76 gigaflops
//     942 µs 10x n=  511 m= 4609 k=  784 multiply_mkl() 1960.17 gigaflops
//
// ansi f32x16 perf on threadripper 7995wx (-march=znver4)
//
//     972 µs 20x n=  511 m= 4609 k=  784 multiply_ansi() 1899.67 gigaflops
//
// ansi f32x8 perf on threadripper 7995wx (-march=skylake)
//
//    1585 µs 20x n=  511 m= 4609 k=  784 multiply_ansi() 1164.97 gigaflops
//
// ansi f32x4 perf on threadripper 7995wx (-march=k8)
//
//   26767 µs 20x n=  511 m= 4609 k=  784 multiply_ansi() 68.9833 gigaflops
//

//
//                   _   _          ___ _      _   ___
//                  | |_(_)_ _ _  _| _ ) |    /_\ / __|
//                  |  _| | ' \ || | _ \ |__ / _ \\__ \.
//                   \__|_|_||_\_, |___/____/_/ \_\___/
//                             |__/
//
//                    BASIC LINEAR ALGEBRA SUBPROGRAMS
//
//
// This file implements multithreaded CPU matrix multiplication for the
// common contiguous use case C = Aᵀ * B. These kernels are designed to
// have excellent performance[1] for matrices that fit in the CPU cache
// without imposing any overhead such as cache filling or malloc calls.
//
// This implementation does not guarantee any upper bound with rounding
// errors, which grow along with k. Our goal's to maximally exploit the
// hardware for performance, and then use whatever resources remain for
// improving numerical accuracy.
//
// [1] J. Tunney, ‘LLaMA Now Goes Faster on CPUs’, Mar. 2024. [Online].
//     Available: https://justine.lol/matmul/. [Accessed: 29-Mar-2024].

#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wnarrowing"
#pragma GCC diagnostic ignored "-Wmissing-braces"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wignored-attributes"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <errno.h>
#include <string.h>
#include <thread>
#include <unistd.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef __x86_64__
#include <immintrin.h>
#endif

#ifdef _OPENMP
#define HAVE_OPENMP 1
#else
#define HAVE_OPENMP 0
#endif

#if defined(__ARM_NEON) || defined(__AVX512F__)
#define VECTOR_REGISTERS 32
#else
#define VECTOR_REGISTERS 16
#endif

namespace {

inline int rand32(void) {
    static unsigned long long lcg = 1;
    lcg *= 6364136223846793005;
    lcg += 1442695040888963407;
    return lcg >> 32;
}

inline float float01(unsigned x) { // (0,1)
    return 1.f / 8388608 * ((x >> 9) + .5f);
}

inline float numba(void) { // (-1,1)
    return float01(rand32()) * 2 - 1;
}

template <typename T> void clean(int m, int n, T *A, int lda) {
    for (int i = 0; i < m; ++i)
        for (int j = n; j < lda; ++j)
            A[lda * i + j] = 0;
}

template <typename T> void randomize(int m, int n, T *A, int lda) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            A[lda * i + j] = numba();
}

template <typename V, typename T> V load(const T *p) {
    V v;
    memcpy(&v, p, sizeof(v));
    return v;
}

template <int N> inline float hsum(const float *p) {
    return hsum<N / 2>(p) + hsum<N / 2>(p + N / 2);
}

template <> inline float hsum<1>(const float *p) {
    return *p;
}

template <int N, typename T> struct Vector {
    T v[N];

    operator float() {
        return hsum<N>(v);
    }

    void operator+=(Vector<N, T> x) {
        for (int i = 0; i < N; ++i)
            v[i] += x.v[i];
    }

    Vector<N, T> operator+(Vector<N, T> x) {
        Vector<N, T> r;
        for (int i = 0; i < N; ++i)
            r.v[i] = v[i] + x.v[i];
        return r;
    }

    Vector<N, T> operator-(Vector<N, T> x) {
        Vector<N, T> r;
        for (int i = 0; i < N; ++i)
            r.v[i] = v[i] - x.v[i];
        return r;
    }

    Vector<N, T> operator*(Vector<N, T> x) {
        Vector<N, T> r;
        for (int i = 0; i < N; ++i)
            r.v[i] = v[i] * x.v[i];
        return r;
    }
};

template <> struct Vector<4, float> {
    float x;
    float y;
    float z;
    float w;

    operator float() {
        return x + z + y + w;
    }

    void operator+=(Vector<4, float> rhs) {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        w += rhs.w;
    }

    Vector<4, float> operator+(Vector<4, float> x) {
        return (Vector<4, float>){
            x + x.x,
            y + x.y,
            z + x.z,
            w + x.w,
        };
    }

    Vector<4, float> operator-(Vector<4, float> x) {
        return (Vector<4, float>){
            x - x.x,
            y - x.y,
            z - x.z,
            w - x.w,
        };
    }

    Vector<4, float> operator*(Vector<4, float> x) {
        return (Vector<4, float>){
            x * x.x,
            y * x.y,
            z * x.z,
            w * x.w,
        };
    }
};

struct f32x16 : public Vector<16, float> {};
struct f32x8 : public Vector<8, float> {};
struct f32x4 : public Vector<4, float> {};

////////////////////////////////////////////////////////////////////////////////////////////////////
// FLOATING POINT MATRIX MULTIPLICATION

template <int KN, typename DOT, typename VECTOR, typename TA, typename TB, typename TC>
class tinyBLAS {
  public:
    tinyBLAS(const TA *A, int lda, const TB *B, int ldb, TC *C, int ldc, int ith, int nth)
        : A(A), B(B), C(C), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    void matmul(int m, int n, int k) {
        mnpack(0, m, 0, n, k);
    }

  private:
    void mnpack(int m0, int m, int n0, int n, int k) {
        int mc, nc, mp, np;
        switch ((std::min(m - m0, 4) << 4) | std::min(n - n0, 4)) {
        case 0x44:
        case 0x43:
            mc = 4;
            nc = 3;
            gemm<4, 3>(m0, m, n0, n, k);
            break;
        case 0x34:
            mc = 3;
            nc = 4;
            gemm<3, 4>(m0, m, n0, n, k);
            break;
        case 0x33:
            mc = 3;
            nc = 3;
            gemm<3, 3>(m0, m, n0, n, k);
            break;
        case 0x42:
            mc = 4;
            nc = 2;
            gemm<4, 2>(m0, m, n0, n, k);
            break;
        case 0x24:
            mc = 2;
            nc = 4;
            gemm<2, 4>(m0, m, n0, n, k);
            break;
        case 0x32:
            mc = 3;
            nc = 2;
            gemm<3, 2>(m0, m, n0, n, k);
            break;
        case 0x23:
            mc = 2;
            nc = 3;
            gemm<2, 3>(m0, m, n0, n, k);
            break;
        case 0x41:
            mc = 4;
            nc = 1;
            gemm<4, 1>(m0, m, n0, n, k);
            break;
        case 0x22:
            mc = 2;
            nc = 2;
            gemm<2, 2>(m0, m, n0, n, k);
            break;
        case 0x14:
            mc = 1;
            nc = 4;
            gemm<1, 4>(m0, m, n0, n, k);
            break;
        case 0x31:
            mc = 3;
            nc = 1;
            gemm<3, 1>(m0, m, n0, n, k);
            break;
        case 0x13:
            mc = 1;
            nc = 3;
            gemm<1, 3>(m0, m, n0, n, k);
            break;
        case 0x21:
            mc = 2;
            nc = 1;
            gemm<2, 1>(m0, m, n0, n, k);
            break;
        case 0x12:
            mc = 1;
            nc = 2;
            gemm<1, 2>(m0, m, n0, n, k);
            break;
        case 0x11:
            mc = 1;
            nc = 1;
            gemm<1, 1>(m0, m, n0, n, k);
            break;
        default:
            return;
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np, k);
        mnpack(m0, m, np, n, k);
    }

    template <int RM, int RN> void gemm(int m0, int m, int n0, int n, int k) {
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
            DOT Cv[RN][RM] = {0};
            for (int l = 0; l < k; l += KN)
                for (int j = 0; j < RN; ++j)
                    for (int i = 0; i < RM; ++i)
                        Cv[j][i] += load<VECTOR>(A + lda * (ii + i) + l) *
                                    load<VECTOR>(B + ldb * (jj + j) + l);
            for (int j = 0; j < RN; ++j)
                for (int i = 0; i < RM; ++i)
                    C[ldc * (jj + j) + (ii + i)] = Cv[j][i];
        }
    }

    const TA *const A;
    const TB *const B;
    TC *const C;
    const int lda;
    const int ldb;
    const int ldc;
    const int ith;
    const int nth;
};

static void ansi_sgemm_impl(int m, int n, int k, const float *A, int lda, const float *B, int ldb,
                            float *C, int ldc, int ith, int nth) {
    assert(m >= 0);
    assert(n >= 0);
    assert(k >= 0);
    assert(lda >= k);
    assert(ldb >= k);
    assert(ldc >= m);
    assert(nth > 0);
    assert(ith < nth);
    assert(1ll * lda * m <= 0x7fffffff);
    assert(1ll * ldb * n <= 0x7fffffff);
    assert(1ll * ldc * n <= 0x7fffffff);
#if defined(__AVX512F__)
    assert(!(lda % 16));
    assert(!(ldb % 16));
    tinyBLAS<16, f32x16, f32x16, float, float, float> tb{A, lda, B, ldb, C, ldc, ith, nth};
#elif defined(__AVX__) || defined(__AVX2__)
    assert(!(lda % 8));
    assert(!(ldb % 8));
    tinyBLAS<8, f32x8, f32x8, float, float, float> tb{A, lda, B, ldb, C, ldc, ith, nth};
#elif defined(__SSE__) || defined(__ARM_NEON)
    assert(!(lda % 4));
    assert(!(ldb % 4));
    tinyBLAS<4, f32x4, f32x4, float, float, float> tb{A, lda, B, ldb, C, ldc, ith, nth};
#else
    tinyBLAS<1, float, float, float, float, float> tb{A, lda, B, ldb, C, ldc, ith, nth};
#endif
    tb.matmul(m, n, k);
}

} // namespace

template <typename T, typename U> T *ansi_malloc(int m, int n, U *out_lda) {
    void *ptr;
    int b = 64 / sizeof(T);
    int lda = (n + b - 1) & -b;
    size_t size = sizeof(T) * m * lda;
    if ((errno = posix_memalign(&ptr, sysconf(_SC_PAGESIZE), size))) {
        perror("posix_memalign");
        exit(1);
    }
    *out_lda = lda;
    return (T *)ptr;
}

template <typename T, typename U> T *ansi_new_test_matrix(int m, int n, U *out_lda) {
    T *A = ansi_malloc<T>(m, n, out_lda);
    randomize(m, n, A, *out_lda);
    clean(m, n, A, *out_lda);
    return A;
}

void ansi_sgemm(int m, int n, int k, const float *A, int lda, const float *B, int ldb, float *C,
                int ldc, int ith, int nth) {
    if (nth) {
        ansi_sgemm_impl(m, n, k, A, lda, B, ldb, C, ldc, ith, nth);
    } else if (!HAVE_OPENMP || 1ll * n * m * k < 3000000) {
        ansi_sgemm_impl(m, n, k, A, lda, B, ldb, C, ldc, 0, 1);
    } else {
        nth = sysconf(_SC_NPROCESSORS_ONLN);
#pragma omp parallel for
        for (ith = 0; ith < nth; ++ith)
            ansi_sgemm_impl(m, n, k, A, lda, B, ldb, C, ldc, ith, nth);
    }
}

template <typename T>
void naive(int m, int n, int k, const T *A, int lda, const T *B, int ldb, T *C, int ldc) {
#pragma omp parallel for collapse(2) if (m * n * k > 300000)
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            T d = 0;
            for (int l = 0; l < k; ++l)
                d += A[lda * i + l] * B[ldb * j + l];
            C[ldc * j + i] = d;
        }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int m = 64;
int n = 768;
int k = 768;

// int m = 512;
// int n = 512;
// int k = 512;

// int m = 1024;
// int n = 1024;
// int k = 1024;

// int m = 4609;
// int n = 511;
// int k = 784;

// int m = 8192;
// int n = 8192;
// int k = 8192;

float *A, *B, *C;
int lda, ldb, ldc;

void multiply_naive() {
    naive(m, n, k, A, lda, B, ldb, C, ldc);
    volatile float x = C[0];
    (void)x;
}

void multiply_ansi() {
    ansi_sgemm(m, n, k, A, lda, B, ldb, C, ldc, 0, 0);
    volatile float x = C[0];
    (void)x;
}

long long micros(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec * 1000000 + (ts.tv_nsec + 999) / 1000;
}

long kHuge1[300 * 1024 * 1024 / sizeof(long)];
long kHuge2[300 * 1024 * 1024 / sizeof(long)];
void smash_data_cache(void) {
    int n = sizeof(kHuge1) / sizeof(*kHuge1);
    for (int i = 0; i < n; ++i) {
        kHuge1[i] = (kHuge1[i] + 1) * (kHuge2[n - i - 1] -= 1);
    }
}

void warm_up_openmp(void) {
    int n = sysconf(_SC_NPROCESSORS_ONLN);
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
    }
}

#define BENCH(N, x) \
    do { \
        if (N == 1) { \
            warm_up_openmp(); \
            smash_data_cache(); \
        } else { \
            x; \
        } \
        long long t1 = micros(); \
        for (long long i = 0; i < N; ++i) { \
            asm volatile("" ::: "memory"); \
            x; \
            asm volatile("" ::: "memory"); \
        } \
        long long t2 = micros(); \
        printf("%8lld µs %2dx n=%5d m=%5d k=%5d %s %g gigaflops\n", (t2 - t1 + N - 1) / N, N, \
               (int)n, (int)m, (int)k, #x, 1e6 / ((t2 - t1 + N - 1) / N) * m * n * k * 1e-9); \
    } while (0)

int main() {
    printf("\n");
    A = ansi_new_test_matrix<float>(m, k, &lda);
    B = ansi_new_test_matrix<float>(n, k, &ldb);
    C = ansi_new_test_matrix<float>(n, m, &ldc);
    BENCH(400, multiply_ansi());
    BENCH(400, multiply_naive());
}
