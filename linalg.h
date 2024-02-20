// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c++ ts=4 sts=4 sw=4 fenc=utf-8 :vi
#pragma once
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#include <utility>
#include <vector>

#define PADDING 0
#define FLAWLESS 0
#define REGISTER 32
#define CACHELINE 64
#define LV1DCACHE 49152
#define THRESHOLD 3000000
#define MEMORYLOC 0x0000000001000000
#define NOINLINE __attribute__((__noinline__))

#ifdef _OPENMP
#define HAVE_OPENMP 1
#else
#define HAVE_OPENMP 0
#endif

#if defined(__OPTIMIZE__) && !defined(__SANITIZE_ADDRESS__)
#define ITERATIONS 30
#else
#define ITERATIONS 1
#endif

#if LOGGING
#define LOG_LOCK() flockfile(stderr)
#define LOG(...) (!is_self_testing && fprintf(stderr, __VA_ARGS__))
#define LOG_UNLOCK() funlockfile(stderr)
#else
#define LOG_LOCK()
#define LOG(...)
#define LOG_UNLOCK()
#endif

#ifndef NDEBUG
#define ASSERT(x)                                                              \
    do {                                                                       \
        if (!(x)) {                                                            \
            fprintf(stderr, "%s:%d: assertion failed: %s\n", __FILE__,         \
                    __LINE__, #x);                                             \
            __builtin_trap();                                                  \
        }                                                                      \
    } while (0)
#else
#define ASSERT(x)                                                              \
    do {                                                                       \
        if (!(x)) {                                                            \
            __builtin_unreachable();                                           \
        }                                                                      \
    } while (0)
#endif

typedef long long i64;
typedef unsigned long long u64;

void dgemm(long, long, long, float, const float *, long, const float *, long,
           float, float *, long);
double diff(long, long, const float *, long, const float *, long);
double diff(long, long, const double *, long, const float *, long);

thread_local static const char *is_self_testing;

unsigned long long lemur(void) {
    static unsigned __int128 s = 2131259787901769494;
    return (s *= 15750249268501108917ull) >> 64;
}

struct Map {
    long p;
    long e;
};

struct Memory {
    std::atomic_long spot;
    std::mutex lock;
    std::vector<Map> maps;
};

static Memory g_memory{ATOMIC_VAR_INIT(MEMORYLOC)};

NOINLINE void rngset(char *p, long n) {
    long i = 0;
    unsigned long long x = lemur();
    while (i < n && ((long)(p + i) & 7))
        p[i++] = x, x >>= 8;
    for (; i + 7 < n; i += 8)
        *(long *)(p + i) = lemur();
    x = lemur();
    while (i < n)
        p[i++] = x, x >>= 8;
}

template <typename T>
NOINLINE T *new_matrix(long m, long n = 1, i64 *out_ldn = nullptr, long b = 1) {
    long ldn = (n + b - 1) & -b;
    long size = sizeof(T) * m * ldn;
    long pagesz = sysconf(_SC_PAGESIZE);
    long addr = g_memory.spot.fetch_add(MEMORYLOC, std::memory_order_relaxed);
    g_memory.spot += MEMORYLOC;
    if (size) {
        char *ptr = (char *)mmap((void *)addr, size, PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED) {
            fprintf(stderr, "error: out of memory: %s\n", strerror(errno));
            _Exit(1);
        }
        addr = (long)ptr;
    }
    std::unique_lock<std::mutex> lock(g_memory.lock);
    long true_size = (size + pagesz - 1) & -pagesz;
    // rngset((char *)addr, true_size);
    Map map{addr, addr + true_size};
    g_memory.maps.emplace_back(std::move(map));
    addr += true_size - size;
    addr &= -(b * sizeof(T));
    if (out_ldn)
        *out_ldn = ldn;
    return (T *)addr;
}

template <typename T> NOINLINE void delete_matrix(T *matrix) {
    std::unique_lock<std::mutex> lock(g_memory.lock);
    long p = (long)matrix;
    std::vector<Map>::reverse_iterator i;
    for (i = g_memory.maps.rbegin(); i != g_memory.maps.rend(); ++i) {
        if ((p >= i->p && p < i->e) || (p == i->p && p == i->e)) {
            if (p >= i->p && p < i->e) {
                if (munmap((void *)i->p, i->e - i->p)) {
                    fprintf(stderr, "error: munmap failed: %s\n",
                            strerror(errno));
                    _Exit(1);
                }
            }
            g_memory.maps.erase(--(i.base()));
            return;
        }
    }
    fprintf(stderr, "error: map not found: %#lx\n", p);
    _Exit(1);
}

template <typename TA>
NOINLINE void show(FILE *f, i64 max, i64 m, i64 n, const TA *A, i64 lda) {
    flockfile(f);
    fprintf(f, "      ");
    for (i64 i = 0; i < n && i < max; ++i) {
        fprintf(f, "%13lld", i);
    }
    fputs_unlocked("\n", f);
    for (i64 i = 0; i < m; ++i) {
        if (i == max) {
            fputs_unlocked("...\n", f);
            break;
        }
        fprintf(f, "%5lld ", i);
        for (i64 j = 0; j < n; ++j) {
            if (j == max) {
                fputs_unlocked(" ...", f);
                break;
            }
            if (j >= n)
                fputs_unlocked("\33[30m", f);
            char ba[16];
            sprintf(ba, "%13.7f", static_cast<double>(A[lda * i + j]));
            fputs_unlocked(ba, f);
            if (j >= n)
                fputs_unlocked("\33[0m", f);
        }
        fputs_unlocked("\n", f);
    }
    funlockfile(f);
}

template <typename TA, typename TB>
NOINLINE void show(FILE *f, i64 max, i64 m, i64 n, const TA *A, i64 lda,
                   const TB *B, i64 ldb) {
    flockfile(f);
    fprintf(f, "      ");
    for (i64 i = 0; i < n && i < max; ++i) {
        fprintf(f, "%13lld", i);
    }
    fprintf(f, "\n");
    for (i64 i = 0; i < m; ++i) {
        if (i == max) {
            fputs_unlocked("...\n", f);
            break;
        }
        fprintf(f, "%5lld ", i);
        for (i64 j = 0; j < n; ++j) {
            if (j == max) {
                fputs_unlocked(" ...", f);
                break;
            }
            char ba[32], bb[32];
            sprintf(ba, "%13.7f", static_cast<double>(A[lda * i + j]));
            sprintf(bb, "%13.7f", static_cast<double>(B[ldb * i + j]));
            for (i64 k = 0; ba[k] && bb[k]; ++k) {
                if (ba[k] != bb[k])
                    fputs_unlocked("\33[31m", f);
                fputc_unlocked(ba[k], f);
                if (ba[k] != bb[k])
                    fputs_unlocked("\33[0m", f);
            }
        }
        fputs_unlocked("\n", f);
    }
    funlockfile(f);
}

template <typename TA, typename TB>
NOINLINE void show_error(FILE *f, i64 max, i64 m, i64 n, const TA *A, i64 lda,
                         const TB *B, i64 ldb, const char *file, int line,
                         double sad, double tol) {
    flockfile(f);
    fprintf(f, "%s:%d: sad %.17g exceeds %g (%s)\nwant\n", file, line, sad, tol,
            is_self_testing ? is_self_testing : "n/a");
    show(f, max, m, n, A, lda, B, ldb);
    fprintf(f, "got\n");
    show(f, max, m, n, B, ldb, A, lda);
    fprintf(f, "\n");
    funlockfile(f);
}

template <typename TA, typename TB>
NOINLINE void check(double tol, i64 m, i64 n, const TA *A, i64 lda, const TB *B,
                    i64 ldb, const char *file, int line) {
    double sad = diff(m, n, A, lda, B, ldb);
    if (sad <= tol) {
        if (!is_self_testing)
            printf("         %g error\n", sad);
    } else {
        flockfile(stderr);
        show_error(stderr, 16, m, n, A, lda, B, ldb, file, line, sad, tol);
        const char *path = "/tmp/openmp_test.log";
        FILE *f = fopen(path, "w");
        if (f) {
            show_error(f, 10000, m, n, A, lda, B, ldb, file, line, sad, tol);
            printf("see also %s\n", path);
        }
        exit(1);
    }
}

#define check(tol, m, n, A, lda, B, ldb)                                       \
    check(tol, m, n, A, lda, B, ldb, __FILE__, __LINE__)

i64 micros(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec * 1000000 + (ts.tv_nsec + 999) / 1000;
}

#define bench(x)                                                               \
    do {                                                                       \
        x;                                                                     \
        i64 t1 = micros();                                                     \
        for (i64 i = 0; i < ITERATIONS; ++i) {                                 \
            asm volatile("" ::: "memory");                                     \
            x;                                                                 \
            asm volatile("" ::: "memory");                                     \
        }                                                                      \
        i64 t2 = micros();                                                     \
        printf("%8lld µs %s\n", (t2 - t1 + ITERATIONS - 1) / ITERATIONS, #x);  \
    } while (0)

void log_mb(i64 m, i64 n, i64 k) {
    double mb = 1024 * 1024;
    fprintf(
        stderr,
        "[%lld, %lld] %g * [%lld, %lld] %g = [%lld, %lld] %g (%g mb total)\n",
        m, k, m * k / mb, k, n, k * n / mb, m, n, m * n / mb,
        (m * k + k * n + m * n) / mb);
}

#define run(x)                                                                 \
    printf("\n%s\n", #x);                                                      \
    x

double real01(unsigned long x) { // (0,1)
    return 1. / 4503599627370496. * ((x >> 12) + .5);
}

double numba(void) { // (-1,1)
    return real01(lemur()) * 2 - 1;
}

template <typename T> NOINLINE void fill(i64 m, i64 n, T *A, i64 lda) {
    for (i64 i = 0; i < m; ++i)
        for (i64 j = 0; j < n; ++j) {
            A[lda * i + j] = numba();
        }
}

template <typename T> NOINLINE void clear(i64 m, i64 n, T *A, i64 lda) {
    for (i64 i = 0; i < m; ++i)
        for (i64 j = 0; j < n; ++j) {
            A[lda * i + j] = 0;
        }
}

template <typename T>
NOINLINE void broadcast(i64 m, i64 n, T *A, i64 lda, T x) {
    for (i64 i = 0; i < m; ++i)
        for (i64 j = 0; j < n; ++j) {
            A[lda * i + j] = x;
        }
}

// m×n → n×m
template <typename TA, typename TB>
NOINLINE void transpose(i64 m, i64 n, const TA *A, i64 lda, TB *B, i64 ldb) {
#pragma omp parallel for collapse(2) if (m * n > THRESHOLD)
    for (i64 i = 0; i < m; ++i)
        for (i64 j = 0; j < n; ++j)
            B[ldb * j + i] = A[lda * i + j];
}

static int get_l1d_cache_size(void) {
#ifdef _SC_LEVEL1_DCACHE_SIZE
    int res = sysconf(_SC_LEVEL1_DCACHE_SIZE);
    if (res >= 4096)
        return res;
#endif
#if defined(__APPLE__) && defined(HW_L1DCACHESIZE)
    int cmd[2];
    long n = 0;
    size_t z = sizeof(n);
    cmd[0] = CTL_HW;
    cmd[1] = HW_L1DCACHESIZE;
    if (sysctl(cmd, 2, &n, &z, 0, 0) != -1 && n >= 4096)
        return n;
#endif
    char buf[8] = {0};
    FILE *f = fopen("/sys/devices/system/cpu/cpu0/cache/index0/size", "rb");
    if (f) {
        fread(buf, 1, 8, f);
        fclose(f);
        return std::max(4096, atoi(buf));
    } else {
        return 32 * 1024;
    }
}

NOINLINE int tinyblas_l1d_cache_size(void) {
    static int save;
    if (!save)
        save = get_l1d_cache_size();
    return save;
}

NOINLINE void affinity(int cpu) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask)) {
        perror("sched_setaffinity");
        _Exit(1);
    }
}
