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

#include <cmath>
#include <cstdio>

void dgemm(long m, long n, long k, float alpha,  //
           const float *A, long lda,             //
           const float *B, long ldb, float beta, //
           float *C, long ldc) {
#pragma omp parallel for collapse(2) if (m * n * k > 30000)
    for (long i = 0; i < m; ++i)
        for (long j = 0; j < n; ++j) {
            double sum = 0;
            for (long l = 0; l < k; ++l)
                sum += A[lda * i + l] * B[ldb * j + l];
            C[ldc * j + i] = sum;
        }
}

template <typename T>
double diff_impl(long m, long n, const T *Want, long lda, const float *Got,
                 long ldb) {
    double s = 0;
    int got_nans = 0;
    int want_nans = 0;
    if (!m || !n)
        return 0;
    for (long i = 0; i < m; ++i)
        for (long j = 0; j < n; ++j)
            if (std::isnan(Want[lda * i + j]))
                ++want_nans;
            else if (std::isnan(Got[ldb * i + j]))
                ++got_nans;
            else
                s += std::fabs(Want[lda * i + j] - Got[ldb * i + j]);
    if (got_nans)
        std::fprintf(stderr, "WARNING: got %d NaNs!\n", got_nans);
    if (want_nans)
        std::fprintf(stderr, "WARNING: want array has %d NaNs!\n", want_nans);
    return s / (m * n);
}

double diff(long m, long n, const float *Want, long lda, const float *Got,
            long ldb) {
    return diff_impl(m, n, Want, lda, Got, ldb);
}

double diff(long m, long n, const double *Want, long lda, const float *Got,
            long ldb) {
    return diff_impl(m, n, Want, lda, Got, ldb);
}
