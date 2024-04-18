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

void dgemm(int m, int n, int k, float alpha, //
           const float *A, int lda, //
           const float *B, int ldb, float beta, //
           float *C, int ldc) {
#pragma omp parallel for collapse(2) if (1ll * m * n * k > 30000)
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            double sum = 0;
            double err = 0;
            for (int l = 0; l < k; ++l) {
                double a = A[lda * i + l];
                double b = B[ldb * j + l];
                double y = a * b - err;
                double t = sum + y;
                err = (t - sum) - y;
                sum = t;
            }
            C[ldc * j + i] = sum;
        }
}

template <typename T>
double diff_impl(long m, long n, const T *Want, long lda, const float *Got, long ldb) {
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

double diff(long m, long n, const float *Want, long lda, const float *Got, long ldb) {
    return diff_impl(m, n, Want, lda, Got, ldb);
}

double diff(long m, long n, const double *Want, long lda, const float *Got, long ldb) {
    return diff_impl(m, n, Want, lda, Got, ldb);
}
