// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c++ ts=4 sts=4 sw=4 fenc=utf-8 :vi
// clang-format off
#pragma once

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef __x86_64__
#include <immintrin.h>
#endif

typedef _Float16 half;

/**
 * Google Brain 16-bit floating point number.
 *
 *       ┌sign
 *       │
 *       │   ┌exponent
 *       │   │
 *       │   │      ┌mantissa
 *       │   │      │
 *       │┌──┴───┐┌─┴───┐
 *     0b0000000000000000 brain16
 *
 * Since bf16 has the same number of exponent bits as a 32bit float,
 * encoding and decoding numbers becomes relatively straightforward.
 *
 *       ┌sign
 *       │
 *       │   ┌exponent
 *       │   │
 *       │   │      ┌mantissa
 *       │   │      │
 *       │┌──┴───┐┌─┴───────────────────┐
 *     0b00000000000000000000000000000000 IEEE binary32
 *
 * For comparison, the standard fp16 format has fewer exponent bits.
 *
 *       ┌sign
 *       │
 *       │  ┌exponent
 *       │  │
 *       │  │    ┌mantissa
 *       │  │    │
 *       │┌─┴─┐┌─┴──────┐
 *     0b0000000000000000 IEEE binary16
 *
 * So be warned that converting between them, destroys several bits.
 *
 * @see IEEE 754-2008
 */
typedef struct {
  uint16_t x;
} ggml_bf16_t;

/**
 * Converts brain16 to float32.
 */
static inline float ggml_bf16_to_fp32(ggml_bf16_t h) {
  union {
    float f;
    uint32_t i;
  } u;
  u.i = (uint32_t)h.x << 16;
  return u.f;
}

/**
 * Converts float32 to brain16.
 *
 * This function is binary identical to AMD Zen4 VCVTNEPS2BF16.
 * Subnormals shall be flushed to zero, and NANs will be quiet.
 * This code should vectorize nicely if using modern compilers.
 */
static inline ggml_bf16_t ggml_fp32_to_bf16(float s) {
  ggml_bf16_t h;
  union {
    float f;
    uint32_t i;
  } u;
  u.f = s;
  if ((u.i & 0x7fffffff) > 0x7f800000) { /* nan */
    h.x = (u.i >> 16) | 64; /* force to quiet */
    return h;
  }
  if (!(u.i & 0x7f800000)) { /* subnormal */
    h.x = (u.i & 0x80000000) >> 16; /* flush to zero */
    return h;
  }
  h.x = (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
  return h;
}

template <typename T, typename U> T load(const U *);

template <> inline float load(const float *p) {
    return *p;
}

#if defined(__ARM_NEON)
template <> inline float32x4_t load(const float *p) {
    return vld1q_f32(p);
}
#if !defined(_MSC_VER)
template <> inline float16x8_t load(const half *p) {
    return vld1q_f16((const float16_t *)p);
}
template <> inline float32x4_t load(const half *p) {
    return vcvt_f32_f16(vld1_f16((const float16_t *)p));
}
#endif // _MSC_VER
#endif // __ARM_NEON

#if defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
template <> inline __m128 load(const float *p) {
    return _mm_loadu_ps(p);
}
#endif // __SSE__

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
template <> inline __m256 load(const float *p) {
    return _mm256_loadu_ps(p);
}
#endif // __AVX__

#if defined(__F16C__)
template <> inline __m256 load(const half *p) {
    return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)p));
}
#endif // __F16C__

#if defined(__AVX512F__)
template <> inline __m512 load(const float *p) {
    return _mm512_loadu_ps(p);
}
template <> inline __m512 load(const half *p) {
    return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)p));
}
#endif // __AVX512F__

#if defined(__AVX512BF16__)
template <> inline __m512bh load(const float *p) {
    return _mm512_cvtne2ps_pbh(_mm512_loadu_ps(p + 16), _mm512_loadu_ps(p));
}
template <> inline __m512bh load(const ggml_bf16_t *p) {
    return (__m512bh)_mm512_loadu_ps((const float *)p);
}
#endif

#if defined(__AVX2__) || defined(__AVX512F__)
template <> inline __m256 load(const ggml_bf16_t *p) {
    return _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)p)), 16));
}
#endif

#if defined(__AVX512BF16__) && defined(__AVX512VL__)
template <> inline __m512 load(const ggml_bf16_t *p) {
    return _mm512_castsi512_ps(
        _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)p)), 16));
}
#endif
