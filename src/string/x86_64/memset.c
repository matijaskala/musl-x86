/*
 * Copyright (c) 2020, 2021, Matija Skala <mskala@gmx.com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *     * Neither the name of the author nor the
 *     names of contributors may be used to endorse or promote products
 *     derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY Matija Skala <mskala@gmx.com> ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL Matija Skala <mskala@gmx.com> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <string.h>
#include <stdint.h>
#include <immintrin.h>

#include "cpu_features.h"

static void *memset_fallback(void *s, int c, size_t n) {
	uint8_t *s8 = s;
	if (n < 1)
		return s;
	s8[0] = s8[n-1] = c;
	if (n < 3)
		return s;
	s8[1] = s8[2] = s8[n-3] = s8[n-2] = c;
	if (n < 7)
		return s;
	size_t padding = 3 - ((uintptr_t)(s8-1) % 4);
	s8 += padding;
	n -= padding;
	uint32_t c32 = (~(uint32_t)0 / 0xff) * (c & 0xff);
	uint32_t *s32 = (void*)s8;
	n /= 4;
	s32[0] = s32[n-1] = c32;
	if (n < 3)
		return s;
	s32[1] = s32[2] = s32[n-3] = s32[n-2] = c32;
	if (n < 7)
		return s;
	s32[3] = s32[4] = s32[5] = s32[6] = s32[n-7] = s32[n-6] = s32[n-5] = s32[n-4] = c32;
	padding = 7 - ((uintptr_t)(s32-1)/4 % 2);
	s32 += padding;
	n -= padding;
	uint64_t c64 = c32;
	c64 = c64 | c64 << 32;
	uint64_t *s64 = (void*)s32;
	n /= 8;
	for (size_t i = 0; i < n; i++) {
		s64[4*i] = c64;
		s64[4*i+1] = c64;
		s64[4*i+2] = c64;
		s64[4*i+3] = c64;
	}
	return s;
}

__attribute__((__target__("sse2")))
static void *memset_sse2(void *s, int c, size_t n) {
	uint8_t *s8 = s;
	if (n < 28) {
		if (n < 1)
			return s;
		s8[0] = s8[n-1] = c;
		if (n < 3)
			return s;
		s8[1] = s8[2] = s8[n-3] = s8[n-2] = c;
		if (n < 7)
			return s;
		size_t padding = 3 - ((uintptr_t)(s8-1) % 4);
		s8 += padding;
		n -= padding;
		uint32_t c32 = (~(uint32_t)0 / 0xff) * (c & 0xff);
		uint32_t *s32 = (void*)s8;
		n /= 4;
		s32[0] = s32[n-1] = c32;
		if (n < 3)
			return s;
		s32[1] = s32[2] = s32[n-3] = s32[n-2] = c32;
	}
	else {
		__m128i c64 = _mm_set1_epi8(c);
		_mm_storeu_si128((__m128i *)&s8[0], c64);
		_mm_storeu_si128((__m128i *)&s8[n-16], c64);
		size_t padding = 15 - ((uintptr_t)(s8-1) % 16);
		s8 += padding;
		n -= padding;
		__m128i *s128 = (void*)s8;
		n /= 16;
		for (size_t i = 0; i < n%4; i++)
			_mm_store_si128(s128++, c64);
		if (n >= 1 << 21) {
			for (size_t i = 0; i < n/4; i++) {
				_mm_stream_si128(&s128[4*i], c64);
				_mm_stream_si128(&s128[4*i+1], c64);
				_mm_stream_si128(&s128[4*i+2], c64);
				_mm_stream_si128(&s128[4*i+3], c64);
			}
			_mm_sfence();
		}
		else
			for (size_t i = 0; i < n/4; i++) {
				_mm_store_si128(&s128[4*i], c64);
				_mm_store_si128(&s128[4*i+1], c64);
				_mm_store_si128(&s128[4*i+2], c64);
				_mm_store_si128(&s128[4*i+3], c64);
			}
	}
	return s;
}

__attribute__((__target__("avx")))
static void *memset_avx(void *s, int c, size_t n) {
	uint8_t *s8 = s;
	if (n < 28) {
		if (n < 1)
			return s;
		s8[0] = s8[n-1] = c;
		if (n < 3)
			return s;
		s8[1] = s8[2] = s8[n-3] = s8[n-2] = c;
		if (n < 7)
			return s;
		size_t padding = 3 - ((uintptr_t)(s8-1) % 4);
		s8 += padding;
		n -= padding;
		uint32_t c32 = (~(uint32_t)0 / 0xff) * (c & 0xff);
		uint32_t *s32 = (void*)s8;
		n /= 4;
		s32[0] = s32[n-1] = c32;
		if (n < 3)
			return s;
		s32[1] = s32[2] = s32[n-3] = s32[n-2] = c32;
		if (n < 7)
			return s;
		s32[3] = s32[4] = s32[5] = s32[6] = s32[n-7] = s32[n-6] = s32[n-5] = s32[n-4] = c32;
	}
	else if (n < 65) {
		__m128i c64 = _mm_set1_epi8(c);
		for (size_t i = 0; i < n/16; i++)
			_mm_storeu_si128((__m128i *)s8+i, c64);
		if (n%16)
			_mm_storeu_si128((__m128i *)&s8[n-16], c64);
	}
	else {
		__m256i c64 = _mm256_set1_epi8(c);
		_mm256_storeu_si256((__m256i *)&s8[0], c64);
		_mm256_storeu_si256((__m256i *)&s8[n-32], c64);
		size_t padding = 31 - ((uintptr_t)(s8-1) % 32);
		s8 += padding;
		n -= padding;
		__m256i *s256 = (void*)s8;
		n /= 32;
		for (size_t i = 0; i < n%4; i++)
			_mm256_store_si256(s256++, c64);
		if (n >= 1 << 21) {
			for (size_t i = 0; i < n/4; i++) {
				_mm256_stream_si256(&s256[4*i], c64);
				_mm256_stream_si256(&s256[4*i+1], c64);
				_mm256_stream_si256(&s256[4*i+2], c64);
				_mm256_stream_si256(&s256[4*i+3], c64);
			}
			_mm_sfence();
		}
		else
			for (size_t i = 0; i < n/4; i++) {
				_mm256_store_si256(&s256[4*i], c64);
				_mm256_store_si256(&s256[4*i+1], c64);
				_mm256_store_si256(&s256[4*i+2], c64);
				_mm256_store_si256(&s256[4*i+3], c64);
			}
	}
	return s;
}

static void *memset_auto(void *s, int c, size_t n);

static void *(*memset_impl)(void *s, int c, size_t n) = memset_auto;

static void *memset_auto(void *s, int c, size_t n) {
	if (has_avx())
		memset_impl = memset_avx;
	else if (has_sse2())
		memset_impl = memset_sse2;
	else
		memset_impl = memset_fallback;
	return memset_impl(s, c, n);
}

void *memset(void *s, int c, size_t n) {
	return memset_impl(s, c, n);
}
