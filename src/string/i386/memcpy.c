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

static void *memcpy_naive(void *restrict dest, const void *restrict src, size_t n) {
	char *d = dest;
	const char *s = src;
	for (size_t i = 0; i < n; i++)
		d[i] = s[i];
	return dest;
}

__attribute__((__target__("sse2")))
static void *memcpy_sse2(void *restrict dest, const void *restrict src, size_t n) {
	char *const d = dest;
	const char *const s = src;
	size_t f = 0;
	if (n < 128) {
		while (n >= 64) {
			for (int i = 0; i < 4; i++) {
#pragma omp simd
				for (int j = 0; j < 16; j++)
					d[f+j] = s[f+j];
				f += 16;
				n -= 16;
			}
		}
		while (n >= 32) {
			for (int i = 0; i < 2; i++) {
#pragma omp simd
				for (int j = 0; j < 16; j++)
					d[f+j] = s[f+j];
				f += 16;
				n -= 16;
			}
		}
		while (n >= 16) {
			for (int i = 0; i < 1; i++) {
#pragma omp simd
				for (int j = 0; j < 16; j++)
					d[f+j] = s[f+j];
				f += 16;
				n -= 16;
			}
		}
		memcpy_naive(d + f, s + f, n);
		return dest;
	}
	if (n >= 1 << 21) {
		for (int i = 0; i < 8; i++) {
#pragma omp simd
			for (int j = 0; j < 16; j++)
				d[f+j] = s[f+j];
			f += 16;
			n -= 16;
		}
		size_t m = (size_t)d % 128;
		f -= m;
		n += m;
		while (n >= 256) {
			__m128i chunk1 = _mm_loadu_si128((const __m128i*)(s+f)+0);
			__m128i chunk2 = _mm_loadu_si128((const __m128i*)(s+f)+1);
			__m128i chunk3 = _mm_loadu_si128((const __m128i*)(s+f)+2);
			__m128i chunk4 = _mm_loadu_si128((const __m128i*)(s+f)+3);
			__m128i chunk5 = _mm_loadu_si128((const __m128i*)(s+f)+4);
			__m128i chunk6 = _mm_loadu_si128((const __m128i*)(s+f)+5);
			__m128i chunk7 = _mm_loadu_si128((const __m128i*)(s+f)+6);
			__m128i chunk8 = _mm_loadu_si128((const __m128i*)(s+f)+7);
			_mm_prefetch(s+256, _MM_HINT_NTA);
			_mm_stream_si128((__m128i*)(d+f)+0, chunk1);
			_mm_stream_si128((__m128i*)(d+f)+1, chunk2);
			_mm_stream_si128((__m128i*)(d+f)+2, chunk3);
			_mm_stream_si128((__m128i*)(d+f)+3, chunk4);
			_mm_stream_si128((__m128i*)(d+f)+4, chunk5);
			_mm_stream_si128((__m128i*)(d+f)+5, chunk6);
			_mm_stream_si128((__m128i*)(d+f)+6, chunk7);
			_mm_stream_si128((__m128i*)(d+f)+7, chunk8);
			f += 128;
			n -= 128;
		}
		_mm_sfence();
	}
	while (n >= 128) {
		for (int i = 0; i < 8; i++) {
#pragma omp simd
			for (int j = 0; j < 16; j++)
				d[f+j] = s[f+j];
			f += 16;
			n -= 16;
		}
	}
	while (n >= 64) {
		for (int i = 0; i < 4; i++) {
#pragma omp simd
			for (int j = 0; j < 16; j++)
				d[f+j] = s[f+j];
			f += 16;
			n -= 16;
		}
	}
	while (n >= 32) {
		for (int i = 0; i < 2; i++) {
#pragma omp simd
			for (int j = 0; j < 16; j++)
				d[f+j] = s[f+j];
			f += 16;
			n -= 16;
		}
	}
	while (n >= 16) {
		for (int i = 0; i < 1; i++) {
#pragma omp simd
			for (int j = 0; j < 16; j++)
				d[f+j] = s[f+j];
			f += 16;
			n -= 16;
		}
	}
	if (n) {
		for (int i = 0; i < 1; i++) {
			f += n - 16;
#pragma omp simd
			for (int j = 0; j < 16; j++)
				d[f+j] = s[f+j];
		}
	}
	return dest;
}

static void *memcpy_auto(void *restrict dest, const void *restrict src, size_t n);

static void *(*memcpy_impl)(void *restrict dest, const void *restrict src, size_t n) = memcpy_auto;

static void *memcpy_auto(void *restrict dest, const void *restrict src, size_t n) {
	if (has_sse2())
		memcpy_impl = memcpy_sse2;
	else
		memcpy_impl = memcpy_naive;
	return memcpy_impl(dest, src, n);
}

void *memcpy(void *restrict dest, const void *restrict src, size_t n) {
	return memcpy_impl(dest, src, n);
}
