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
#include <immintrin.h>

#include "cpu_features.h"

static void *memccpy_naive(void *restrict dest, const void *restrict src, int c, size_t n) {
	char *d = dest;
	const char *s = src;
	for (size_t i = 0; i < n; i++)
		if ((d[i] = s[i]) == (char)c)
			return d+i+1;
	return NULL;
}

__attribute__((__target__("sse2")))
static void *memccpy_sse2(void *restrict dest, const void *restrict src, int c, size_t n) {
	char *d = dest;
	const char *s = src;
	__m128i v = _mm_set1_epi8(c);
	while (n >= 16) {
		__m128i x = _mm_loadu_si128((const __m128i*)s);
		if (_mm_movemask_epi8(_mm_cmpeq_epi8(x, v)))
			break;
		_mm_storeu_si128((__m128i*)d, x);
		s += 16;
		d += 16;
		n -= 16;
	}
	return memccpy_naive(d, s, c, n);
}

static void *memccpy_auto(void *restrict dest, const void *restrict src, int c, size_t n);

static void *(*memccpy_impl)(void *restrict dest, const void *restrict src, int c, size_t n) = memccpy_auto;

static void *memccpy_auto(void *restrict dest, const void *restrict src, int c, size_t n) {
	if (has_sse2())
		memccpy_impl = memccpy_sse2;
	else
		memccpy_impl = memccpy_naive;
	return memccpy_impl(dest, src, c, n);
}

void *memccpy(void *restrict dest, const void *restrict src, int c, size_t n) {
	return memccpy_impl(dest, src, c, n);
}
