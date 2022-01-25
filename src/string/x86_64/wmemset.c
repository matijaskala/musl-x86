/*
 * Copyright (c) 2021, Matija Skala <mskala@gmx.com>
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

#include <wchar.h>
#include <stdint.h>
#include <immintrin.h>

#include "cpu_features.h"

static wchar_t *wmemset_fallback(wchar_t *s, wchar_t c, size_t n) {
	for (size_t i = 0; i < n; i++) {
		s[i] = c;
	}
	return s;
}

__attribute__((__target__("sse2")))
static wchar_t *wmemset_sse2(wchar_t *s, wchar_t c, size_t n) {
	if (n < 1)
		return s;
	s[0] = s[n-1] = c;
	if (n < 3)
		return s;
	s[1] = s[2] = s[n-3] = s[n-2] = c;
	if (n < 7)
		return s;
	size_t padding = (15 - ((uintptr_t)(s-1) % 16)) / sizeof(wchar_t);
	n -= padding;
	__m128i *s128 = (void*)(s+padding);
	__m128i c64 = _mm_set1_epi32(c);
	n /= 4;
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
	return s;
}

__attribute__((__target__("avx")))
static wchar_t *wmemset_avx(wchar_t *s, wchar_t c, size_t n) {
	if (n < 17) {
		if (n < 1)
			return s;
		s[0] = s[n-1] = c;
		if (n < 3)
			return s;
		s[1] = s[2] = s[n-3] = s[n-2] = c;
		if (n < 7)
			return s;
		size_t padding = (15 - ((uintptr_t)(s-1) % 16)) / sizeof(wchar_t);
		n -= padding;
		__m128i *s128 = (void*)(s+padding);
		__m128i c64 = _mm_set1_epi32(c);
		n /= 4;
		for (size_t i = 0; i < n; i++)
			_mm_store_si128(s128+i, c64);
	}
	else {
		__m256i c64 = _mm256_set1_epi32(c);
		_mm256_storeu_si256((__m256i *)&s[0], c64);
		_mm256_storeu_si256((__m256i *)&s[n-8], c64);
		size_t padding = 31 - ((uintptr_t)(s-1) % 32) / sizeof(wchar_t);
		n -= padding;
		__m256i *s256 = (void*)(s+padding);
		n /= 8;
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

static wchar_t *wmemset_auto(wchar_t *s, wchar_t c, size_t n);

static wchar_t *(*wmemset_impl)(wchar_t *s, wchar_t c, size_t n) = wmemset_auto;

static wchar_t *wmemset_auto(wchar_t *s, wchar_t c, size_t n) {
	if (has_avx())
		wmemset_impl = wmemset_avx;
	else if (has_sse2())
		wmemset_impl = wmemset_sse2;
	else
		wmemset_impl = wmemset_fallback;
	return wmemset_impl(s, c, n);
}

wchar_t *wmemset(wchar_t *s, wchar_t c, size_t n) {
	return wmemset_impl(s, c, n);
}
