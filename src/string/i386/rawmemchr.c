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
#include "helpers.h"

static void* rawmemchr_fallback(const void *haystack, int n) {
	while ((size_t)haystack % sizeof(size_t)) {
		if (*(unsigned char*)haystack == n)
			return (void*)haystack;
		haystack = (char*)haystack + 1;
	}
	size_t lowbits = ~(size_t)0 / 0xff;
	size_t highbits = lowbits * 0x80;
	size_t repeated_n = lowbits * n;
	for (;;) {
		size_t m1 = *(const size_t*)haystack ^ repeated_n;
		size_t m2 = *((const size_t*)haystack+1) ^ repeated_n;
		if ((((m1-lowbits) & ~m1) | ((m2-lowbits) & ~m2)) & highbits) {
			while (*(unsigned char*)haystack != n)
				haystack = (char*)haystack + 1;
			return (void*)haystack;
		}
		haystack = (const size_t*)haystack + 2;
	}
}

__attribute__((__target__("sse2")))
static void* rawmemchr_sse2(const void *haystack, int n) {
	while ((size_t)haystack % 4) {
		if (*(unsigned char*)haystack == n)
			return (void*)haystack;
		haystack = (char*)haystack + 1;
	}
	uint32_t lowbits = ~(uint32_t)0 / 0xff;
	uint32_t highbits = lowbits * 0x80;
	uint32_t repeated_n = lowbits * n;
	while ((size_t)haystack % 16) {
		uint32_t m = *(const uint32_t*)haystack ^ repeated_n;
		if ((m-lowbits) & ~m & highbits) {
			while (*(unsigned char*)haystack != n)
				haystack = (char*)haystack + 1;
			return (void*)haystack;
		}
		haystack = (uint32_t*)haystack + 1;
	}
	__m128i vn = _mm_set1_epi8(n);
	if ((size_t)haystack % 32) {
		__m128i x = _mm_load_si128(haystack);
		__m128i eq = _mm_cmpeq_epi8(x, vn);
		int mask = _mm_movemask_epi8(eq);
		if (mask)
			return (char*)haystack + trailing_zeros(mask);
		haystack = (const __m128i*)haystack + 1;
	}
	if ((size_t)haystack % 64) {
		__m128i a = _mm_load_si128(haystack);
		__m128i b = _mm_load_si128((__m128i*)haystack + 1);
		__m128i eqa = _mm_cmpeq_epi8(a, vn);
		__m128i eqb = _mm_cmpeq_epi8(b, vn);
		__m128i or1 = _mm_or_si128(eqa, eqb);
		if (_mm_movemask_epi8(or1)) {
			int mask = _mm_movemask_epi8(eqa);
			if (mask)
				return (char*)haystack + trailing_zeros(mask);
			mask = _mm_movemask_epi8(eqb);
			return (char*)haystack + 16 + trailing_zeros(mask);
		}
		haystack = (const __m128i*)haystack + 2;
	}
	const __m128i *ptr = haystack;
	for (;;) {
		__m128i a = _mm_load_si128(ptr);
		__m128i b = _mm_load_si128(ptr+1);
		__m128i eqa = _mm_cmpeq_epi8(vn, a);
		__m128i eqb = _mm_cmpeq_epi8(vn, b);
		__m128i or1 = _mm_or_si128(eqa, eqb);

		__m128i c = _mm_load_si128(ptr+2);
		__m128i d = _mm_load_si128(ptr+3);
		__m128i eqc = _mm_cmpeq_epi8(vn, c);
		__m128i eqd = _mm_cmpeq_epi8(vn, d);
		__m128i or2 = _mm_or_si128(eqc, eqd);

		__m128i or3 = _mm_or_si128(or1, or2);
		if (_mm_movemask_epi8(or3)) {
			int mask;
			if ((mask = _mm_movemask_epi8(eqa)))
				return (char*)ptr + trailing_zeros(mask);
			if ((mask = _mm_movemask_epi8(eqb)))
				return (char*)(ptr+1) + trailing_zeros(mask);
			if ((mask = _mm_movemask_epi8(eqc)))
				return (char*)(ptr+2) + trailing_zeros(mask);
			if ((mask = _mm_movemask_epi8(eqd)))
				return (char*)(ptr+3) + trailing_zeros(mask);
			return NULL;
		}
		ptr += 4;
	}
}

__attribute__((__target__("avx2")))
static void* rawmemchr_avx2(const void *haystack, int n) {
	while ((size_t)haystack % 4) {
		if (*(unsigned char*)haystack == n)
			return (void*)haystack;
		haystack = (char*)haystack + 1;
	}
	uint32_t lowbits = ~(uint32_t)0 / 0xff;
	uint32_t highbits = lowbits * 0x80;
	uint32_t repeated_n = lowbits * n;
	while ((size_t)haystack % 16) {
		uint32_t m = *(const uint32_t*)haystack ^ repeated_n;
		if ((m-lowbits) & ~m & highbits) {
			while (*(unsigned char*)haystack != n)
				haystack = (char*)haystack + 1;
			return (void*)haystack;
		}
		haystack = (uint32_t*)haystack + 1;
	}
	__m128i v16n = _mm_set1_epi8(n);
	while ((size_t)haystack % 64) {
		__m128i x = _mm_load_si128(haystack);
		__m128i eq = _mm_cmpeq_epi8(x, v16n);
		int mask = _mm_movemask_epi8(eq);
		if (mask)
			return (char*)haystack + trailing_zeros(mask);
		haystack = (const __m128i*)haystack + 1;
	}
	__m256i vn = _mm256_set1_epi8(n);
	if ((size_t)haystack % 128) {
		__m256i a = _mm256_load_si256(haystack);
		__m256i b = _mm256_load_si256((__m256i*)haystack + 1);
		__m256i eqa = _mm256_cmpeq_epi8(a, vn);
		__m256i eqb = _mm256_cmpeq_epi8(b, vn);
		if (_mm256_movemask_epi8(_mm256_or_si256(eqa, eqb))) {
			int mask = _mm256_movemask_epi8(eqa);
			if (mask)
				return (char*)haystack + trailing_zeros(mask);
			mask = _mm256_movemask_epi8(eqb);
			return (char*)haystack + 32 + trailing_zeros(mask);
		}
		haystack = (const __m256i*)haystack + 2;
	}
	const __m256i *ptr = haystack;

	for (;;) {
		__m256i a = _mm256_load_si256(ptr);
		__m256i b = _mm256_load_si256(ptr+1);
		__m256i eqa = _mm256_cmpeq_epi8(vn, a);
		__m256i eqb = _mm256_cmpeq_epi8(vn, b);
		__m256i or1 = _mm256_or_si256(eqa, eqb);

		__m256i c = _mm256_load_si256(ptr+2);
		__m256i d = _mm256_load_si256(ptr+3);
		__m256i eqc = _mm256_cmpeq_epi8(vn, c);
		__m256i eqd = _mm256_cmpeq_epi8(vn, d);
		__m256i or2 = _mm256_or_si256(eqc, eqd);

		__m256i or3 = _mm256_or_si256(or1, or2);
		if (_mm256_movemask_epi8(or3)) {
			int mask;
			if ((mask = _mm256_movemask_epi8(eqa)))
				return (char*)ptr + trailing_zeros(mask);
			if ((mask = _mm256_movemask_epi8(eqb)))
				return (char*)(ptr+1) + trailing_zeros(mask);
			if ((mask = _mm256_movemask_epi8(eqc)))
				return (char*)(ptr+2) + trailing_zeros(mask);
			if ((mask = _mm256_movemask_epi8(eqd)))
				return (char*)(ptr+3) + trailing_zeros(mask);
			return NULL;
		}
		ptr += 4;
	}
}

static void *rawmemchr_auto(const void *s, int c);

static void *(*rawmemchr_impl)(const void *s, int) = rawmemchr_auto;

static void *rawmemchr_auto(const void *s, int c) {
	if (has_avx2())
		rawmemchr_impl = rawmemchr_avx2;
	else if (has_sse2())
		rawmemchr_impl = rawmemchr_sse2;
	else
		rawmemchr_impl = rawmemchr_fallback;
	return rawmemchr_impl(s, c);
}

void *rawmemchr(const void *s, int c) {
	return rawmemchr_impl(s, (unsigned char)c);
}
