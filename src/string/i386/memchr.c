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

static void *memchr_fallback(const void *haystack, int needle, size_t size) {
	while ((size_t)haystack % sizeof(size_t) && size > 0) {
		if (*(unsigned char*)haystack == needle)
			return (void*)haystack;
		haystack = (char*)haystack + 1;
		size--;
	}
	size_t lowbits = ~(size_t)0 / 0xff;
	size_t highbits = lowbits * 0x80;
	size_t repeated_n = lowbits * needle;
	while (size >= 2*sizeof(size_t)) {
		size_t m1 = *(const size_t*)haystack ^ repeated_n;
		size_t m2 = *((const size_t*)haystack+1) ^ repeated_n;
		if ((((m1-lowbits) & ~m1) | ((m2-lowbits) & ~m2)) & highbits) {
			while (*(unsigned char*)haystack != needle)
				haystack = (char*)haystack + 1;
			return (void*)haystack;
		}
		haystack = (const size_t*)haystack + 2;
		size -= 2*sizeof(size_t);
	}
	while (size) {
		if (*(unsigned char*)haystack == needle)
			return (void*)haystack;
		haystack = (char*)haystack + 1;
		size--;
	}
	return NULL;
}

__attribute__((__target__("sse2")))
static void *memchr_sse2(const void *haystack, int needle, size_t size) {
	while ((size_t)haystack % 4 && size > 0) {
		if (*(unsigned char*)haystack == needle)
			return (void*)haystack;
		haystack = (char*)haystack + 1;
		size--;
	}
	uint32_t lowbits = ~(uint32_t)0 / 0xff;
	uint32_t highbits = lowbits * 0x80;
	uint32_t repeated_n = lowbits * needle;
	while ((size_t)haystack % 16 && size >= 4) {
		uint32_t m = *(const uint32_t*)haystack ^ repeated_n;
		if ((m-lowbits) & ~m & highbits) {
			while (*(unsigned char*)haystack != needle)
				haystack = (char*)haystack + 1;
			return (void*)haystack;
		}
		haystack = (uint32_t*)haystack + 1;
		size -= 4;
	}
	__m128i vn = _mm_set1_epi8(needle);
	if ((size_t)haystack % 32 && size >= 16) {
		__m128i x = _mm_load_si128(haystack);
		__m128i eq = _mm_cmpeq_epi8(x, vn);
		int mask = _mm_movemask_epi8(eq);
		if (mask)
			return (char*)haystack + trailing_zeros(mask);
		haystack = (const __m128i*)haystack + 1;
		size -= 16;
	}
	if ((size_t)haystack % 64 && size >= 32) {
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
		size -= 32;
	}
	while (size >= 64) {
		__m128i a = _mm_load_si128((const __m128i*)haystack);
		__m128i b = _mm_load_si128((const __m128i*)haystack+1);
		__m128i eqa = _mm_cmpeq_epi8(vn, a);
		__m128i eqb = _mm_cmpeq_epi8(vn, b);
		__m128i or1 = _mm_or_si128(eqa, eqb);

		__m128i c = _mm_load_si128((const __m128i*)haystack+2);
		__m128i d = _mm_load_si128((const __m128i*)haystack+3);
		__m128i eqc = _mm_cmpeq_epi8(vn, c);
		__m128i eqd = _mm_cmpeq_epi8(vn, d);
		__m128i or2 = _mm_or_si128(eqc, eqd);

		__m128i or3 = _mm_or_si128(or1, or2);
		if (_mm_movemask_epi8(or3)) {
			int mask;
			if ((mask = _mm_movemask_epi8(eqa)))
				return (char*)(const __m128i*)haystack + trailing_zeros(mask);
			if ((mask = _mm_movemask_epi8(eqb)))
				return (char*)((const __m128i*)haystack+1) + trailing_zeros(mask);
			if ((mask = _mm_movemask_epi8(eqc)))
				return (char*)((const __m128i*)haystack+2) + trailing_zeros(mask);
			if ((mask = _mm_movemask_epi8(eqd)))
				return (char*)((const __m128i*)haystack+3) + trailing_zeros(mask);
			return NULL;
		}
		haystack = (const __m128i*)haystack + 4;
		size -= 64;
	}
	while (size >= 16) {
		__m128i x = _mm_load_si128(haystack);
		__m128i eq = _mm_cmpeq_epi8(x, vn);
		int mask = _mm_movemask_epi8(eq);
		if (mask)
			return (char*)haystack + trailing_zeros(mask);
		haystack = (const __m128i*)haystack + 1;
		size -= 16;
	}
	while (size) {
		if (*(unsigned char*)haystack == needle)
			return (void*)haystack;
		haystack = (char*)haystack + 1;
		size--;
	}
	return NULL;
}

__attribute__((__target__("avx2")))
static void *memchr_avx2(const void *haystack, int needle, size_t size) {
	while ((size_t)haystack % 4 && size > 0) {
		if (*(unsigned char*)haystack == needle)
			return (void*)haystack;
		haystack = (char*)haystack + 1;
		size--;
	}
	uint32_t lowbits = ~(uint32_t)0 / 0xff;
	uint32_t highbits = lowbits * 0x80;
	uint32_t repeated_n = lowbits * needle;
	while (size >= 4 && (size_t)haystack % 16) {
		uint32_t m = *(const uint32_t*)haystack ^ repeated_n;
		if ((m-lowbits) & ~m & highbits) {
			while (*(unsigned char*)haystack != needle)
				haystack = (char*)haystack + 1;
			return (void*)haystack;
		}
		haystack = (uint32_t*)haystack + 1;
		size -= 4;
	}
	__m128i v16n = _mm_set1_epi8(needle);
	while ((size_t)haystack % 64 && size >= 16) {
		__m128i x = _mm_load_si128(haystack);
		__m128i eq = _mm_cmpeq_epi8(x, v16n);
		int mask = _mm_movemask_epi8(eq);
		if (mask)
			return (char*)haystack + trailing_zeros(mask);
		haystack = (const __m128i*)haystack + 1;
		size -= 16;
	}
	__m256i vn = _mm256_set1_epi8(needle);
	if ((size_t)haystack % 128 && size >= 64) {
		__m256i a = _mm256_load_si256(haystack);
		__m256i b = _mm256_load_si256((__m256i*)haystack + 1);
		__m256i eqa = _mm256_cmpeq_epi8(a, vn);
		__m256i eqb = _mm256_cmpeq_epi8(b, vn);
		__m256i or1 = _mm256_or_si256(eqa, eqb);
		if (_mm256_movemask_epi8(or1)) {
			int mask = _mm256_movemask_epi8(eqa);
			if (mask)
				return (char*)haystack + trailing_zeros(mask);
			mask = _mm256_movemask_epi8(eqb);
			return (char*)haystack + 32 + trailing_zeros(mask);
		}
		haystack = (const __m256i*)haystack + 2;
		size -= 64;
	}

	while (size >= 128) {
		__m256i a = _mm256_load_si256((const __m256i*)haystack);
		__m256i b = _mm256_load_si256((const __m256i*)haystack+1);
		__m256i eqa = _mm256_cmpeq_epi8(vn, a);
		__m256i eqb = _mm256_cmpeq_epi8(vn, b);
		__m256i or1 = _mm256_or_si256(eqa, eqb);

		__m256i c = _mm256_load_si256((const __m256i*)haystack+2);
		__m256i d = _mm256_load_si256((const __m256i*)haystack+3);
		__m256i eqc = _mm256_cmpeq_epi8(vn, c);
		__m256i eqd = _mm256_cmpeq_epi8(vn, d);
		__m256i or2 = _mm256_or_si256(eqc, eqd);

		__m256i or3 = _mm256_or_si256(or1, or2);
		if (_mm256_movemask_epi8(or3)) {
			int mask;
			if ((mask = _mm256_movemask_epi8(eqa)))
				return (char*)(const __m256i*)haystack + trailing_zeros(mask);
			if ((mask = _mm256_movemask_epi8(eqb)))
				return (char*)((const __m256i*)haystack+1) + trailing_zeros(mask);
			if ((mask = _mm256_movemask_epi8(eqc)))
				return (char*)((const __m256i*)haystack+2) + trailing_zeros(mask);
			if ((mask = _mm256_movemask_epi8(eqd)))
				return (char*)((const __m256i*)haystack+3) + trailing_zeros(mask);
			return NULL;
		}
		haystack = (const __m256i*)haystack + 4;
		size -= 128;
	}
	while (size >= 16) {
		__m128i x = _mm_load_si128(haystack);
		__m128i eq = _mm_cmpeq_epi8(x, v16n);
		int mask = _mm_movemask_epi8(eq);
		if (mask)
			return (char*)haystack + trailing_zeros(mask);
		haystack = (const __m128i*)haystack + 1;
		size -= 16;
	}
	while (size) {
		if (*(unsigned char*)haystack == needle)
			return (void*)haystack;
		haystack = (char*)haystack + 1;
		size--;
	}
	return NULL;
}

static void *memchr_auto(const void *haystack, int c, size_t n);

static void *(*memchr_impl)(const void *haystack, int c, size_t n) = memchr_auto;

static void *memchr_auto(const void *haystack, int c, size_t n) {
	if (has_avx2())
		memchr_impl = memchr_avx2;
	else if (has_sse2())
		memchr_impl = memchr_sse2;
	else
		memchr_impl = memchr_fallback;
	return memchr_impl(haystack, c, n);
}

void *memchr(const void *haystack, int c, size_t n) {
	return memchr_impl(haystack, (unsigned char)c, n);
}
