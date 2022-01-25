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

static void* rawmemchr2_fallback(const void *haystack, int n1, int n2) {
	while ((size_t)haystack % sizeof(size_t)) {
		if (*(unsigned char*)haystack == n1 || *(unsigned char*)haystack == n2)
			return (void*)haystack;
		haystack = (char*)haystack + 1;
	}
	size_t lowbits = ~(size_t)0 / 0xff;
	size_t highbits = lowbits * 0x80;
	size_t repeated_n1 = lowbits * n1;
	size_t repeated_n2 = lowbits * n2;
	for (;;) {
		size_t m1 = *(const size_t*)haystack ^ repeated_n1;
		size_t m2 = *(const size_t*)haystack ^ repeated_n2;
		if ((((m1-lowbits) & ~m1) | ((m2-lowbits) & ~m2)) & highbits) {
			while (*(unsigned char*)haystack != n1 && *(unsigned char*)haystack != n2)
				haystack = (char*)haystack + 1;
			return (void*)haystack;
		}
		haystack = (const size_t*)haystack + 1;
	}
}

__attribute__((__target__("sse2")))
static void* rawmemchr2_sse2(const void *haystack, int n1, int n2) {
	while ((size_t)haystack % 4) {
		if (*(unsigned char*)haystack == n1 || *(unsigned char*)haystack == n2)
			return (void*)haystack;
		haystack = (char*)haystack + 1;
	}
	uint32_t lowbits = ~(uint32_t)0 / 0xff;
	uint32_t highbits = lowbits * 0x80;
	uint32_t repeated_n1 = lowbits * n1;
	uint32_t repeated_n2 = lowbits * n2;
	while ((size_t)haystack % 16) {
		uint32_t m1 = *(const uint32_t*)haystack ^ repeated_n1;
		uint32_t m2 = *(const uint32_t*)haystack ^ repeated_n2;
		if ((((m1-lowbits) & ~m1) | ((m2-lowbits) & ~m2)) & highbits) {
			while (*(unsigned char*)haystack != n1 && *(unsigned char*)haystack != n2)
				haystack = (char*)haystack + 1;
			return (void*)haystack;
		}
		haystack = (int32_t*)haystack + 1;
	}
	__m128i vn1 = _mm_set1_epi8(n1);
	__m128i vn2 = _mm_set1_epi8(n2);
	if ((size_t)haystack % 32) {
		__m128i x = _mm_load_si128(haystack);
		__m128i eq1 = _mm_cmpeq_epi8(x, vn1);
		__m128i eq2 = _mm_cmpeq_epi8(x, vn2);
		if (_mm_movemask_epi8(_mm_or_si128(eq1, eq2))) {
			int mask1 = _mm_movemask_epi8(eq1);
			int mask2 = _mm_movemask_epi8(eq2);
			return (char*)haystack + trailing_zeros(mask1 | mask2);
		}
		haystack = (const __m128i*)haystack + 1;
	}
	const __m128i *ptr = haystack;
	for (;;) {
		__m128i a = _mm_load_si128(ptr);
		__m128i b = _mm_load_si128(ptr+1);
		__m128i eqa1 = _mm_cmpeq_epi8(vn1, a);
		__m128i eqb1 = _mm_cmpeq_epi8(vn1, b);
		__m128i eqa2 = _mm_cmpeq_epi8(vn2, a);
		__m128i eqb2 = _mm_cmpeq_epi8(vn2, b);
		__m128i or1 = _mm_or_si128(eqa1, eqb1);
		__m128i or2 = _mm_or_si128(eqa2, eqb2);
		__m128i or3 = _mm_or_si128(or1, or2);

		if (_mm_movemask_epi8(or3)) {
			int mask1, mask2;
			mask1 = _mm_movemask_epi8(eqa1);
			mask2 = _mm_movemask_epi8(eqa2);
			if (mask1 || mask2)
				return (char*)ptr + trailing_zeros(mask1 | mask2);
			mask1 = _mm_movemask_epi8(eqb1);
			mask2 = _mm_movemask_epi8(eqb2);
			return (char*)(ptr+1) + trailing_zeros(mask1 | mask2);
		}
		ptr += 2;
	}
}

__attribute__((__target__("avx2")))
static void* rawmemchr2_avx2(const void *haystack, int n1, int n2) {
	while ((size_t)haystack % 4) {
		if (*(unsigned char*)haystack == n1 || *(unsigned char*)haystack == n2)
			return (void*)haystack;
		haystack = (char*)haystack + 1;
	}
	uint32_t lowbits = ~(uint32_t)0 / 0xff;
	uint32_t highbits = lowbits * 0x80;
	uint32_t repeated_n1 = lowbits * n1;
	uint32_t repeated_n2 = lowbits * n2;
	while ((size_t)haystack % 16) {
		uint32_t m1 = *(const uint32_t*)haystack ^ repeated_n1;
		uint32_t m2 = *(const uint32_t*)haystack ^ repeated_n2;
		if ((((m1-lowbits) & ~m1) | ((m2-lowbits) & ~m2)) & highbits) {
			while (*(unsigned char*)haystack != n1 && *(unsigned char*)haystack != n2)
				haystack = (char*)haystack + 1;
			return (void*)haystack;
		}
		haystack = (int32_t*)haystack + 1;
	}
	__m128i v16n1 = _mm_set1_epi8(n1);
	__m128i v16n2 = _mm_set1_epi8(n2);
	while ((size_t)haystack % 64) {
		__m128i x = _mm_load_si128(haystack);
		__m128i eq1 = _mm_cmpeq_epi8(x, v16n1);
		__m128i eq2 = _mm_cmpeq_epi8(x, v16n2);
		if (_mm_movemask_epi8(_mm_or_si128(eq1, eq2))) {
			int mask1 = _mm_movemask_epi8(eq1);
			int mask2 = _mm_movemask_epi8(eq2);
			return (char*)haystack + trailing_zeros(mask1 | mask2);
		}
		haystack = (const __m128i*)haystack + 1;
	}
	__m256i vn1 = _mm256_set1_epi8(n1);
	__m256i vn2 = _mm256_set1_epi8(n2);
	const __m256i *ptr = haystack;
	for (;;) {
		__m256i a = _mm256_load_si256(ptr);
		__m256i b = _mm256_load_si256(ptr+1);
		__m256i eqa1 = _mm256_cmpeq_epi8(vn1, a);
		__m256i eqb1 = _mm256_cmpeq_epi8(vn1, b);
		__m256i eqa2 = _mm256_cmpeq_epi8(vn2, a);
		__m256i eqb2 = _mm256_cmpeq_epi8(vn2, b);
		__m256i or1 = _mm256_or_si256(eqa1, eqb1);
		__m256i or2 = _mm256_or_si256(eqa2, eqb2);
		__m256i or3 = _mm256_or_si256(or1, or2);

		if (_mm256_movemask_epi8(or3)) {
			int mask1, mask2;
			mask1 = _mm256_movemask_epi8(eqa1);
			mask2 = _mm256_movemask_epi8(eqa2);
			if (mask1 || mask2)
				return (char*)ptr + trailing_zeros(mask1 | mask2);
			mask1 = _mm256_movemask_epi8(eqb1);
			mask2 = _mm256_movemask_epi8(eqb2);
			return (char*)(ptr+1) + trailing_zeros(mask1 | mask2);
		}
		ptr += 2;
	}
}

static void *rawmemchr2_auto(const void *haystack, int n1, int n2);

static void *(*rawmemchr2_impl)(const void *haystack, int n1, int n2) = rawmemchr2_auto;

static void *rawmemchr2_auto(const void *haystack, int n1, int n2) {
	if (has_avx2())
		rawmemchr2_impl = rawmemchr2_avx2;
	else if (has_sse2())
		rawmemchr2_impl = rawmemchr2_sse2;
	else
		rawmemchr2_impl = rawmemchr2_fallback;
	return rawmemchr2_impl(haystack, n1, n2);
}

char *__strchrnul(const char *s, int c)
{
	return rawmemchr2_impl(s, (unsigned char)c, 0);
}

weak_alias(__strchrnul, strchrnul);
