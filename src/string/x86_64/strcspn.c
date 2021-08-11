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

static void* rawmemchr3_fallback(const void *haystack, int n1, int n2, int n3) {
	while ((size_t)haystack % sizeof(size_t)) {
		if (*(unsigned char*)haystack == n1 || *(unsigned char*)haystack == n2 || *(unsigned char*)haystack == n3)
			return (void*)haystack;
		haystack = (char*)haystack + 1;
	}
	size_t lowbits = ~(size_t)0 / 0xff;
	size_t highbits = lowbits * 0x80;
	size_t repeated_n1 = lowbits * n1;
	size_t repeated_n2 = lowbits * n2;
	size_t repeated_n3 = lowbits * n3;
	for (;;) {
		size_t m1 = *(const size_t*)haystack ^ repeated_n1;
		size_t m2 = *(const size_t*)haystack ^ repeated_n2;
		size_t m3 = *(const size_t*)haystack ^ repeated_n3;
		if ((((m1-lowbits) & ~m1) | ((m2-lowbits) & ~m2) | ((m3-lowbits) & ~m3)) & highbits) {
			while (*(unsigned char*)haystack != n1 && *(unsigned char*)haystack != n2 && *(unsigned char*)haystack != n3)
				haystack = (char*)haystack + 1;
			return (void*)haystack;
		}
		haystack = (const size_t*)haystack + 1;
	}
}

__attribute__((__target__("sse2")))
static void* rawmemchr3_sse2(const void *haystack, int n1, int n2, int n3) {
	while ((size_t)haystack % sizeof(size_t)) {
		if (*(unsigned char*)haystack == n1 || *(unsigned char*)haystack == n2 || *(unsigned char*)haystack == n3)
			return (void*)haystack;
		haystack = (char*)haystack + 1;
	}
	uint32_t lowbits = ~(uint32_t)0 / 0xff;
	uint32_t highbits = lowbits * 0x80;
	uint32_t repeated_n1 = lowbits * n1;
	uint32_t repeated_n2 = lowbits * n2;
	uint32_t repeated_n3 = lowbits * n3;
	while ((size_t)haystack % 16) {
		uint32_t m1 = *(const uint32_t*)haystack ^ repeated_n1;
		uint32_t m2 = *(const uint32_t*)haystack ^ repeated_n2;
		uint32_t m3 = *(const uint32_t*)haystack ^ repeated_n3;
		if ((((m1-lowbits) & ~m1) | ((m2-lowbits) & ~m2) | ((m3-lowbits) & ~m3)) & highbits) {
			while (*(unsigned char*)haystack != n1 && *(unsigned char*)haystack != n2 && *(unsigned char*)haystack != n3)
				haystack = (char*)haystack + 1;
			return (void*)haystack;
		}
		haystack = (int32_t*)haystack + 1;
	}
	__m128i vn1 = _mm_set1_epi8(n1);
	__m128i vn2 = _mm_set1_epi8(n2);
	__m128i vn3 = _mm_set1_epi8(n3);
	if ((size_t)haystack % 32) {
		__m128i x = _mm_load_si128(haystack);
		__m128i eq1 = _mm_cmpeq_epi8(x, vn1);
		__m128i eq2 = _mm_cmpeq_epi8(x, vn2);
		__m128i eq3 = _mm_cmpeq_epi8(x, vn3);
		if (_mm_movemask_epi8(_mm_or_si128(_mm_or_si128(eq1, eq2), eq3))) {
			int mask1 = _mm_movemask_epi8(eq1);
			int mask2 = _mm_movemask_epi8(eq2);
			int mask3 = _mm_movemask_epi8(eq3);
			return (char*)haystack + trailing_zeros(mask1 | mask2 | mask3);
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
		__m128i eqa3 = _mm_cmpeq_epi8(vn3, a);
		__m128i eqb3 = _mm_cmpeq_epi8(vn3, b);
		__m128i or1 = _mm_or_si128(eqa1, eqb1);
		__m128i or2 = _mm_or_si128(eqa2, eqb2);
		__m128i or3 = _mm_or_si128(eqa3, eqb3);
		__m128i or4 = _mm_or_si128(or1, or2);
		__m128i or5 = _mm_or_si128(or3, or4);

		if (_mm_movemask_epi8(or5)) {
			int mask1 = _mm_movemask_epi8(eqa1);
			int mask2 = _mm_movemask_epi8(eqa2);
			int mask3 = _mm_movemask_epi8(eqa3);
			if (mask1 || mask2 || mask3)
				return (char*)ptr + trailing_zeros(mask1 | mask2 | mask3);
			mask1 = _mm_movemask_epi8(eqb1);
			mask2 = _mm_movemask_epi8(eqb2);
			mask3 = _mm_movemask_epi8(eqb3);
			return (char*)(ptr+1) + trailing_zeros(mask1 | mask2 | mask3);
		}
		ptr += 2;
	}
}

__attribute__((__target__("avx2")))
static void* rawmemchr3_avx2(const void *haystack, int n1, int n2, int n3) {
	while ((size_t)haystack % sizeof(size_t)) {
		if (*(unsigned char*)haystack == n1 || *(unsigned char*)haystack == n2 || *(unsigned char*)haystack == n3)
			return (void*)haystack;
		haystack = (char*)haystack + 1;
	}
	uint32_t lowbits = ~(uint32_t)0 / 0xff;
	uint32_t highbits = lowbits * 0x80;
	uint32_t repeated_n1 = lowbits * n1;
	uint32_t repeated_n2 = lowbits * n2;
	uint32_t repeated_n3 = lowbits * n3;
	while ((size_t)haystack % 16) {
		uint32_t m1 = *(const uint32_t*)haystack ^ repeated_n1;
		uint32_t m2 = *(const uint32_t*)haystack ^ repeated_n2;
		uint32_t m3 = *(const uint32_t*)haystack ^ repeated_n3;
		if ((((m1-lowbits) & ~m1) | ((m2-lowbits) & ~m2) | ((m3-lowbits) & ~m3)) & highbits) {
			while (*(unsigned char*)haystack != n1 && *(unsigned char*)haystack != n2 && *(unsigned char*)haystack != n3)
				haystack = (char*)haystack + 1;
			return (void*)haystack;
		}
		haystack = (int32_t*)haystack + 1;
	}
	__m128i v16n1 = _mm_set1_epi8(n1);
	__m128i v16n2 = _mm_set1_epi8(n2);
	__m128i v16n3 = _mm_set1_epi8(n3);
	while ((size_t)haystack % 64) {
		__m128i x = _mm_load_si128(haystack);
		__m128i eq1 = _mm_cmpeq_epi8(x, v16n1);
		__m128i eq2 = _mm_cmpeq_epi8(x, v16n2);
		__m128i eq3 = _mm_cmpeq_epi8(x, v16n3);
		if (_mm_movemask_epi8(_mm_or_si128(_mm_or_si128(eq1, eq2), eq3))) {
			int mask1 = _mm_movemask_epi8(eq1);
			int mask2 = _mm_movemask_epi8(eq2);
			int mask3 = _mm_movemask_epi8(eq3);
			return (char*)haystack + trailing_zeros(mask1 | mask2 | mask3);
		}
		haystack = (const __m128i*)haystack + 1;
	}
	__m256i vn1 = _mm256_set1_epi8(n1);
	__m256i vn2 = _mm256_set1_epi8(n2);
	__m256i vn3 = _mm256_set1_epi8(n3);
	const __m256i *ptr = haystack;
	for (;;) {
		__m256i a = _mm256_load_si256(ptr);
		__m256i b = _mm256_load_si256(ptr+1);
		__m256i eqa1 = _mm256_cmpeq_epi8(vn1, a);
		__m256i eqb1 = _mm256_cmpeq_epi8(vn1, b);
		__m256i eqa2 = _mm256_cmpeq_epi8(vn2, a);
		__m256i eqb2 = _mm256_cmpeq_epi8(vn2, b);
		__m256i eqa3 = _mm256_cmpeq_epi8(vn3, a);
		__m256i eqb3 = _mm256_cmpeq_epi8(vn3, b);
		__m256i or1 = _mm256_or_si256(eqa1, eqb1);
		__m256i or2 = _mm256_or_si256(eqa2, eqb2);
		__m256i or3 = _mm256_or_si256(eqa3, eqb3);
		__m256i or4 = _mm256_or_si256(or1, or2);
		__m256i or5 = _mm256_or_si256(or3, or4);

		if (_mm256_movemask_epi8(or5)) {
			int mask1 = _mm256_movemask_epi8(eqa1);
			int mask2 = _mm256_movemask_epi8(eqa2);
			int mask3 = _mm256_movemask_epi8(eqa3);
			if (mask1 || mask2 || mask3)
				return (char*)ptr + trailing_zeros(mask1 | mask2 | mask3);
			mask1 = _mm256_movemask_epi8(eqb1);
			mask2 = _mm256_movemask_epi8(eqb2);
			mask3 = _mm256_movemask_epi8(eqb3);
			return (char*)(ptr+1) + trailing_zeros(mask1 | mask2 | mask3);
		}
		ptr += 2;
	}
}

static void *rawmemchr3_auto(const void *haystack, int n1, int n2, int n3);

static void *(*rawmemchr3_impl)(const void *haystack, int n1, int n2, int n3) = rawmemchr3_auto;

static void *rawmemchr3_auto(const void *haystack, int n1, int n2, int n3) {
	if (has_avx2())
		rawmemchr3_impl = rawmemchr3_avx2;
	else if (has_sse2())
		rawmemchr3_impl = rawmemchr3_sse2;
	else
		rawmemchr3_impl = rawmemchr3_fallback;
	return rawmemchr3_impl(haystack, n1, n2, n3);
}

size_t strcspn(const char *s, const char *reject) {
	extern char *strchrnul(const char *s, int c);
	if (!reject[0])
		return strlen(s);
	else if (!reject[1])
		return strchrnul(s, (unsigned char)reject[0]) - s;
	else if (!reject[2])
		return (char*)rawmemchr3_impl(s, (unsigned char)reject[0], (unsigned char)reject[1], (unsigned char)reject[2]) - s;
	uint64_t byteset[4] = {0};
	byteset[0] |= 1;
	while (*reject) {
		byteset[(unsigned char)*reject/64] |= (uint64_t)1 << (unsigned char)*reject%64;
		reject++;
	}
	size_t i = 0;
	while (!(byteset[(unsigned char)s[i]/64] & (uint64_t)1 << (unsigned char)s[i]%64))
		i++;
	return i;
}
