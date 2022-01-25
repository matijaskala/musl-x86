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
#include <immintrin.h>

#include "cpu_features.h"
#include "helpers.h"

static size_t wcslen_fallback(const wchar_t *haystack) {
	const wchar_t *start = haystack;
	while ((size_t)haystack % sizeof(size_t)) {
		if (!*haystack)
			return haystack - start;
		haystack++;
	}
	size_t lowbits = ~(size_t)0 / (size_t)~(wchar_t)0;
	size_t highbits = lowbits << (sizeof(wchar_t) * 8 - 1);
	for (;;) {
		size_t m1 = *(const size_t*)haystack;
		size_t m2 = *((const size_t*)haystack+1);
		if ((((m1-lowbits) & ~m1) | ((m2-lowbits) & ~m2)) & highbits) {
			while (*haystack)
				haystack++;
			return haystack - start;
		}
		haystack += 2 * sizeof(size_t) / sizeof(wchar_t);
	}
}

__attribute__((__target__("sse2")))
static size_t wcslen_sse2(const wchar_t *haystack) {
	const wchar_t *start = haystack;
	while ((size_t)haystack % 16) {
		if (!*haystack)
			return haystack - start;
		haystack++;
	}
	__m128i zero = _mm_set1_epi32(0);
	if ((size_t)haystack % 32) {
		__m128i x = _mm_load_si128((__m128i*)haystack);
		__m128i eq = _mm_cmpeq_epi32(x, zero);
		int mask = _mm_movemask_epi8(eq);
		if (mask)
			return (wchar_t*)((char*)haystack + trailing_zeros(mask)) - start;
		haystack = (wchar_t*)((const __m128i*)haystack + 1);
	}
	if ((size_t)haystack % 64) {
		__m128i a = _mm_load_si128((__m128i*)haystack);
		__m128i b = _mm_load_si128((__m128i*)haystack + 1);
		__m128i eqa = _mm_cmpeq_epi32(a, zero);
		__m128i eqb = _mm_cmpeq_epi32(b, zero);
		__m128i or1 = _mm_or_si128(eqa, eqb);
		if (_mm_movemask_epi8(or1)) {
			int mask = _mm_movemask_epi8(eqa);
			if (mask)
				return (wchar_t*)((char*)haystack + trailing_zeros(mask)) - start;
			mask = _mm_movemask_epi8(eqb);
			return (wchar_t*)((char*)haystack + 16 + trailing_zeros(mask)) - start;
		}
		haystack = (wchar_t*)((const __m128i*)haystack + 2);
	}
	const __m128i *ptr = (const void*)haystack;
	for (;;) {
		__m128i a = _mm_load_si128(ptr);
		__m128i b = _mm_load_si128(ptr+1);
		__m128i eqa = _mm_cmpeq_epi32(zero, a);
		__m128i eqb = _mm_cmpeq_epi32(zero, b);
		__m128i or1 = _mm_or_si128(eqa, eqb);

		__m128i c = _mm_load_si128(ptr+2);
		__m128i d = _mm_load_si128(ptr+3);
		__m128i eqc = _mm_cmpeq_epi32(zero, c);
		__m128i eqd = _mm_cmpeq_epi32(zero, d);
		__m128i or2 = _mm_or_si128(eqc, eqd);

		__m128i or3 = _mm_or_si128(or1, or2);
		if (_mm_movemask_epi8(or3)) {
			int mask;
			if ((mask = _mm_movemask_epi8(eqa)))
				return (wchar_t*)((char*)ptr + trailing_zeros(mask)) - start;
			if ((mask = _mm_movemask_epi8(eqb)))
				return (wchar_t*)((char*)(ptr+1) + trailing_zeros(mask)) - start;
			if ((mask = _mm_movemask_epi8(eqc)))
				return (wchar_t*)((char*)(ptr+2) + trailing_zeros(mask)) - start;
			if ((mask = _mm_movemask_epi8(eqd)))
				return (wchar_t*)((char*)(ptr+3) + trailing_zeros(mask)) - start;
		}
		ptr += 4;
	}
}

__attribute__((__target__("avx2")))
static size_t wcslen_avx2(const wchar_t *haystack) {
	const wchar_t *start = haystack;
	while ((size_t)haystack % 16) {
		if (!*haystack)
			return haystack - start;
		haystack++;
	}
	__m128i v16z = _mm_set1_epi32(0);
	while ((size_t)haystack % 64) {
		__m128i x = _mm_load_si128((__m128i*)haystack);
		__m128i eq = _mm_cmpeq_epi32(x, v16z);
		int mask = _mm_movemask_epi8(eq);
		if (mask)
			return (wchar_t*)((char*)haystack + trailing_zeros(mask)) - start;
		haystack = (wchar_t*)((const __m128i*)haystack + 1);
	}
	__m256i zero = _mm256_set1_epi32(0);
	if ((size_t)haystack % 128) {
		__m256i a = _mm256_load_si256((__m256i*)haystack);
		__m256i b = _mm256_load_si256((__m256i*)haystack + 1);
		__m256i eqa = _mm256_cmpeq_epi32(a, zero);
		__m256i eqb = _mm256_cmpeq_epi32(b, zero);
		if (_mm256_movemask_epi8(_mm256_or_si256(eqa, eqb))) {
			int mask = _mm256_movemask_epi8(eqa);
			if (mask)
				return (wchar_t*)((char*)haystack + trailing_zeros(mask)) - start;
			mask = _mm256_movemask_epi8(eqb);
			return (wchar_t*)((char*)haystack + 32 + trailing_zeros(mask)) - start;
		}
		haystack = (wchar_t*)((const __m256i*)haystack + 2);
	}
	const __m256i *ptr = (const void*)haystack;

	for (;;) {
		__m256i a = _mm256_load_si256(ptr);
		__m256i b = _mm256_load_si256(ptr+1);
		__m256i eqa = _mm256_cmpeq_epi32(zero, a);
		__m256i eqb = _mm256_cmpeq_epi32(zero, b);
		__m256i or1 = _mm256_or_si256(eqa, eqb);

		__m256i c = _mm256_load_si256(ptr+2);
		__m256i d = _mm256_load_si256(ptr+3);
		__m256i eqc = _mm256_cmpeq_epi32(zero, c);
		__m256i eqd = _mm256_cmpeq_epi32(zero, d);
		__m256i or2 = _mm256_or_si256(eqc, eqd);

		__m256i or3 = _mm256_or_si256(or1, or2);
		if (_mm256_movemask_epi8(or3)) {
			int mask;
			if ((mask = _mm256_movemask_epi8(eqa)))
				return (wchar_t*)((char*)ptr + trailing_zeros(mask)) - start;
			if ((mask = _mm256_movemask_epi8(eqb)))
				return (wchar_t*)((char*)(ptr+1) + trailing_zeros(mask)) - start;
			if ((mask = _mm256_movemask_epi8(eqc)))
				return (wchar_t*)((char*)(ptr+2) + trailing_zeros(mask)) - start;
			if ((mask = _mm256_movemask_epi8(eqd)))
				return (wchar_t*)((char*)(ptr+3) + trailing_zeros(mask)) - start;
		}
		ptr += 4;
	}
}

static size_t wcslen_auto(const wchar_t *s);

static size_t (*wcslen_impl)(const wchar_t *s) = wcslen_auto;

static size_t wcslen_auto(const wchar_t *s) {
	if (has_avx2())
		wcslen_impl = wcslen_avx2;
	else if (has_sse2())
		wcslen_impl = wcslen_sse2;
	else
		wcslen_impl = wcslen_fallback;
	return wcslen_impl(s);
}

size_t wcslen(const wchar_t *s) {
	return wcslen_impl(s);
}
