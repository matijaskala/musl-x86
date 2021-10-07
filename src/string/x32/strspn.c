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

static char* strspn1_fallback(const void *haystack, int n) {
	while ((size_t)haystack % sizeof(size_t)) {
		if (*(unsigned char*)haystack != n)
			return (void*)haystack;
		haystack = (char*)haystack + 1;
	}
	size_t lowbits = ~(size_t)0 / 0xff;
	size_t repeated_n = lowbits * n;
	for (;;) {
		size_t m1 = *(const size_t*)haystack ^ repeated_n;
		size_t m2 = *((const size_t*)haystack+1) ^ repeated_n;
		if (m1 != 0 || m2 != 0)
			for (int i = 0; i < 2*sizeof(size_t); i++)
				if (((unsigned char*)haystack)[i] != n)
					return (char*)haystack + i;
		haystack = (const size_t*)haystack + 2;
	}
}

__attribute__((__target__("sse2")))
static char* strspn1_sse2(const void *haystack, int n) {
	while ((size_t)haystack % 4) {
		if (*(unsigned char*)haystack != n)
			return (void*)haystack;
		haystack = (char*)haystack + 1;
	}
	uint32_t lowbits = ~(uint32_t)0 / 0xff;
	uint32_t repeated_n = lowbits * n;
	while ((size_t)haystack % 16) {
		uint32_t m = *(const uint32_t*)haystack ^ repeated_n;
		if (m != 0)
			for (int i = 0; i < sizeof(uint32_t); i++)
				if (((unsigned char*)haystack)[i] != n)
					return (char*)haystack + i;
		haystack = (uint32_t*)haystack + 1;
	}
	__m128i vn = _mm_set1_epi8(n);
	if ((size_t)haystack % 32) {
		__m128i x = _mm_load_si128(haystack);
		__m128i eq = _mm_cmpeq_epi8(x, vn);
		int mask = _mm_movemask_epi8(eq);
		if (mask != 0xffff)
			return (char*)haystack + trailing_zeros(~mask);
		haystack = (const __m128i*)haystack + 1;
	}
	if ((size_t)haystack % 64) {
		__m128i a = _mm_load_si128(haystack);
		__m128i b = _mm_load_si128((__m128i*)haystack + 1);
		__m128i eqa = _mm_cmpeq_epi8(a, vn);
		__m128i eqb = _mm_cmpeq_epi8(b, vn);
		__m128i or1 = _mm_or_si128(eqa, eqb);
		if (_mm_movemask_epi8(or1) != 0xffff) {
			int mask = _mm_movemask_epi8(eqa);
			if (mask != 0xffff)
				return (char*)haystack + trailing_zeros(~mask);
			mask = _mm_movemask_epi8(eqb);
			return (char*)haystack + 16 + trailing_zeros(~mask);
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
		if (_mm_movemask_epi8(or3) != 0xffff) {
			int mask;
			if ((mask = _mm_movemask_epi8(eqa)) != 0xffff)
				return (char*)ptr + trailing_zeros(~mask);
			if ((mask = _mm_movemask_epi8(eqb)) != 0xffff)
				return (char*)(ptr+1) + trailing_zeros(~mask);
			if ((mask = _mm_movemask_epi8(eqc)) != 0xffff)
				return (char*)(ptr+2) + trailing_zeros(~mask);
			if ((mask = _mm_movemask_epi8(eqd)) != 0xffff)
				return (char*)(ptr+3) + trailing_zeros(~mask);
			return NULL;
		}
		ptr += 4;
	}
}

__attribute__((__target__("avx2")))
static char* strspn1_avx2(const void *haystack, int n) {
	while ((size_t)haystack % 4) {
		if (*(unsigned char*)haystack != n)
			return (void*)haystack;
		haystack = (char*)haystack + 1;
	}
	uint32_t lowbits = ~(uint32_t)0 / 0xff;
	uint32_t repeated_n = lowbits * n;
	while ((size_t)haystack % 16) {
		uint32_t m = *(const uint32_t*)haystack ^ repeated_n;
		if (m != 0)
			for (int i = 0; i < sizeof(uint32_t); i++)
				if (((unsigned char*)haystack)[i] != n)
					return (char*)haystack + i;
		haystack = (uint32_t*)haystack + 1;
	}
	__m128i v16n = _mm_set1_epi8(n);
	while ((size_t)haystack % 64) {
		__m128i x = _mm_load_si128(haystack);
		__m128i eq = _mm_cmpeq_epi8(x, v16n);
		int mask = _mm_movemask_epi8(eq);
		if (mask != 0xffff)
			return (char*)haystack + trailing_zeros(~mask);
		haystack = (const __m128i*)haystack + 1;
	}
	__m256i vn = _mm256_set1_epi8(n);
	if ((size_t)haystack % 128) {
		__m256i a = _mm256_load_si256(haystack);
		__m256i b = _mm256_load_si256((__m256i*)haystack + 1);
		__m256i eqa = _mm256_cmpeq_epi8(a, vn);
		__m256i eqb = _mm256_cmpeq_epi8(b, vn);
		if (_mm256_movemask_epi8(_mm256_or_si256(eqa, eqb)) != -1) {
			int mask = _mm256_movemask_epi8(eqa);
			if (mask != -1)
				return (char*)haystack + trailing_zeros(~mask);
			mask = _mm256_movemask_epi8(eqb);
			return (char*)haystack + 32 + trailing_zeros(~mask);
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
		if (_mm256_movemask_epi8(or3) != -1) {
			int mask;
			if ((mask = _mm256_movemask_epi8(eqa)) != -1)
				return (char*)ptr + trailing_zeros(~mask);
			if ((mask = _mm256_movemask_epi8(eqb)) != -1)
				return (char*)(ptr+1) + trailing_zeros(~mask);
			if ((mask = _mm256_movemask_epi8(eqc)) != -1)
				return (char*)(ptr+2) + trailing_zeros(~mask);
			if ((mask = _mm256_movemask_epi8(eqd)) != -1)
				return (char*)(ptr+3) + trailing_zeros(~mask);
			return NULL;
		}
		ptr += 4;
	}
}

static char* strspn2_fallback(const void *haystack, int n1, int n2) {
	while ((size_t)haystack % sizeof(size_t)) {
		if (*(unsigned char*)haystack != n1 && *(unsigned char*)haystack != n2)
			return (void*)haystack;
		haystack = (char*)haystack + 1;
	}
	size_t lowbits = ~(size_t)0 / 0xff;
	size_t repeated_n1 = lowbits * n1;
	size_t repeated_n2 = lowbits * n2;
	for (;;) {
		size_t m1 = *(const size_t*)haystack ^ repeated_n1;
		size_t m2 = *(const size_t*)haystack ^ repeated_n2;
		if (m1 != 0 && m2 != 0) {
			int i = 0;
			do {
				int z1 = trailing_zeros(m1) / 8;
				int z2 = trailing_zeros(m2) / 8;
				int z = z1 > z2 ? z1 : z2;
				if (z == 0)
					return (char*)haystack + i;
				i += z;
				m1 >>= z * 8;
				m2 >>= z * 8;
			} while (m1 != 0 && m2 != 0 && i < sizeof(size_t));
		}
		haystack = (const size_t*)haystack + 1;
	}
}

__attribute__((__target__("sse2")))
static char* strspn2_sse2(const void *haystack, int n1, int n2) {
	while ((size_t)haystack % sizeof(size_t)) {
		if (*(unsigned char*)haystack != n1 && *(unsigned char*)haystack != n2)
			return (void*)haystack;
		haystack = (char*)haystack + 1;
	}
	uint32_t lowbits = ~(uint32_t)0 / 0xff;
	uint32_t repeated_n1 = lowbits * n1;
	uint32_t repeated_n2 = lowbits * n2;
	while ((size_t)haystack % 16) {
		uint32_t m1 = *(const uint32_t*)haystack ^ repeated_n1;
		uint32_t m2 = *(const uint32_t*)haystack ^ repeated_n2;
		if (m1 != 0 && m2 != 0) {
			int i = 0;
			do {
				int z1 = trailing_zeros(m1) / 8;
				int z2 = trailing_zeros(m2) / 8;
				int z = z1 > z2 ? z1 : z2;
				if (z == 0)
					return (char*)haystack + i;
				i += z;
				m1 >>= z * 8;
				m2 >>= z * 8;
			} while (m1 != 0 && m2 != 0 && i < sizeof(uint32_t));
		}
		haystack = (uint32_t*)haystack + 1;
	}
	__m128i vn1 = _mm_set1_epi8(n1);
	__m128i vn2 = _mm_set1_epi8(n2);
	if ((size_t)haystack % 32) {
		__m128i x = _mm_load_si128(haystack);
		__m128i eq1 = _mm_cmpeq_epi8(x, vn1);
		__m128i eq2 = _mm_cmpeq_epi8(x, vn2);
		if (_mm_movemask_epi8(_mm_or_si128(eq1, eq2)) != 0xffff) {
			int mask1 = _mm_movemask_epi8(eq1);
			int mask2 = _mm_movemask_epi8(eq2);
			return (char*)haystack + trailing_zeros(~mask1 & ~mask2);
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

		if (_mm_movemask_epi8(or3) != 0xffff) {
			int mask1, mask2;
			mask1 = _mm_movemask_epi8(eqa1);
			mask2 = _mm_movemask_epi8(eqa2);
			if (mask1 != 0xffff || mask2 != 0xffff)
				return (char*)ptr + trailing_zeros(~mask1 & ~mask2);
			mask1 = _mm_movemask_epi8(eqb1);
			mask2 = _mm_movemask_epi8(eqb2);
			return (char*)(ptr+1) + trailing_zeros(~mask1 & ~mask2);
		}
		ptr += 2;
	}
}

__attribute__((__target__("avx2")))
static char* strspn2_avx2(const void *haystack, int n1, int n2) {
	while ((size_t)haystack % sizeof(size_t)) {
		if (*(unsigned char*)haystack != n1 && *(unsigned char*)haystack != n2)
			return (void*)haystack;
		haystack = (char*)haystack + 1;
	}
	uint32_t lowbits = ~(uint32_t)0 / 0xff;
	uint32_t repeated_n1 = lowbits * n1;
	uint32_t repeated_n2 = lowbits * n2;
	while ((size_t)haystack % 16) {
		uint32_t m1 = *(const uint32_t*)haystack ^ repeated_n1;
		uint32_t m2 = *(const uint32_t*)haystack ^ repeated_n2;
		if (m1 != 0 && m2 != 0) {
			int i = 0;
			do {
				int z1 = trailing_zeros(m1) / 8;
				int z2 = trailing_zeros(m2) / 8;
				int z = z1 > z2 ? z1 : z2;
				if (z == 0)
					return (char*)haystack + i;
				i += z;
				m1 >>= z * 8;
				m2 >>= z * 8;
			} while (m1 != 0 && m2 != 0 && i < sizeof(uint32_t));
		}
		haystack = (uint32_t*)haystack + 1;
	}
	__m128i v16n1 = _mm_set1_epi8(n1);
	__m128i v16n2 = _mm_set1_epi8(n2);
	while ((size_t)haystack % 64) {
		__m128i x = _mm_load_si128(haystack);
		__m128i eq1 = _mm_cmpeq_epi8(x, v16n1);
		__m128i eq2 = _mm_cmpeq_epi8(x, v16n2);
		if (_mm_movemask_epi8(_mm_or_si128(eq1, eq2)) != 0xffff) {
			int mask1 = _mm_movemask_epi8(eq1);
			int mask2 = _mm_movemask_epi8(eq2);
			return (char*)haystack + trailing_zeros(~mask1 & ~mask2);
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
		__m256i or1 = _mm256_or_si256(eqa1, eqa2);
		__m256i or2 = _mm256_or_si256(eqb1, eqb2);
		__m256i or3 = _mm256_and_si256(or1, or2);

		if (_mm256_movemask_epi8(or3) != -1) {
			int mask1, mask2;
			mask1 = _mm256_movemask_epi8(eqa1);
			mask2 = _mm256_movemask_epi8(eqa2);
			if (mask1 != -1 || mask2 != -1)
				return (char*)ptr + trailing_zeros(~mask1 & ~mask2);
			mask1 = _mm256_movemask_epi8(eqb1);
			mask2 = _mm256_movemask_epi8(eqb2);
			return (char*)(ptr+1) + trailing_zeros(~mask1 & ~mask2);
		}
		ptr += 2;
	}
}

static char *strspn1_auto(const void *, int);
static char *strspn2_auto(const void *, int, int);

static char *(*strspn1_impl)(const void *, int) = strspn1_auto;
static char *(*strspn2_impl)(const void *, int, int) = strspn2_auto;

static void strspn_init() {
	if (has_avx2()) {
		strspn1_impl = strspn1_avx2;
		strspn2_impl = strspn2_avx2;
	}
	else if (has_sse2()) {
		strspn1_impl = strspn1_sse2;
		strspn2_impl = strspn2_sse2;
	}
	else {
		strspn1_impl = strspn1_fallback;
		strspn2_impl = strspn2_fallback;
	}
}

static char *strspn1_auto(const void *haystack, int n1) {
	strspn_init();
	return strspn1_impl(haystack, n1);
}
static char *strspn2_auto(const void *haystack, int n1, int n2) {
	strspn_init();
	return strspn2_impl(haystack, n1, n2);
}

size_t strspn(const char *s, const char *accept) {
	if (!accept[0])
		return 0;
	else if (!accept[1])
		return strspn1_impl(s, (unsigned char)accept[0]) - s;
	else if (!accept[2])
		return strspn2_impl(s, (unsigned char)accept[0], (unsigned char)accept[1]) - s;
	uint64_t byteset[4] = {0};
	while (*accept) {
		byteset[(unsigned char)*accept/64] |= (uint64_t)1 << (unsigned char)*accept%64;
		accept++;
	}
	size_t i = 0;
	while (byteset[(unsigned char)s[i]/64] & (uint64_t)1 << (unsigned char)s[i]%64)
		i++;
	return i;
}
