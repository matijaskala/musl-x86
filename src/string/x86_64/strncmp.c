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

static int strncmp_naive(const char *s1, const char *s2, size_t n)
{
	const unsigned char *l = (const unsigned char *)s1;
	const unsigned char *r = (const unsigned char *)s2;
	while (n) {
		if (*l != *r)
			return *l-*r;
		if (!*l)
			break;
		l++;
		r++;
		n--;
	}
	return 0;
}

__attribute__((__target__("sse2")))
static int strncmp_sse2(const char *s1, const char *s2, size_t n)
{
	const size_t padding1 = 127 - ((uintptr_t)(s1-1) % 128);
	const size_t padding2 = 127 - ((uintptr_t)(s2-1) % 128);
	size_t padding = padding1 < padding2 ? padding1 : padding2;
	const unsigned char *l = (const unsigned char *)s1;
	const unsigned char *r = (const unsigned char *)s2;
	const __m128i zero = _mm_set1_epi8(0);
	if (padding >= 64 && n >= 64) {
		__m128i l1 = _mm_loadu_si128((const __m128i*)l);
		__m128i r1 = _mm_loadu_si128((const __m128i*)r);
		__m128i l2 = _mm_loadu_si128((const __m128i*)l+1);
		__m128i r2 = _mm_loadu_si128((const __m128i*)r+1);
		__m128i l3 = _mm_loadu_si128((const __m128i*)l+2);
		__m128i r3 = _mm_loadu_si128((const __m128i*)r+2);
		__m128i l4 = _mm_loadu_si128((const __m128i*)l+3);
		__m128i r4 = _mm_loadu_si128((const __m128i*)r+3);
		__m128i n1 = _mm_andnot_si128(_mm_cmpeq_epi8(l1, zero), _mm_cmpeq_epi8(l1, r1));
		__m128i n2 = _mm_andnot_si128(_mm_cmpeq_epi8(l2, zero), _mm_cmpeq_epi8(l2, r2));
		__m128i n3 = _mm_andnot_si128(_mm_cmpeq_epi8(l3, zero), _mm_cmpeq_epi8(l3, r3));
		__m128i n4 = _mm_andnot_si128(_mm_cmpeq_epi8(l4, zero), _mm_cmpeq_epi8(l4, r4));
		if (_mm_movemask_epi8(_mm_and_si128(_mm_and_si128(n1, n2), _mm_and_si128(n3, n4))) != 0xffff) {
			int o;
			if ((o = _mm_movemask_epi8(n1)) != 0xffff)
				o = trailing_zeros(~o);
			else if ((o = _mm_movemask_epi8(n2)) != 0xffff)
				o = trailing_zeros(~o) + 16;
			else if ((o = _mm_movemask_epi8(n3)) != 0xffff)
				o = trailing_zeros(~o) + 32;
			else
				o = trailing_zeros(~_mm_movemask_epi8(n4)) + 48;
			return l[o]-r[o];
		}
		l += 64;
		r += 64;
		n -= 64;
		padding -= 64;
	}
	if (padding >= 32 && n >= 32) {
		__m128i l1 = _mm_loadu_si128((const __m128i*)l);
		__m128i r1 = _mm_loadu_si128((const __m128i*)r);
		__m128i l2 = _mm_loadu_si128((const __m128i*)l+1);
		__m128i r2 = _mm_loadu_si128((const __m128i*)r+1);
		__m128i n1 = _mm_andnot_si128(_mm_cmpeq_epi8(l1, zero), _mm_cmpeq_epi8(l1, r1));
		__m128i n2 = _mm_andnot_si128(_mm_cmpeq_epi8(l2, zero), _mm_cmpeq_epi8(l2, r2));
		if (_mm_movemask_epi8(_mm_and_si128(n1, n2)) != 0xffff) {
			int o;
			if ((o = _mm_movemask_epi8(n1)) != 0xffff)
				o = trailing_zeros(~o);
			else
				o = trailing_zeros(~_mm_movemask_epi8(n2)) + 16;
			return l[o]-r[o];
		}
		l += 32;
		r += 32;
		n -= 32;
		padding -= 32;
	}
	if (padding >= 16 && n >= 16) {
		__m128i l1 = _mm_loadu_si128((const __m128i*)l);
		__m128i r1 = _mm_loadu_si128((const __m128i*)r);
		int o = _mm_movemask_epi8(_mm_andnot_si128(_mm_cmpeq_epi8(l1, zero), _mm_cmpeq_epi8(l1, r1)));
		if (o != 0xffff) {
			o = trailing_zeros(~o);
			return l[o]-r[o];
		}
		l += 16;
		r += 16;
		n -= 16;
		padding -= 16;
	}
	while (padding && n) {
		if (*l != *r)
			return *l-*r;
		if (!*l)
			return 0;
		l++;
		r++;
		n--;
		padding--;
	}
	if (padding1 == padding2)
		while (n >= 64) {
			__m128i l1 = _mm_load_si128((const __m128i*)l);
			__m128i r1 = _mm_load_si128((const __m128i*)r);
			__m128i l2 = _mm_load_si128((const __m128i*)l+1);
			__m128i r2 = _mm_load_si128((const __m128i*)r+1);
			__m128i l3 = _mm_load_si128((const __m128i*)l+2);
			__m128i r3 = _mm_load_si128((const __m128i*)r+2);
			__m128i l4 = _mm_load_si128((const __m128i*)l+3);
			__m128i r4 = _mm_load_si128((const __m128i*)r+3);
			__m128i n1 = _mm_andnot_si128(_mm_cmpeq_epi8(l1, zero), _mm_cmpeq_epi8(l1, r1));
			__m128i n2 = _mm_andnot_si128(_mm_cmpeq_epi8(l2, zero), _mm_cmpeq_epi8(l2, r2));
			__m128i n3 = _mm_andnot_si128(_mm_cmpeq_epi8(l3, zero), _mm_cmpeq_epi8(l3, r3));
			__m128i n4 = _mm_andnot_si128(_mm_cmpeq_epi8(l4, zero), _mm_cmpeq_epi8(l4, r4));
			if (_mm_movemask_epi8(_mm_and_si128(_mm_and_si128(n1, n2), _mm_and_si128(n3, n4))) != 0xffff) {
				int o;
				if ((o = _mm_movemask_epi8(n1)) != 0xffff)
					o = trailing_zeros(~o);
				else if ((o = _mm_movemask_epi8(n2)) != 0xffff)
					o = trailing_zeros(~o) + 16;
				else if ((o = _mm_movemask_epi8(n3)) != 0xffff)
					o = trailing_zeros(~o) + 32;
				else
					o = trailing_zeros(~_mm_movemask_epi8(n4)) + 48;
				return l[o]-r[o];
			}
			l += 64;
			r += 64;
			n -= 64;
		}
	for (;;) {
		const size_t padding1 = 127 - ((uintptr_t)(l-1) % 128);
		const size_t padding2 = 127 - ((uintptr_t)(r-1) % 128);
		padding = padding1 | padding2;
		if (!padding)
			padding = 128;
		if (padding >= 64 && n >= 64) {
			__m128i l1 = _mm_loadu_si128((const __m128i*)l);
			__m128i r1 = _mm_loadu_si128((const __m128i*)r);
			__m128i l2 = _mm_loadu_si128((const __m128i*)l+1);
			__m128i r2 = _mm_loadu_si128((const __m128i*)r+1);
			__m128i l3 = _mm_loadu_si128((const __m128i*)l+2);
			__m128i r3 = _mm_loadu_si128((const __m128i*)r+2);
			__m128i l4 = _mm_loadu_si128((const __m128i*)l+3);
			__m128i r4 = _mm_loadu_si128((const __m128i*)r+3);
			__m128i n1 = _mm_andnot_si128(_mm_cmpeq_epi8(l1, zero), _mm_cmpeq_epi8(l1, r1));
			__m128i n2 = _mm_andnot_si128(_mm_cmpeq_epi8(l2, zero), _mm_cmpeq_epi8(l2, r2));
			__m128i n3 = _mm_andnot_si128(_mm_cmpeq_epi8(l3, zero), _mm_cmpeq_epi8(l3, r3));
			__m128i n4 = _mm_andnot_si128(_mm_cmpeq_epi8(l4, zero), _mm_cmpeq_epi8(l4, r4));
			if (_mm_movemask_epi8(_mm_and_si128(_mm_and_si128(n1, n2), _mm_and_si128(n3, n4))) != 0xffff) {
				int o;
				if ((o = _mm_movemask_epi8(n1)) != 0xffff)
					o = trailing_zeros(~o);
				else if ((o = _mm_movemask_epi8(n2)) != 0xffff)
					o = trailing_zeros(~o) + 16;
				else if ((o = _mm_movemask_epi8(n3)) != 0xffff)
					o = trailing_zeros(~o) + 32;
				else
					o = trailing_zeros(~_mm_movemask_epi8(n4)) + 48;
				return l[o]-r[o];
			}
			l += 64;
			r += 64;
			n -= 64;
			padding -= 64;
		}
		if (padding >= 32 && n >= 32) {
			__m128i l1 = _mm_loadu_si128((const __m128i*)l);
			__m128i r1 = _mm_loadu_si128((const __m128i*)r);
			__m128i l2 = _mm_loadu_si128((const __m128i*)l+1);
			__m128i r2 = _mm_loadu_si128((const __m128i*)r+1);
			__m128i n1 = _mm_andnot_si128(_mm_cmpeq_epi8(l1, zero), _mm_cmpeq_epi8(l1, r1));
			__m128i n2 = _mm_andnot_si128(_mm_cmpeq_epi8(l2, zero), _mm_cmpeq_epi8(l2, r2));
			if (_mm_movemask_epi8(_mm_and_si128(n1, n2)) != 0xffff) {
				int o;
				if ((o = _mm_movemask_epi8(n1)) != 0xffff)
					o = trailing_zeros(~o);
				else
					o = trailing_zeros(~_mm_movemask_epi8(n2)) + 16;
				return l[o]-r[o];
			}
			l += 32;
			r += 32;
			n -= 32;
			padding -= 32;
		}
		if (padding >= 16 && n >= 16) {
			__m128i l1 = _mm_loadu_si128((const __m128i*)l);
			__m128i r1 = _mm_loadu_si128((const __m128i*)r);
			int o = _mm_movemask_epi8(_mm_andnot_si128(_mm_cmpeq_epi8(l1, zero), _mm_cmpeq_epi8(l1, r1)));
			if (o != 0xffff) {
				o = trailing_zeros(~o);
				return l[o]-r[o];
			}
			l += 16;
			r += 16;
			n -= 16;
			padding -= 16;
		}
		while (padding && n) {
			if (*l != *r)
				return *l-*r;
			if (!*l)
				return 0;
			l++;
			r++;
			n--;
			padding--;
		}
		if (!n)
			return 0;
	}
}

__attribute__((__target__("avx2")))
static int strncmp_avx2(const char *s1, const char *s2, size_t n)
{
	const size_t padding1 = 127 - ((uintptr_t)(s1-1) % 128);
	const size_t padding2 = 127 - ((uintptr_t)(s2-1) % 128);
	size_t padding = padding1 < padding2 ? padding1 : padding2;
	const unsigned char *l = (const unsigned char *)s1;
	const unsigned char *r = (const unsigned char *)s2;
	const __m256i zero = _mm256_set1_epi8(0);
	const __m128i zero128 = _mm_set1_epi8(0);
	if (padding >= 64 && n >= 64) {
		__m256i l1 = _mm256_loadu_si256((const __m256i*)l);
		__m256i r1 = _mm256_loadu_si256((const __m256i*)r);
		__m256i l2 = _mm256_loadu_si256((const __m256i*)l+1);
		__m256i r2 = _mm256_loadu_si256((const __m256i*)r+1);
		__m256i n1 = _mm256_andnot_si256(_mm256_cmpeq_epi8(l1, zero), _mm256_cmpeq_epi8(l1, r1));
		__m256i n2 = _mm256_andnot_si256(_mm256_cmpeq_epi8(l2, zero), _mm256_cmpeq_epi8(l2, r2));
		if (_mm256_movemask_epi8(_mm256_and_si256(n1, n2)) != -1) {
			int o;
			if ((o = _mm256_movemask_epi8(n1)) != -1)
				o = trailing_zeros(~o);
			else
				o = trailing_zeros(~_mm256_movemask_epi8(n2)) + 32;
			return l[o]-r[o];
		}
		l += 64;
		r += 64;
		n -= 64;
		padding -= 64;
	}
	if (padding >= 32 && n >= 32) {
		__m256i l1 = _mm256_loadu_si256((const __m256i*)l);
		__m256i r1 = _mm256_loadu_si256((const __m256i*)r);
		int o = _mm256_movemask_epi8(_mm256_andnot_si256(_mm256_cmpeq_epi8(l1, zero), _mm256_cmpeq_epi8(l1, r1)));
		if (o != -1) {
			o = trailing_zeros(~o);
			return l[o]-r[o];
		}
		l += 32;
		r += 32;
		n -= 32;
		padding -= 32;
	}
	if (padding >= 16 && n >= 16) {
		__m128i l1 = _mm_loadu_si128((const __m128i*)l);
		__m128i r1 = _mm_loadu_si128((const __m128i*)r);
		int o = _mm_movemask_epi8(_mm_andnot_si128(_mm_cmpeq_epi8(l1, zero128), _mm_cmpeq_epi8(l1, r1)));
		if (o != 0xffff) {
			o = trailing_zeros(~o);
			return l[o]-r[o];
		}
		l += 16;
		r += 16;
		n -= 16;
		padding -= 16;
	}
	while (padding && n) {
		if (*l != *r)
			return *l-*r;
		if (!*l)
			return 0;
		l++;
		r++;
		n--;
		padding--;
	}
	if (padding1 == padding2)
		while (n >= 128) {
			__m256i l1 = _mm256_load_si256((const __m256i*)l);
			__m256i r1 = _mm256_load_si256((const __m256i*)r);
			__m256i l2 = _mm256_load_si256((const __m256i*)l+1);
			__m256i r2 = _mm256_load_si256((const __m256i*)r+1);
			__m256i l3 = _mm256_load_si256((const __m256i*)l+2);
			__m256i r3 = _mm256_load_si256((const __m256i*)r+2);
			__m256i l4 = _mm256_load_si256((const __m256i*)l+3);
			__m256i r4 = _mm256_load_si256((const __m256i*)r+3);
			_mm_prefetch(l+256, _MM_HINT_NTA);
			_mm_prefetch(r+256, _MM_HINT_NTA);
			__m256i n1 = _mm256_andnot_si256(_mm256_cmpeq_epi8(l1, zero), _mm256_cmpeq_epi8(l1, r1));
			__m256i n2 = _mm256_andnot_si256(_mm256_cmpeq_epi8(l2, zero), _mm256_cmpeq_epi8(l2, r2));
			__m256i n3 = _mm256_andnot_si256(_mm256_cmpeq_epi8(l3, zero), _mm256_cmpeq_epi8(l3, r3));
			__m256i n4 = _mm256_andnot_si256(_mm256_cmpeq_epi8(l4, zero), _mm256_cmpeq_epi8(l4, r4));
			if (_mm256_movemask_epi8(_mm256_and_si256(_mm256_and_si256(n1, n2), _mm256_and_si256(n3, n4))) != -1) {
				int o;
				if ((o = _mm256_movemask_epi8(n1)) != -1)
					o = trailing_zeros(~o);
				else if ((o = _mm256_movemask_epi8(n2)) != -1)
					o = trailing_zeros(~o) + 32;
				else if ((o = _mm256_movemask_epi8(n3)) != -1)
					o = trailing_zeros(~o) + 64;
				else if ((o = _mm256_movemask_epi8(n4)) != -1)
					o = trailing_zeros(~o) + 96;
				return l[o]-r[o];
			}
			l += 128;
			r += 128;
			n -= 128;
		}
	for (;;) {
		const size_t padding1 = 127 - ((uintptr_t)(l-1) % 128);
		const size_t padding2 = 127 - ((uintptr_t)(r-1) % 128);
		padding = padding1 | padding2;
		if (!padding)
			padding = 128;
		if (padding >= 64 && n >= 64) {
			__m256i l1 = _mm256_loadu_si256((const __m256i*)l);
			__m256i r1 = _mm256_loadu_si256((const __m256i*)r);
			__m256i l2 = _mm256_loadu_si256((const __m256i*)l+1);
			__m256i r2 = _mm256_loadu_si256((const __m256i*)r+1);
			__m256i n1 = _mm256_andnot_si256(_mm256_cmpeq_epi8(l1, zero), _mm256_cmpeq_epi8(l1, r1));
			__m256i n2 = _mm256_andnot_si256(_mm256_cmpeq_epi8(l2, zero), _mm256_cmpeq_epi8(l2, r2));
			if (_mm256_movemask_epi8(_mm256_and_si256(n1, n2)) != -1) {
				int o;
				if ((o = _mm256_movemask_epi8(n1)) != -1)
					o = trailing_zeros(~o);
				else
					o = trailing_zeros(~_mm256_movemask_epi8(n2)) + 32;
				return l[o]-r[o];
			}
			l += 64;
			r += 64;
			n -= 64;
			padding -= 64;
		}
		if (padding >= 32 && n >= 32) {
			__m256i l1 = _mm256_loadu_si256((const __m256i*)l);
			__m256i r1 = _mm256_loadu_si256((const __m256i*)r);
			int o = _mm256_movemask_epi8(_mm256_andnot_si256(_mm256_cmpeq_epi8(l1, zero), _mm256_cmpeq_epi8(l1, r1)));
			if (o != -1) {
				o = trailing_zeros(~o);
				return l[o]-r[o];
			}
			l += 32;
			r += 32;
			n -= 32;
			padding -= 32;
		}
		if (padding >= 16 && n >= 16) {
			__m128i l1 = _mm_loadu_si128((const __m128i*)l);
			__m128i r1 = _mm_loadu_si128((const __m128i*)r);
			int o = _mm_movemask_epi8(_mm_andnot_si128(_mm_cmpeq_epi8(l1, zero128), _mm_cmpeq_epi8(l1, r1)));
			if (o != 0xffff) {
				o = trailing_zeros(~o);
				return l[o]-r[o];
			}
			l += 16;
			r += 16;
			n -= 16;
			padding -= 16;
		}
		while (padding && n) {
			if (*l != *r)
				return *l-*r;
			if (!*l)
				return 0;
			l++;
			r++;
			n--;
			padding--;
		}
		if (!n)
			return 0;
	}
}

static int strncmp_auto(const char *s1, const char *s2, size_t n);

static int (*strncmp_impl)(const char *s1, const char *s2, size_t n) = strncmp_auto;

static int strncmp_auto(const char *s1, const char *s2, size_t n) {
	if (has_avx2())
		strncmp_impl = strncmp_avx2;
	else if (has_sse2())
		strncmp_impl = strncmp_sse2;
	else
		strncmp_impl = strncmp_naive;
	return strncmp_impl(s1, s2, n);
}

int strncmp(const char *s1, const char *s2, size_t n) {
	return strncmp_impl(s1, s2, n);
}
