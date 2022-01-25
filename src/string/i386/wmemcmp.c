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

#include <wchar.h>
#include <immintrin.h>

#include "cpu_features.h"
#include "helpers.h"

static int wmemcmp_naive(const wchar_t *s1, const wchar_t *s2, size_t n)
{
	while (n) {
		if (*s1 != *s2)
			return *s1-*s2;
		s1++;
		s2++;
		n--;
	}
	return 0;
}

__attribute__((__target__("sse2")))
static int wmemcmp_sse2(const wchar_t *s1, const wchar_t *s2, size_t n)
{
	const wchar_t *l = s1;
	const wchar_t *r = s2;
	while (n >= 32) {
		__m128i l1 = _mm_loadu_si128((const __m128i*)l);
		__m128i r1 = _mm_loadu_si128((const __m128i*)r);
		__m128i l2 = _mm_loadu_si128((const __m128i*)l+1);
		__m128i r2 = _mm_loadu_si128((const __m128i*)r+1);
		__m128i l3 = _mm_loadu_si128((const __m128i*)l+2);
		__m128i r3 = _mm_loadu_si128((const __m128i*)r+2);
		__m128i l4 = _mm_loadu_si128((const __m128i*)l+3);
		__m128i r4 = _mm_loadu_si128((const __m128i*)r+3);
		__m128i l5 = _mm_loadu_si128((const __m128i*)l+4);
		__m128i r5 = _mm_loadu_si128((const __m128i*)r+4);
		__m128i l6 = _mm_loadu_si128((const __m128i*)l+5);
		__m128i r6 = _mm_loadu_si128((const __m128i*)r+5);
		__m128i l7 = _mm_loadu_si128((const __m128i*)l+6);
		__m128i r7 = _mm_loadu_si128((const __m128i*)r+6);
		__m128i l8 = _mm_loadu_si128((const __m128i*)l+7);
		__m128i r8 = _mm_loadu_si128((const __m128i*)r+7);
		__m128i n1 = _mm_and_si128(_mm_cmpeq_epi8(l1, r1), _mm_cmpeq_epi8(l2, r2));
		__m128i n2 = _mm_and_si128(_mm_cmpeq_epi8(l3, r3), _mm_cmpeq_epi8(l4, r4));
		__m128i n3 = _mm_and_si128(_mm_cmpeq_epi8(l5, r5), _mm_cmpeq_epi8(l6, r6));
		__m128i n4 = _mm_and_si128(_mm_cmpeq_epi8(l7, r7), _mm_cmpeq_epi8(l8, r8));
		if (_mm_movemask_epi8(_mm_and_si128(_mm_and_si128(n1, n2), _mm_and_si128(n3, n4))) != 0xffff) {
			int o;
			if ((o = _mm_movemask_epi8(n1)) != 0xffff) {
				o = trailing_zeros(~o) / sizeof(wchar_t);
				l += o;
				r += o;
			}
			else if ((o = _mm_movemask_epi8(n2)) != 0xffff) {
				o = trailing_zeros(~o) / sizeof(wchar_t) + 8;
				l += o;
				r += o;
			}
			else if ((o = _mm_movemask_epi8(n3)) != 0xffff) {
				o = trailing_zeros(~o) / sizeof(wchar_t) + 16;
				l += o;
				r += o;
			}
			else if ((o = _mm_movemask_epi8(n4)) != 0xffff) {
				o = trailing_zeros(~o) / sizeof(wchar_t) + 24;
				l += o;
				r += o;
			}
			l1 = _mm_loadu_si128((const __m128i*)l);
			r1 = _mm_loadu_si128((const __m128i*)r);
			o = trailing_zeros(~_mm_movemask_epi8(_mm_cmpeq_epi8(l1, r1))) / sizeof(wchar_t);
			l += o;
			r += o;
			while (*l == *r) {
				l++;
				r++;
			}
			return *l-*r;
		}
		l += 32;
		r += 32;
		n -= 32;
	}
	if (n >= 16) {
		__m128i l1 = _mm_loadu_si128((const __m128i*)l);
		__m128i r1 = _mm_loadu_si128((const __m128i*)r);
		__m128i l2 = _mm_loadu_si128((const __m128i*)l+1);
		__m128i r2 = _mm_loadu_si128((const __m128i*)r+1);
		__m128i l3 = _mm_loadu_si128((const __m128i*)l+2);
		__m128i r3 = _mm_loadu_si128((const __m128i*)r+2);
		__m128i l4 = _mm_loadu_si128((const __m128i*)l+3);
		__m128i r4 = _mm_loadu_si128((const __m128i*)r+3);
		__m128i n1 = _mm_cmpeq_epi8(l1, r1);
		__m128i n2 = _mm_cmpeq_epi8(l2, r2);
		__m128i n3 = _mm_cmpeq_epi8(l3, r3);
		__m128i n4 = _mm_cmpeq_epi8(l4, r4);
		if (_mm_movemask_epi8(_mm_and_si128(_mm_and_si128(n1, n2), _mm_and_si128(n3, n4))) != 0xffff) {
			int o;
			if ((o = _mm_movemask_epi8(n1)) != 0xffff)
				o = trailing_zeros(~o) / sizeof(wchar_t);
			else if ((o = _mm_movemask_epi8(n2)) != 0xffff)
				o = trailing_zeros(~o) / sizeof(wchar_t) + 4;
			else if ((o = _mm_movemask_epi8(n3)) != 0xffff)
				o = trailing_zeros(~o) / sizeof(wchar_t) + 8;
			else
				o = trailing_zeros(~_mm_movemask_epi8(n2)) / sizeof(wchar_t) + 12;
			return l[o]-r[o];
		}
		l += 16;
		r += 16;
		n -= 16;
	}
	if (n >= 8) {
		__m128i l1 = _mm_loadu_si128((const __m128i*)l);
		__m128i r1 = _mm_loadu_si128((const __m128i*)r);
		__m128i l2 = _mm_loadu_si128((const __m128i*)l+1);
		__m128i r2 = _mm_loadu_si128((const __m128i*)r+1);
		__m128i n1 = _mm_cmpeq_epi8(l1, r1);
		__m128i n2 = _mm_cmpeq_epi8(l2, r2);
		if (_mm_movemask_epi8(_mm_and_si128(n1, n2)) != 0xffff) {
			int o;
			if ((o = _mm_movemask_epi8(n1)) != 0xffff)
				o = trailing_zeros(~o) / sizeof(wchar_t);
			else
				o = trailing_zeros(~_mm_movemask_epi8(n2)) / sizeof(wchar_t) + 4;
			return l[o]-r[o];
		}
		l += 8;
		r += 8;
		n -= 8;
	}
	while (n >= 4) {
		__m128i l1 = _mm_loadu_si128((const __m128i*)l);
		__m128i r1 = _mm_loadu_si128((const __m128i*)r);
		int o = _mm_movemask_epi8(_mm_cmpeq_epi8(l1, r1));
		if (o != 0xffff) {
			o = trailing_zeros(~o) / sizeof(wchar_t);
			return l[o]-r[o];
		}
		l += 4;
		r += 4;
		n -= 4;
	}
	return wmemcmp_naive(l, r, n);
}

__attribute__((__target__("avx2")))
static int wmemcmp_avx2(const wchar_t *s1, const wchar_t *s2, size_t n)
{
	const wchar_t *l = s1;
	const wchar_t *r = s2;
#ifdef __x86_64__
	while (n >= 64) {
		__m256i l1 = _mm256_loadu_si256((const __m256i*)l);
		__m256i r1 = _mm256_loadu_si256((const __m256i*)r);
		__m256i l2 = _mm256_loadu_si256((const __m256i*)l+1);
		__m256i r2 = _mm256_loadu_si256((const __m256i*)r+1);
		__m256i l3 = _mm256_loadu_si256((const __m256i*)l+2);
		__m256i r3 = _mm256_loadu_si256((const __m256i*)r+2);
		__m256i l4 = _mm256_loadu_si256((const __m256i*)l+3);
		__m256i r4 = _mm256_loadu_si256((const __m256i*)r+3);
		__m256i l5 = _mm256_loadu_si256((const __m256i*)l+4);
		__m256i r5 = _mm256_loadu_si256((const __m256i*)r+4);
		__m256i l6 = _mm256_loadu_si256((const __m256i*)l+5);
		__m256i r6 = _mm256_loadu_si256((const __m256i*)r+5);
		__m256i l7 = _mm256_loadu_si256((const __m256i*)l+6);
		__m256i r7 = _mm256_loadu_si256((const __m256i*)r+6);
		__m256i l8 = _mm256_loadu_si256((const __m256i*)l+7);
		__m256i r8 = _mm256_loadu_si256((const __m256i*)r+7);
		_mm_prefetch(l+64, _MM_HINT_NTA);
		_mm_prefetch(r+64, _MM_HINT_NTA);
		__m256i n1 = _mm256_and_si256(_mm256_cmpeq_epi8(l1, r1), _mm256_cmpeq_epi8(l2, r2));
		__m256i n2 = _mm256_and_si256(_mm256_cmpeq_epi8(l3, r3), _mm256_cmpeq_epi8(l4, r4));
		__m256i n3 = _mm256_and_si256(_mm256_cmpeq_epi8(l5, r5), _mm256_cmpeq_epi8(l6, r6));
		__m256i n4 = _mm256_and_si256(_mm256_cmpeq_epi8(l7, r7), _mm256_cmpeq_epi8(l8, r8));
		if (_mm256_movemask_epi8(_mm256_and_si256(_mm256_and_si256(n1, n2), _mm256_and_si256(n3, n4))) != -1) {
			int o;
			if ((o = _mm256_movemask_epi8(n1)) != -1) {
				o = trailing_zeros(~o) / sizeof(wchar_t);
				l += o;
				r += o;
			}
			else if ((o = _mm256_movemask_epi8(n2)) != -1) {
				o = trailing_zeros(~o) / sizeof(wchar_t) + 16;
				l += o;
				r += o;
			}
			else if ((o = _mm256_movemask_epi8(n3)) != -1) {
				o = trailing_zeros(~o) / sizeof(wchar_t) + 32;
				l += o;
				r += o;
			}
			else if ((o = _mm256_movemask_epi8(n4)) != -1) {
				o = trailing_zeros(~o) / sizeof(wchar_t) + 48;
				l += o;
				r += o;
			}
			l1 = _mm256_loadu_si256((const __m256i*)l);
			r1 = _mm256_loadu_si256((const __m256i*)r);
			o = _mm256_movemask_epi8(_mm256_cmpeq_epi8(l1, r1));
			o = o == -1 ? 8 : (trailing_zeros(~o) / sizeof(wchar_t));
			l += o;
			r += o;
			while (*l == *r) {
				l++;
				r++;
			}
			return *l-*r;
		}
		l += 64;
		r += 64;
		n -= 64;
	}
#endif
	while (n >= 32) {
		__m256i l1 = _mm256_loadu_si256((const __m256i*)l);
		__m256i r1 = _mm256_loadu_si256((const __m256i*)r);
		__m256i l2 = _mm256_loadu_si256((const __m256i*)l+1);
		__m256i r2 = _mm256_loadu_si256((const __m256i*)r+1);
		__m256i l3 = _mm256_loadu_si256((const __m256i*)l+2);
		__m256i r3 = _mm256_loadu_si256((const __m256i*)r+2);
		__m256i l4 = _mm256_loadu_si256((const __m256i*)l+3);
		__m256i r4 = _mm256_loadu_si256((const __m256i*)r+3);
		_mm_prefetch(l+64, _MM_HINT_NTA);
		_mm_prefetch(r+64, _MM_HINT_NTA);
		__m256i n1 = _mm256_cmpeq_epi8(l1, r1);
		__m256i n2 = _mm256_cmpeq_epi8(l2, r2);
		__m256i n3 = _mm256_cmpeq_epi8(l3, r3);
		__m256i n4 = _mm256_cmpeq_epi8(l4, r4);
		if (_mm256_movemask_epi8(_mm256_and_si256(_mm256_and_si256(n1, n2), _mm256_and_si256(n3, n4))) != -1) {
			int o;
			if ((o = _mm256_movemask_epi8(n1)) != -1)
				o = trailing_zeros(~o) / sizeof(wchar_t);
			else if ((o = _mm256_movemask_epi8(n2)) != -1)
				o = trailing_zeros(~o) / sizeof(wchar_t) + 8;
			else if ((o = _mm256_movemask_epi8(n3)) != -1)
				o = trailing_zeros(~o) / sizeof(wchar_t) + 16;
			else if ((o = _mm256_movemask_epi8(n4)) != -1)
				o = trailing_zeros(~o) / sizeof(wchar_t) + 24;
			return l[o]-r[o];
		}
		l += 32;
		r += 32;
		n -= 32;
	}
	if (n >= 16) {
		__m256i l1 = _mm256_loadu_si256((const __m256i*)l);
		__m256i r1 = _mm256_loadu_si256((const __m256i*)r);
		__m256i l2 = _mm256_loadu_si256((const __m256i*)l+1);
		__m256i r2 = _mm256_loadu_si256((const __m256i*)r+1);
		__m256i n1 = _mm256_cmpeq_epi8(l1, r1);
		__m256i n2 = _mm256_cmpeq_epi8(l2, r2);
		if (_mm256_movemask_epi8(_mm256_and_si256(n1, n2)) != -1) {
			int o;
			if ((o = _mm256_movemask_epi8(n1)) != -1)
				o = trailing_zeros(~o) / sizeof(wchar_t);
			else
				o = trailing_zeros(~_mm256_movemask_epi8(n2)) / sizeof(wchar_t) + 8;
			return l[o]-r[o];
		}
		l += 16;
		r += 16;
		n -= 16;
	}
	if (n >= 8) {
		__m256i l1 = _mm256_loadu_si256((const __m256i*)l);
		__m256i r1 = _mm256_loadu_si256((const __m256i*)r);
		int o = _mm256_movemask_epi8(_mm256_cmpeq_epi8(l1, r1));
		if (o != -1) {
			o = trailing_zeros(~o) / sizeof(wchar_t);
			return l[o]-r[o];
		}
		l += 8;
		r += 8;
		n -= 8;
	}
	if (n >= 4) {
		__m128i l1 = _mm_loadu_si128((const __m128i*)l);
		__m128i r1 = _mm_loadu_si128((const __m128i*)r);
		int o = _mm_movemask_epi8(_mm_cmpeq_epi8(l1, r1));
		if (o != 0xffff) {
			o = trailing_zeros(~o) / sizeof(wchar_t);
			return l[o]-r[o];
		}
		l += 4;
		r += 4;
		n -= 4;
	}
	return wmemcmp_naive(l, r, n);
}

static int wmemcmp_auto(const wchar_t *s1, const wchar_t *s2, size_t n);

static int (*wmemcmp_impl)(const wchar_t *s1, const wchar_t *s2, size_t n) = wmemcmp_auto;

static int wmemcmp_auto(const wchar_t *s1, const wchar_t *s2, size_t n) {
	if (has_avx2())
		wmemcmp_impl = wmemcmp_avx2;
	else if (has_sse2())
		wmemcmp_impl = wmemcmp_sse2;
	else
		wmemcmp_impl = wmemcmp_naive;
	return wmemcmp_impl(s1, s2, n);
}

int wmemcmp(const wchar_t *s1, const wchar_t *s2, size_t n) {
	return wmemcmp_impl(s1, s2, n);
}
