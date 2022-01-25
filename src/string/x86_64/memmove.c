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
#include <stdint.h>
#include <immintrin.h>

#include "cpu_features.h"

static void *memmove_naive(void *dest, const void *src, size_t n) {
	char *d = dest;
	const char *s = src;
	if (dest > src) {
		for (size_t i = 1; i <= n; i++)
			d[n-i] = s[n-i];
	}
	else {
		for (size_t i = 0; i < n; i++)
			d[i] = s[i];
	}
	return dest;
}

__attribute__((__target__("sse2")))
static void *memmove_sse2(void *dest, const void *src, size_t n) {
	if (dest == src)
		return dest;
	char *d = dest;
	const char *s = src;
	if (n < 128) {
		if (dest > src) {
			if (n >= 64) {
				__m128i chunk1 = _mm_loadu_si128((const __m128i*)(s+n)-1);
				__m128i chunk2 = _mm_loadu_si128((const __m128i*)(s+n)-2);
				__m128i chunk3 = _mm_loadu_si128((const __m128i*)(s+n)-3);
				__m128i chunk4 = _mm_loadu_si128((const __m128i*)(s+n)-4);
				_mm_storeu_si128((__m128i*)(d+n)-1, chunk1);
				_mm_storeu_si128((__m128i*)(d+n)-2, chunk2);
				_mm_storeu_si128((__m128i*)(d+n)-3, chunk3);
				_mm_storeu_si128((__m128i*)(d+n)-4, chunk4);
				n -= 64;
			}
			if (n >= 32) {
				__m128i chunk1 = _mm_loadu_si128((const __m128i*)(s+n)-1);
				__m128i chunk2 = _mm_loadu_si128((const __m128i*)(s+n)-2);
				_mm_storeu_si128((__m128i*)(d+n)-1, chunk1);
				_mm_storeu_si128((__m128i*)(d+n)-2, chunk2);
				n -= 32;
			}
			if (n >= 16) {
				__m128i chunk1 = _mm_loadu_si128((const __m128i*)(s+n)-1);
				_mm_storeu_si128((__m128i*)(d+n)-1, chunk1);
				n -= 16;
			}
			for (size_t i = 1; i <= n; i++)
				d[n-i] = s[n-i];
		}
		else {
			if (n >= 64) {
				__m128i chunk1 = _mm_loadu_si128((const __m128i*)(s)+0);
				__m128i chunk2 = _mm_loadu_si128((const __m128i*)(s)+1);
				__m128i chunk3 = _mm_loadu_si128((const __m128i*)(s)+2);
				__m128i chunk4 = _mm_loadu_si128((const __m128i*)(s)+3);
				_mm_storeu_si128((__m128i*)(d)+0, chunk1);
				_mm_storeu_si128((__m128i*)(d)+1, chunk2);
				_mm_storeu_si128((__m128i*)(d)+2, chunk3);
				_mm_storeu_si128((__m128i*)(d)+3, chunk4);
				s += 64;
				d += 64;
				n -= 64;
			}
			if (n >= 32) {
				__m128i chunk1 = _mm_loadu_si128((const __m128i*)(s)+0);
				__m128i chunk2 = _mm_loadu_si128((const __m128i*)(s)+1);
				_mm_storeu_si128((__m128i*)(d)+0, chunk1);
				_mm_storeu_si128((__m128i*)(d)+1, chunk2);
				s += 32;
				d += 32;
				n -= 32;
			}
			if (n >= 16) {
				__m128i chunk1 = _mm_loadu_si128((const __m128i*)(s)+0);
				_mm_storeu_si128((__m128i*)(d)+0, chunk1);
				s += 16;
				d += 16;
				n -= 16;
			}
			for (size_t i = 0; i < n; i++)
				d[i] = s[i];
		}
		return dest;
	}
	if (dest > src) {
		size_t padding = (uintptr_t)(d+n) % 16;
		for (size_t i = 1; i <= padding; i++)
			d[n-i] = s[n-i];
		n -= padding;
		if (n >= 1 << 21) {
			padding = ((uintptr_t)(d+n) % 128);
			for (size_t i = 1; i <= padding/16; i++) {
				__m128i chunk1 = _mm_loadu_si128((const __m128i*)(s+n)-i);
				_mm_store_si128((__m128i*)(d+n)-i, chunk1);
			}
			n -= padding;
			while (n >= 256) {
				__m128i chunk1 = _mm_loadu_si128((const __m128i*)(s+n)-1);
				__m128i chunk2 = _mm_loadu_si128((const __m128i*)(s+n)-2);
				__m128i chunk3 = _mm_loadu_si128((const __m128i*)(s+n)-3);
				__m128i chunk4 = _mm_loadu_si128((const __m128i*)(s+n)-4);
				__m128i chunk5 = _mm_loadu_si128((const __m128i*)(s+n)-5);
				__m128i chunk6 = _mm_loadu_si128((const __m128i*)(s+n)-6);
				__m128i chunk7 = _mm_loadu_si128((const __m128i*)(s+n)-7);
				__m128i chunk8 = _mm_loadu_si128((const __m128i*)(s+n)-8);
				_mm_prefetch(s+n-384, _MM_HINT_NTA);
				_mm_stream_si128((__m128i*)(d+n)-1, chunk1);
				_mm_stream_si128((__m128i*)(d+n)-2, chunk2);
				_mm_stream_si128((__m128i*)(d+n)-3, chunk3);
				_mm_stream_si128((__m128i*)(d+n)-4, chunk4);
				_mm_stream_si128((__m128i*)(d+n)-5, chunk5);
				_mm_stream_si128((__m128i*)(d+n)-6, chunk6);
				_mm_stream_si128((__m128i*)(d+n)-7, chunk7);
				_mm_stream_si128((__m128i*)(d+n)-8, chunk8);
				n -= 128;
			}
			_mm_sfence();
		}
		while (n >= 64) {
			__m128i chunk1 = _mm_loadu_si128((const __m128i*)(s+n)-1);
			__m128i chunk2 = _mm_loadu_si128((const __m128i*)(s+n)-2);
			__m128i chunk3 = _mm_loadu_si128((const __m128i*)(s+n)-3);
			__m128i chunk4 = _mm_loadu_si128((const __m128i*)(s+n)-4);
			_mm_storeu_si128((__m128i*)(d+n)-1, chunk1);
			_mm_storeu_si128((__m128i*)(d+n)-2, chunk2);
			_mm_storeu_si128((__m128i*)(d+n)-3, chunk3);
			_mm_storeu_si128((__m128i*)(d+n)-4, chunk4);
			n -= 64;
		}
		if (n >= 32) {
			__m128i chunk1 = _mm_loadu_si128((const __m128i*)(s+n)-1);
			__m128i chunk2 = _mm_loadu_si128((const __m128i*)(s+n)-2);
			_mm_storeu_si128((__m128i*)(d+n)-1, chunk1);
			_mm_storeu_si128((__m128i*)(d+n)-2, chunk2);
			n -= 32;
		}
		if (n >= 16) {
			__m128i chunk1 = _mm_loadu_si128((const __m128i*)(s+n)-1);
			_mm_storeu_si128((__m128i*)(d+n)-1, chunk1);
			n -= 16;
		}
		for (int i = n - 1; i >= 0; i--) {
			d[i] = s[i];
		}
	}
	else {
		size_t padding = 15 - ((uintptr_t)(d-1) % 16);
		for (size_t i = 0; i < padding; i++)
			d[i] = s[i];
		s += padding;
		d += padding;
		n -= padding;
		if (n >= 1 << 21) {
			padding = 127 - ((uintptr_t)(d-1) % 128);
			for (size_t i = 0; i < padding/16; i++) {
				__m128i chunk1 = _mm_loadu_si128((const __m128i*)s+i);
				_mm_store_si128((__m128i*)d+i, chunk1);
			}
			s += padding;
			d += padding;
			n -= padding;
			while (n >= 256) {
				__m128i chunk1 = _mm_loadu_si128((const __m128i*)s+0);
				__m128i chunk2 = _mm_loadu_si128((const __m128i*)s+1);
				__m128i chunk3 = _mm_loadu_si128((const __m128i*)s+2);
				__m128i chunk4 = _mm_loadu_si128((const __m128i*)s+3);
				__m128i chunk5 = _mm_loadu_si128((const __m128i*)s+4);
				__m128i chunk6 = _mm_loadu_si128((const __m128i*)s+5);
				__m128i chunk7 = _mm_loadu_si128((const __m128i*)s+6);
				__m128i chunk8 = _mm_loadu_si128((const __m128i*)s+7);
				_mm_prefetch(s+256, _MM_HINT_NTA);
				_mm_stream_si128((__m128i*)d+0, chunk1);
				_mm_stream_si128((__m128i*)d+1, chunk2);
				_mm_stream_si128((__m128i*)d+2, chunk3);
				_mm_stream_si128((__m128i*)d+3, chunk4);
				_mm_stream_si128((__m128i*)d+4, chunk5);
				_mm_stream_si128((__m128i*)d+5, chunk6);
				_mm_stream_si128((__m128i*)d+6, chunk7);
				_mm_stream_si128((__m128i*)d+7, chunk8);
				d += 128;
				s += 128;
				n -= 128;
			}
			_mm_sfence();
		}
		while (n >= 64) {
			__m128i chunk1 = _mm_loadu_si128((const __m128i*)(s)+0);
			__m128i chunk2 = _mm_loadu_si128((const __m128i*)(s)+1);
			__m128i chunk3 = _mm_loadu_si128((const __m128i*)(s)+2);
			__m128i chunk4 = _mm_loadu_si128((const __m128i*)(s)+3);
			_mm_storeu_si128((__m128i*)(d)+0, chunk1);
			_mm_storeu_si128((__m128i*)(d)+1, chunk2);
			_mm_storeu_si128((__m128i*)(d)+2, chunk3);
			_mm_storeu_si128((__m128i*)(d)+3, chunk4);
			s += 64;
			d += 64;
			n -= 64;
		}
		if (n >= 32) {
			__m128i chunk1 = _mm_loadu_si128((const __m128i*)(s)+0);
			__m128i chunk2 = _mm_loadu_si128((const __m128i*)(s)+1);
			_mm_storeu_si128((__m128i*)(d)+0, chunk1);
			_mm_storeu_si128((__m128i*)(d)+1, chunk2);
			s += 32;
			d += 32;
			n -= 32;
		}
		if (n >= 16) {
			__m128i chunk1 = _mm_loadu_si128((const __m128i*)(s)+0);
			_mm_storeu_si128((__m128i*)(d)+0, chunk1);
			s += 16;
			d += 16;
			n -= 16;
		}
		for (size_t i = 0; i < n; i++)
			d[i] = s[i];
	}
	return dest;
}

static void *memmove_auto(void *dest, const void *src, size_t n);

static void *(*memmove_impl)(void *dest, const void *src, size_t n) = memmove_auto;

static void *memmove_auto(void *dest, const void *src, size_t n) {
	if (has_sse2())
		memmove_impl = memmove_sse2;
	else
		memmove_impl = memmove_naive;
	return memmove_impl(dest, src, n);
}

void *memmove(void *dest, const void *src, size_t n) {
	return memmove_impl(dest, src, n);
}
