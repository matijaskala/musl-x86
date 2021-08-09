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

static char *stpncpy_internal_naive(char *dest, const char *src, size_t n) {
	for (size_t i = 0; i < n; i++) {
		if (!src[i]) {
			return dest + i;
		}
		dest[i] = src[i];
	}
	return dest + n;
}

static char *stpncpy_internal_sse2(char *dest, const char *src, size_t n) {
	const __m128i zero = _mm_set1_epi8(0);
	if (n < 128) {
		if (n >= 64) {
			__m128i chunk1 = _mm_loadu_si128((const __m128i*)(src)+0);
			__m128i chunk2 = _mm_loadu_si128((const __m128i*)(src)+1);
			__m128i chunk3 = _mm_loadu_si128((const __m128i*)(src)+2);
			__m128i chunk4 = _mm_loadu_si128((const __m128i*)(src)+3);
			_mm_storeu_si128((__m128i*)(dest)+0, chunk1);
			_mm_storeu_si128((__m128i*)(dest)+1, chunk2);
			_mm_storeu_si128((__m128i*)(dest)+2, chunk3);
			_mm_storeu_si128((__m128i*)(dest)+3, chunk4);
			__m128i z1 = _mm_cmpeq_epi8(chunk1, zero);
			__m128i z2 = _mm_cmpeq_epi8(chunk2, zero);
			__m128i z3 = _mm_cmpeq_epi8(chunk3, zero);
			__m128i z4 = _mm_cmpeq_epi8(chunk4, zero);
			__m128i z12 = _mm_or_si128(z1, z2);
			__m128i z34 = _mm_or_si128(z3, z4);
			if (_mm_movemask_epi8(_mm_or_si128(z12, z34))) {
				unsigned o;
				if (_mm_movemask_epi8(z12)) {
					if ((o = _mm_movemask_epi8(z1))) {
						o = trailing_zeros(o);
						return dest + o;
					}
					else {
						o = trailing_zeros(_mm_movemask_epi8(z2));
						return dest + 16 + o;
					}
				}
				else {
					if ((o = _mm_movemask_epi8(z3))) {
						o = trailing_zeros(o);
						return dest + 32 + o;
					}
					else {
						o = trailing_zeros(_mm_movemask_epi8(z4));
						return dest + 48 + o;
					}
				}
			}
			src += 64;
			dest += 64;
			n -= 64;
		}
		if (n >= 32) {
			__m128i chunk1 = _mm_loadu_si128((const __m128i*)(src)+0);
			__m128i chunk2 = _mm_loadu_si128((const __m128i*)(src)+1);
			_mm_storeu_si128((__m128i*)(dest)+0, chunk1);
			_mm_storeu_si128((__m128i*)(dest)+1, chunk2);
			__m128i z1 = _mm_cmpeq_epi8(chunk1, zero);
			__m128i z2 = _mm_cmpeq_epi8(chunk2, zero);
			if (_mm_movemask_epi8(_mm_or_si128(z1, z2))) {
				unsigned o;
				if ((o = _mm_movemask_epi8(z1))) {
					o = trailing_zeros(o);
					return dest + o;
				}
				else {
					o = trailing_zeros(_mm_movemask_epi8(z2));
					return dest + 16 + o;
				}
			}
			src += 32;
			dest += 32;
			n -= 32;
		}
		if (n >= 16) {
			__m128i chunk1 = _mm_loadu_si128((const __m128i*)(src)+0);
			_mm_storeu_si128((__m128i*)(dest)+0, chunk1);
			unsigned o = _mm_movemask_epi8(_mm_cmpeq_epi8(chunk1, zero));
			if (o) {
				o = trailing_zeros(o);
				return dest + o;
			}
			src += 16;
			dest += 16;
			n -= 16;
		}
		return stpncpy_internal_naive(dest, src, n);
	}
	size_t padding = 15 - ((uintptr_t)(dest-1) % 16);
	if (padding) {
		__m128i chunk = _mm_loadu_si128((const __m128i*)src);
		unsigned o = _mm_movemask_epi8(_mm_cmpeq_epi8(chunk, zero));
		if (o) {
			o = trailing_zeros(o);
			_mm_storeu_si128((__m128i*)dest, chunk);
			return dest + o;
		}
		_mm_storeu_si128((__m128i*)dest, chunk);
		src += padding;
		dest += padding;
		n -= padding;
	}
	if (n >= 1 << 21) {
		padding = 127 - ((uintptr_t)(dest-1) % 128);
		for (size_t i = 0; i < padding/16; i++) {
			__m128i chunk1 = _mm_loadu_si128((const __m128i*)src+i);
			_mm_store_si128((__m128i*)dest+i, chunk1);
		}
		src += padding;
		dest += padding;
		n -= padding;
		for (size_t i = 0; i < n/64; i++) {
			__m128i chunk1 = _mm_loadu_si128((const __m128i*)(src+64*i)+0);
			__m128i chunk2 = _mm_loadu_si128((const __m128i*)(src+64*i)+1);
			__m128i chunk3 = _mm_loadu_si128((const __m128i*)(src+64*i)+2);
			__m128i chunk4 = _mm_loadu_si128((const __m128i*)(src+64*i)+3);
			_mm_prefetch(src+64*i+128, _MM_HINT_NTA);
			_mm_stream_si128((__m128i*)(dest+64*i)+0, chunk1);
			_mm_stream_si128((__m128i*)(dest+64*i)+1, chunk2);
			_mm_stream_si128((__m128i*)(dest+64*i)+2, chunk3);
			_mm_stream_si128((__m128i*)(dest+64*i)+3, chunk4);
			__m128i z1 = _mm_cmpeq_epi8(chunk1, zero);
			__m128i z2 = _mm_cmpeq_epi8(chunk2, zero);
			__m128i z3 = _mm_cmpeq_epi8(chunk3, zero);
			__m128i z4 = _mm_cmpeq_epi8(chunk4, zero);
			__m128i z12 = _mm_or_si128(z1, z2);
			__m128i z34 = _mm_or_si128(z3, z4);
			if (_mm_movemask_epi8(_mm_or_si128(z12, z34))) {
				dest += 64*i;
				unsigned o;
				if (_mm_movemask_epi8(z12)) {
					if ((o = _mm_movemask_epi8(z1))) {
						o = trailing_zeros(o);
						return dest + o;
					}
					else {
						o = trailing_zeros(_mm_movemask_epi8(z2));
						return dest + 16 + o;
					}
				}
				else {
					if ((o = _mm_movemask_epi8(z3))) {
						o = trailing_zeros(o);
						return dest + 32 + o;
					}
					else {
						o = trailing_zeros(_mm_movemask_epi8(z4));
						return dest + 48 + o;
					}
				}
			}
		}
		_mm_sfence();
		src += n & ~63;
		dest += n & ~63;
		n %= 64;
	}
	else if (n >= 1 << 7) {
		for (size_t i = 0; i < n/64; i++) {
			__m128i chunk1 = _mm_loadu_si128((const __m128i*)(src+64*i)+0);
			__m128i chunk2 = _mm_loadu_si128((const __m128i*)(src+64*i)+1);
			__m128i chunk3 = _mm_loadu_si128((const __m128i*)(src+64*i)+2);
			__m128i chunk4 = _mm_loadu_si128((const __m128i*)(src+64*i)+3);
			_mm_prefetch(src+64*i+64, _MM_HINT_NTA);
			_mm_store_si128((__m128i*)(dest+64*i)+0, chunk1);
			_mm_store_si128((__m128i*)(dest+64*i)+1, chunk2);
			_mm_store_si128((__m128i*)(dest+64*i)+2, chunk3);
			_mm_store_si128((__m128i*)(dest+64*i)+3, chunk4);
			__m128i z1 = _mm_cmpeq_epi8(chunk1, zero);
			__m128i z2 = _mm_cmpeq_epi8(chunk2, zero);
			__m128i z3 = _mm_cmpeq_epi8(chunk3, zero);
			__m128i z4 = _mm_cmpeq_epi8(chunk4, zero);
			__m128i z12 = _mm_or_si128(z1, z2);
			__m128i z34 = _mm_or_si128(z3, z4);
			if (_mm_movemask_epi8(_mm_or_si128(z12, z34))) {
				dest += 64*i;
				unsigned o;
				if (_mm_movemask_epi8(z12)) {
					if ((o = _mm_movemask_epi8(z1))) {
						o = trailing_zeros(o);
						return dest + o;
					}
					else {
						o = trailing_zeros(_mm_movemask_epi8(z2));
						return dest + 16 + o;
					}
				}
				else {
					if ((o = _mm_movemask_epi8(z3))) {
						o = trailing_zeros(o);
						return dest + 32 + o;
					}
					else {
						o = trailing_zeros(_mm_movemask_epi8(z4));
						return dest + 48 + o;
					}
				}
			}
		}
		src += n & ~63;
		dest += n & ~63;
		n %= 64;
	}
	if (n >= 32) {
		__m128i chunk1 = _mm_loadu_si128((const __m128i*)(src)+0);
		__m128i chunk2 = _mm_loadu_si128((const __m128i*)(src)+1);
		_mm_storeu_si128((__m128i*)(dest)+0, chunk1);
		_mm_storeu_si128((__m128i*)(dest)+1, chunk2);
		__m128i z1 = _mm_cmpeq_epi8(chunk1, zero);
		__m128i z2 = _mm_cmpeq_epi8(chunk2, zero);
		if (_mm_movemask_epi8(_mm_or_si128(z1, z2))) {
			unsigned o;
			if ((o = _mm_movemask_epi8(z1))) {
				o = trailing_zeros(o);
				return dest + o;
			}
			else {
				o = trailing_zeros(_mm_movemask_epi8(z2));
				return dest + 16 + o;
			}
		}
		src += 32;
		dest += 32;
		n -= 32;
	}
	if (n >= 16) {
		__m128i chunk1 = _mm_loadu_si128((const __m128i*)(src)+0);
		_mm_storeu_si128((__m128i*)(dest)+0, chunk1);
		unsigned o = _mm_movemask_epi8(_mm_cmpeq_epi8(chunk1, zero));
		if (o) {
			o = trailing_zeros(o);
			return dest + o;
		}
		src += 16;
		dest += 16;
		n -= 16;
	}
	if (n) {
		__m128i chunk = _mm_loadu_si128((__m128i*)(src+n)-1);
		_mm_storeu_si128((__m128i*)(dest+n)-1, chunk);
		unsigned o = _mm_movemask_epi8(_mm_cmpeq_epi8(chunk, zero));
		if (o) {
			o = trailing_zeros(o);
			if (o < n)
				return dest + o;
		}
	}
	return dest + n;
}

static char *stpncpy_internal_auto(char *dest, const char *src, size_t n);

static char *(*stpncpy_internal)(char *dest, const char *src, size_t n) = stpncpy_internal_auto;

static char *stpncpy_internal_auto(char *dest, const char *src, size_t n) {
	if (has_sse2())
		stpncpy_internal = stpncpy_internal_sse2;
	else
		stpncpy_internal = stpncpy_internal_naive;
	return stpncpy_internal(dest, src, n);
}

char *__stpncpy(char *dest, const char *src, size_t n) {
	char *r = stpncpy_internal(dest, src, n);
	return memset(r, 0, (dest + n) - r);
}

weak_alias(__stpncpy, stpncpy);
