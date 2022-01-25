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

static char *stpcpy_naive(char *restrict dest, const char *restrict src) {
	for (size_t i = 0;; i++) {
		if (!(dest[i] = src[i]))
			return dest+i;
	}
}

__attribute__((__target__("sse2")))
static char *stpcpy_sse2(char *restrict dest, const char *restrict src) {
	const __m128i zero = _mm_set1_epi8(0);
	size_t padding = 15 - ((uintptr_t)(src-1) % 16);
	if (padding) {
		for (int i = 0; i < padding; i++)
			if (!(dest[i] = src[i]))
				return dest + i;
		src += padding;
		dest += padding;
	}
	padding = 31 - ((uintptr_t)(src-1) % 32);
	if (padding) {
		__m128i chunk = _mm_load_si128((const __m128i*)src);
		int o = _mm_movemask_epi8(_mm_cmpeq_epi8(chunk, zero));
		if (o) {
			o = trailing_zeros(o);
			for (int i = 0; i <= o; i++)
				dest[i] = src[i];
			return dest + o;
		}
		_mm_storeu_si128((__m128i*)dest, chunk);
		src += padding;
		dest += padding;
	}
	for (;;) {
		__m128i chunk1 = _mm_load_si128((const __m128i*)(src)+0);
		__m128i chunk2 = _mm_load_si128((const __m128i*)(src)+1);
		__m128i z1 = _mm_cmpeq_epi8(chunk1, zero);
		__m128i z2 = _mm_cmpeq_epi8(chunk2, zero);
		if (_mm_movemask_epi8(_mm_or_si128(z1, z2))) {
			int o = _mm_movemask_epi8(z1);
			if (o) {
				o = trailing_zeros(o);
				for (int i = 0; i <= o; i++)
					dest[i] = src[i];
				return dest + o;
			}
			else {
				o = trailing_zeros(_mm_movemask_epi8(z2));
				_mm_storeu_si128((__m128i*)(dest)+0, chunk1);
				for (int i = 0; i <= o; i++)
					dest[16+i] = src[16+i];
				return dest + 16 + o;
			}
		}
		_mm_storeu_si128((__m128i*)(dest)+0, chunk1);
		_mm_storeu_si128((__m128i*)(dest)+1, chunk2);
		src += 32;
		dest += 32;
	}
}

static char *stpcpy_auto(char *restrict dest, const char *restrict src);

static char *(*stpcpy_impl)(char *restrict dest, const char *restrict src) = stpcpy_auto;

static char *stpcpy_auto(char *restrict dest, const char *restrict src) {
	if (has_sse2())
		stpcpy_impl = stpcpy_sse2;
	else
		stpcpy_impl = stpcpy_naive;
	return stpcpy_impl(dest, src);
}

char *__stpcpy(char *restrict dest, const char *restrict src) {
	return stpcpy_impl(dest, src);
}

weak_alias(__stpcpy, stpcpy);
