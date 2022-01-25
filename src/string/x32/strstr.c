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

static char *strstr2(const uint8_t *h, const uint8_t *n)
{
	uint16_t nw = (uintmax_t)n[0] << 8 | n[1];
	uint16_t hw = (uintmax_t)h[0] << 8 | h[1];
	while (h[1]) {
		if (hw == nw)
			return (void*)h;
		hw <<= 8;
		hw |= h[2];
		h++;
	}
	return NULL;
}

static char *strstr3(const uint8_t *h, const uint8_t *n)
{
	uint32_t nw = (uintmax_t)n[0] << 24 | (uintmax_t)n[1] << 16 | (uintmax_t)n[2] << 8;
	uint32_t hw = (uintmax_t)h[0] << 24 | (uintmax_t)h[1] << 16 | (uintmax_t)h[2] << 8;
	while (h[2]) {
		if (hw == nw)
			return (void*)h;
		hw <<= 8;
		hw |= (uintmax_t)h[3] << 8;
		h++;
	}
	return NULL;
}

static char *strstr4(const uint8_t *h, const uint8_t *n)
{
	uint32_t nw = (uintmax_t)n[0] << 24 | (uintmax_t)n[1] << 16 | (uintmax_t)n[2] << 8 | n[3];
	uint32_t hw = (uintmax_t)h[0] << 24 | (uintmax_t)h[1] << 16 | (uintmax_t)h[2] << 8 | h[3];
	while (h[3]) {
		if (hw == nw)
			return (void*)h;
		hw <<= 8;
		hw |= h[4];
		h++;
	}
	return NULL;
}

static char *strstr5(const uint8_t *h, const uint8_t *n)
{
	uint64_t nw = (uintmax_t)n[0] << 56 | (uintmax_t)n[1] << 48 | (uintmax_t)n[2] << 40 | (uintmax_t)n[3] << 32 | (uintmax_t)n[4] << 24;
	uint64_t hw = (uintmax_t)h[0] << 56 | (uintmax_t)h[1] << 48 | (uintmax_t)h[2] << 40 | (uintmax_t)h[3] << 32 | (uintmax_t)h[4] << 24;
	while (h[4]) {
		if (hw == nw)
			return (void*)h;
		hw <<= 8;
		hw |= (uintmax_t)h[5] << 24;
		h++;
	}
	return NULL;
}

static char *strstr6(const uint8_t *h, const uint8_t *n)
{
	uint64_t nw = (uintmax_t)n[0] << 56 | (uintmax_t)n[1] << 48 | (uintmax_t)n[2] << 40 | (uintmax_t)n[3] << 32 | (uintmax_t)n[4] << 24 | (uintmax_t)n[5] << 16;
	uint64_t hw = (uintmax_t)h[0] << 56 | (uintmax_t)h[1] << 48 | (uintmax_t)h[2] << 40 | (uintmax_t)h[3] << 32 | (uintmax_t)h[4] << 24 | (uintmax_t)h[5] << 16;
	while (h[5]) {
		if (hw == nw)
			return (void*)h;
		hw <<= 8;
		hw |= (uintmax_t)h[6] << 16;
		h++;
	}
	return NULL;
}

static char *strstr7(const uint8_t *h, const uint8_t *n)
{
	uint64_t nw = (uintmax_t)n[0] << 56 | (uintmax_t)n[1] << 48 | (uintmax_t)n[2] << 40 | (uintmax_t)n[3] << 32 | (uintmax_t)n[4] << 24 | (uintmax_t)n[5] << 16 | (uintmax_t)n[6] << 8;
	uint64_t hw = (uintmax_t)h[0] << 56 | (uintmax_t)h[1] << 48 | (uintmax_t)h[2] << 40 | (uintmax_t)h[3] << 32 | (uintmax_t)h[4] << 24 | (uintmax_t)h[5] << 16 | (uintmax_t)h[6] << 8;
	while (h[6]) {
		if (hw == nw)
			return (void*)h;
		hw <<= 8;
		hw |= (uintmax_t)h[7] << 8;
		h++;
	}
	return NULL;
}

static char *strstr8(const uint8_t *h, const uint8_t *n)
{
	uint64_t nw = (uintmax_t)n[0] << 56 | (uintmax_t)n[1] << 48 | (uintmax_t)n[2] << 40 | (uintmax_t)n[3] << 32 | (uintmax_t)n[4] << 24 | (uintmax_t)n[5] << 16 | (uintmax_t)n[6] << 8 | n[7];
	uint64_t hw = (uintmax_t)h[0] << 56 | (uintmax_t)h[1] << 48 | (uintmax_t)h[2] << 40 | (uintmax_t)h[3] << 32 | (uintmax_t)h[4] << 24 | (uintmax_t)h[5] << 16 | (uintmax_t)h[6] << 8 | h[7];
	while (h[7]) {
		if (hw == nw)
			return (void*)h;
		hw <<= 8;
		hw |= h[8];
		h++;
	}
	return NULL;
}

#define DEFINE_STRSTR(L) \
__attribute__((__target__("sse2"))) \
static char *strstr##L(const uint8_t *h, const uint8_t *n) \
{ \
	__m128i nw = _mm_or_si128(_mm_bslli_si128(_mm_loadu_si64(n+(L)-8), (L)-8), _mm_loadu_si64((void*)n)); \
	__m128i hw = _mm_or_si128(_mm_bslli_si128(_mm_loadu_si64(h+(L)-8), (L)-8), _mm_loadu_si64((void*)h)); \
	while (h[(L)-1]) { \
		if (_mm_movemask_epi8(_mm_cmpeq_epi8(hw, nw)) == 0xffff) \
			return (void*)h; \
		hw = _mm_or_si128(_mm_bsrli_si128(hw, 1), _mm_bslli_si128(_mm_loadu_si64(h+(L)-7), (L)-8)); \
		h++; \
	} \
	return NULL; \
}

DEFINE_STRSTR(9)
DEFINE_STRSTR(10)
DEFINE_STRSTR(11)
DEFINE_STRSTR(12)
DEFINE_STRSTR(13)
DEFINE_STRSTR(14)
DEFINE_STRSTR(15)

#undef DEFINE_STRSTR

__attribute__((__target__("sse2")))
static char *strstr16(const uint8_t *h, const uint8_t *n)
{
	__m128i nw = _mm_loadu_si128((void*)n);
	__m128i hw = _mm_loadu_si128((void*)h);
	while (h[15]) {
		if (_mm_movemask_epi8(_mm_cmpeq_epi8(hw, nw)) == 0xffff)
			return (void*)h;
		hw = _mm_or_si128(_mm_bsrli_si128(hw, 1), _mm_bslli_si128(_mm_loadu_si64(h+9), 8));
		h++;
	}
	return NULL;
}

/*
 * Copyright (c) 2005-2014 Rich Felker, et al.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define BITOP(a, b, op) \
	((a)[(size_t)(b) / (8 * sizeof *(a))] op \
	    (size_t)1 << ((size_t)(b) % (8 * sizeof *(a))))

static char *twoway_strstr(const unsigned char *h, const unsigned char *n)
{
	const unsigned char *z;
	size_t l, ip, jp, k, p, ms, p0, mem, mem0;
	size_t byteset[32 / sizeof(size_t)] = { 0 };
	size_t shift[256];

	/* Computing length of needle and fill shift table */
	for (l = 0; n[l] && h[l]; l++)
		BITOP(byteset, n[l], |=), shift[n[l]] = l + 1;
	if (n[l])
		return 0; /* hit the end of h */

	/* Compute maximal suffix */
	ip = -1;
	jp = 0;
	k = p = 1;
	while (jp + k < l) {
		if (n[ip + k] == n[jp + k]) {
			if (k == p) {
				jp += p;
				k = 1;
			} else
				k++;
		} else if (n[ip + k] > n[jp + k]) {
			jp += k;
			k = 1;
			p = jp - ip;
		} else {
			ip = jp++;
			k = p = 1;
		}
	}
	ms = ip;
	p0 = p;

	/* And with the opposite comparison */
	ip = -1;
	jp = 0;
	k = p = 1;
	while (jp + k < l) {
		if (n[ip + k] == n[jp + k]) {
			if (k == p) {
				jp += p;
				k = 1;
			} else
				k++;
		} else if (n[ip + k] < n[jp + k]) {
			jp += k;
			k = 1;
			p = jp - ip;
		} else {
			ip = jp++;
			k = p = 1;
		}
	}
	if (ip + 1 > ms + 1)
		ms = ip;
	else
		p = p0;

	/* Periodic needle? */
	if (memcmp(n, n + p, ms + 1)) {
		mem0 = 0;
		p = MAX(ms, l - ms - 1) + 1;
	} else
		mem0 = l - p;
	mem = 0;

	/* Initialize incremental end-of-haystack pointer */
	z = h;

	/* Search loop */
	for (;;) {
		/* Update incremental end-of-haystack pointer */
		if ((size_t)(z - h) < l) {
			/* Fast estimate for MIN(l,63) */
			size_t grow = l | 63;
			const unsigned char *z2 = memchr(z, 0, grow);
			if (z2) {
				z = z2;
				if ((size_t)(z - h) < l)
					return 0;
			} else
				z += grow;
		}

		/* Check last byte first; advance by shift on mismatch */
		if (BITOP(byteset, h[l - 1], &)) {
			k = l - shift[h[l - 1]];
			if (k) {
				if (k < mem)
					k = mem;
				h += k;
				mem = 0;
				continue;
			}
		} else {
			h += l;
			mem = 0;
			continue;
		}

		/* Compare right half */
		for (k = MAX(ms + 1, mem); n[k] && n[k] == h[k]; k++)
			;
		if (n[k]) {
			h += k - ms;
			mem = 0;
			continue;
		}
		/* Compare left half */
		for (k = ms + 1; k > mem && n[k - 1] == h[k - 1]; k--)
			;
		if (k <= mem)
			return (char *)h;
		h += p;
		mem = mem0;
	}
}

char *strstr(const char *h, const char *n)
{
	if (!n[0])
		return (char*)h;
	h = strchr(h, n[0]);
	if (!h || !n[1])
		return (char*)h;
	if (!h[1])
		return NULL;
	if (!n[2])
		return strstr2((void*)h, (void*)n);
	if (!h[2])
		return NULL;
	if (!n[3])
		return strstr3((void*)h, (void*)n);
	if (!h[3])
		return NULL;
	if (!n[4])
		return strstr4((void*)h, (void*)n);
	if (!h[4])
		return NULL;
	if (!n[5])
		return strstr5((void*)h, (void*)n);
	if (!h[5])
		return NULL;
	if (!n[6])
		return strstr6((void*)h, (void*)n);
	if (!h[6])
		return NULL;
	if (!n[7])
		return strstr7((void*)h, (void*)n);
	if (!h[7])
		return NULL;
	if (!n[8])
		return strstr8((void*)h, (void*)n);
	if (!h[8])
		return NULL;
	if (has_sse2()) {
		if (!n[9])
			return strstr9((void*)h, (void*)n);
		if (!h[9])
			return NULL;
		if (!n[10])
			return strstr10((void*)h, (void*)n);
		if (!h[10])
			return NULL;
		if (!n[11])
			return strstr11((void*)h, (void*)n);
		if (!h[11])
			return NULL;
		if (!n[12])
			return strstr12((void*)h, (void*)n);
		if (!h[12])
			return NULL;
		if (!n[13])
			return strstr13((void*)h, (void*)n);
		if (!h[13])
			return NULL;
		if (!n[14])
			return strstr14((void*)h, (void*)n);
		if (!h[14])
			return NULL;
		if (!n[15])
			return strstr15((void*)h, (void*)n);
		if (!h[15])
			return NULL;
		if (!n[16])
			return strstr16((void*)h, (void*)n);
		if (!h[16])
			return NULL;
	}
	size_t l = strnlen(n, 512);
	if (l < 512) {
		do
			if (!strncmp(h, n, l))
				return (char*)h;
		while (*++h);
		return NULL;
	}
	return twoway_strstr((void*)h, (void*)n);
}
