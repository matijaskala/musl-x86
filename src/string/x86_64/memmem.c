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

#include <string.h>
#include <stdint.h>
#include <immintrin.h>

#include "cpu_features.h"

static char *memmem2(const uint8_t *h, size_t k, const uint8_t *n)
{
	uint16_t nw = (uintmax_t)n[0] << 8 | n[1];
	uint16_t hw = (uintmax_t)h[0] << 8 | h[1];
	while (k >= 2) {
		if (hw == nw)
			return (void*)h;
		hw <<= 8;
		hw |= h[2];
		h++;
		k--;
	}
	return NULL;
}

static char *memmem3(const uint8_t *h, size_t k, const uint8_t *n)
{
	uint32_t nw = (uintmax_t)n[0] << 24 | (uintmax_t)n[1] << 16 | (uintmax_t)n[2] << 8;
	uint32_t hw = (uintmax_t)h[0] << 24 | (uintmax_t)h[1] << 16 | (uintmax_t)h[2] << 8;
	while (k >= 3) {
		if (hw == nw)
			return (void*)h;
		hw <<= 8;
		hw |= (uintmax_t)h[3] << 8;
		h++;
		k--;
	}
	return NULL;
}

static char *memmem4(const uint8_t *h, size_t k, const uint8_t *n)
{
	uint32_t nw = (uintmax_t)n[0] << 24 | (uintmax_t)n[1] << 16 | (uintmax_t)n[2] << 8 | n[3];
	uint32_t hw = (uintmax_t)h[0] << 24 | (uintmax_t)h[1] << 16 | (uintmax_t)h[2] << 8 | h[3];
	while (k >= 4) {
		if (hw == nw)
			return (void*)h;
		hw <<= 8;
		hw |= h[4];
		h++;
		k--;
	}
	return NULL;
}

static char *memmem5(const uint8_t *h, size_t k, const uint8_t *n)
{
	uint64_t nw = (uintmax_t)n[0] << 56 | (uintmax_t)n[1] << 48 | (uintmax_t)n[2] << 40 | (uintmax_t)n[3] << 32 | (uintmax_t)n[4] << 24;
	uint64_t hw = (uintmax_t)h[0] << 56 | (uintmax_t)h[1] << 48 | (uintmax_t)h[2] << 40 | (uintmax_t)h[3] << 32 | (uintmax_t)h[4] << 24;
	while (k >= 5) {
		if (hw == nw)
			return (void*)h;
		hw <<= 8;
		hw |= (uintmax_t)h[5] << 24;
		h++;
		k--;
	}
	return NULL;
}

static char *memmem6(const uint8_t *h, size_t k, const uint8_t *n)
{
	uint64_t nw = (uintmax_t)n[0] << 56 | (uintmax_t)n[1] << 48 | (uintmax_t)n[2] << 40 | (uintmax_t)n[3] << 32 | (uintmax_t)n[4] << 24 | (uintmax_t)n[5] << 16;
	uint64_t hw = (uintmax_t)h[0] << 56 | (uintmax_t)h[1] << 48 | (uintmax_t)h[2] << 40 | (uintmax_t)h[3] << 32 | (uintmax_t)h[4] << 24 | (uintmax_t)h[5] << 16;
	while (k >= 6) {
		if (hw == nw)
			return (void*)h;
		hw <<= 8;
		hw |= (uintmax_t)h[6] << 16;
		h++;
		k--;
	}
	return NULL;
}

static char *memmem7(const uint8_t *h, size_t k, const uint8_t *n)
{
	uint64_t nw = (uintmax_t)n[0] << 56 | (uintmax_t)n[1] << 48 | (uintmax_t)n[2] << 40 | (uintmax_t)n[3] << 32 | (uintmax_t)n[4] << 24 | (uintmax_t)n[5] << 16 | (uintmax_t)n[6] << 8;
	uint64_t hw = (uintmax_t)h[0] << 56 | (uintmax_t)h[1] << 48 | (uintmax_t)h[2] << 40 | (uintmax_t)h[3] << 32 | (uintmax_t)h[4] << 24 | (uintmax_t)h[5] << 16 | (uintmax_t)h[6] << 8;
	while (k >= 7) {
		if (hw == nw)
			return (void*)h;
		hw <<= 8;
		hw |= (uintmax_t)h[7] << 8;
		h++;
		k--;
	}
	return NULL;
}

static char *memmem8(const uint8_t *h, size_t k, const uint8_t *n)
{
	uint64_t nw = (uintmax_t)n[0] << 56 | (uintmax_t)n[1] << 48 | (uintmax_t)n[2] << 40 | (uintmax_t)n[3] << 32 | (uintmax_t)n[4] << 24 | (uintmax_t)n[5] << 16 | (uintmax_t)n[6] << 8 | n[7];
	uint64_t hw = (uintmax_t)h[0] << 56 | (uintmax_t)h[1] << 48 | (uintmax_t)h[2] << 40 | (uintmax_t)h[3] << 32 | (uintmax_t)h[4] << 24 | (uintmax_t)h[5] << 16 | (uintmax_t)h[6] << 8 | h[7];
	while (k >= 8) {
		if (hw == nw)
			return (void*)h;
		hw <<= 8;
		hw |= h[8];
		h++;
		k--;
	}
	return NULL;
}

#define DEFINE_MEMMEM(L) \
__attribute__((__target__("sse2"))) \
static char *memmem##L(const uint8_t *h, size_t k, const uint8_t *n) \
{ \
	__m128i nw = _mm_or_si128(_mm_bslli_si128(_mm_loadu_si64(n+(L)-8), (L)-8), _mm_loadu_si64((void*)n)); \
	__m128i hw = _mm_or_si128(_mm_bslli_si128(_mm_loadu_si64(h+(L)-8), (L)-8), _mm_loadu_si64((void*)h)); \
	while (k >= (L)) { \
		if (_mm_movemask_epi8(_mm_cmpeq_epi8(hw, nw)) == 0xffff) \
			return (void*)h; \
		hw = _mm_or_si128(_mm_bsrli_si128(hw, 1), _mm_bslli_si128(_mm_loadu_si64(h+(L)-7), (L)-8)); \
		h++; \
		k--; \
	} \
	return NULL; \
}

DEFINE_MEMMEM(9)
DEFINE_MEMMEM(10)
DEFINE_MEMMEM(11)
DEFINE_MEMMEM(12)
DEFINE_MEMMEM(13)
DEFINE_MEMMEM(14)
DEFINE_MEMMEM(15)

#undef DEFINE_MEMMEM

__attribute__((__target__("sse2")))
static char *memmem16(const uint8_t *h, size_t k, const uint8_t *n)
{
	__m128i nw = _mm_loadu_si128((void*)n);
	__m128i hw = _mm_loadu_si128((void*)h);
	while (k >= 16) {
		if (_mm_movemask_epi8(_mm_cmpeq_epi8(hw, nw)) == 0xffff)
			return (void*)h;
		hw = _mm_or_si128(_mm_bsrli_si128(hw, 1), _mm_bslli_si128(_mm_loadu_si64(h+9), 8));
		h++;
		k--;
	}
	return NULL;
}

#define DEFINE_MEMMEM(L) \
__attribute__((__target__("avx2"))) \
static char *memmem##L(const uint8_t *h, size_t k, const uint8_t *n) \
{ \
	__m256i nw = _mm256_bslli_epi128(_mm256_castsi128_si256(_mm_loadu_si128((void*)(n+(L)-16))), (L)-16); \
	__m256i hw = _mm256_bslli_epi128(_mm256_castsi128_si256(_mm_loadu_si128((void*)(h+(L)-16))), (L)-16); \
	nw = _mm256_insertf128_si256(nw, _mm_loadu_si128((void*)n), 0); \
	hw = _mm256_insertf128_si256(hw, _mm_loadu_si128((void*)h), 0); \
	while (k >= (L)) { \
		if ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(hw, nw)) & ((1ull << (L)) - 1)) == (1ull << (L)) - 1) \
			return (void*)h; \
		hw = _mm256_insert_epi8(_mm256_bsrli_epi128(hw, 1), h[L], (L)-1); \
		h++; \
		k--; \
	} \
	return NULL; \
}

DEFINE_MEMMEM(17)
DEFINE_MEMMEM(18)
DEFINE_MEMMEM(19)
DEFINE_MEMMEM(20)
DEFINE_MEMMEM(21)
DEFINE_MEMMEM(22)
DEFINE_MEMMEM(23)
DEFINE_MEMMEM(24)
DEFINE_MEMMEM(25)
DEFINE_MEMMEM(26)
DEFINE_MEMMEM(27)
DEFINE_MEMMEM(28)
DEFINE_MEMMEM(29)
DEFINE_MEMMEM(30)
DEFINE_MEMMEM(31)

#undef DEFINE_MEMMEM

__attribute__((__target__("avx2")))
static char *memmem32(const uint8_t *h, size_t k, const uint8_t *n)
{
	__m256i nw = _mm256_loadu_si256((void*)n);
	__m256i hw = _mm256_loadu_si256((void*)h);
	while (k >= 32) {
		if (_mm256_movemask_epi8(_mm256_cmpeq_epi8(hw, nw)) == -1)
			return (void*)h;
		hw = _mm256_insert_epi8(_mm256_bsrli_epi128(hw, 1), h[32], 31);
		h++;
		k--;
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

static char *twoway_memmem(const unsigned char *h, const unsigned char *z, const unsigned char *n, size_t l)
{
	size_t i, ip, jp, k, p, ms, p0, mem, mem0;
	size_t byteset[32 / sizeof(size_t)] = { 0 };
	size_t shift[256];

	/* Computing length of needle and fill shift table */
	for (i = 0; i < l; i++)
		BITOP(byteset, n[i], |=), shift[n[i]] = i + 1;

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

	/* Search loop */
	for (;;) {
		/* If remainder of haystack is shorter than needle, done */
		if ((size_t)(z - h) < l)
			return 0;

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
		for (k = MAX(ms + 1, mem); k < l && n[k] == h[k]; k++)
			;
		if (k < l) {
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

void *memmem(const void *h0, size_t k, const void *n0, size_t l)
{
	const uint8_t *h;
	const uint8_t *n = n0;

	if (!l)
		return (void*)h0;

	if (k < l)
		return NULL;
	h = memchr(h0, *n, k);
	if (!h || l == 1)
		return (void*)h;
	k -= h - (const unsigned char*)h0;
	if (k < l)
		return NULL;
	switch (l) {
		case 2:
			return memmem2(h, k, n);
		case 3:
			return memmem3(h, k, n);
		case 4:
			return memmem4(h, k, n);
		case 5:
			return memmem5(h, k, n);
		case 6:
			return memmem6(h, k, n);
		case 7:
			return memmem7(h, k, n);
		case 8:
			return memmem8(h, k, n);
	}
	if (has_sse2()) {
		switch (l) {
			case 9:
				return memmem9(h, k, n);
			case 10:
				return memmem10(h, k, n);
			case 11:
				return memmem11(h, k, n);
			case 12:
				return memmem12(h, k, n);
			case 13:
				return memmem13(h, k, n);
			case 14:
				return memmem14(h, k, n);
			case 15:
				return memmem15(h, k, n);
			case 16:
				return memmem16(h, k, n);
		}
		if (has_avx2()) {
			switch (l) {
				case 17:
					return memmem17(h, k, n);
				case 18:
					return memmem18(h, k, n);
				case 19:
					return memmem19(h, k, n);
				case 20:
					return memmem20(h, k, n);
				case 21:
					return memmem21(h, k, n);
				case 22:
					return memmem22(h, k, n);
				case 23:
					return memmem23(h, k, n);
				case 24:
					return memmem24(h, k, n);
				case 25:
					return memmem25(h, k, n);
				case 26:
					return memmem26(h, k, n);
				case 27:
					return memmem27(h, k, n);
				case 28:
					return memmem28(h, k, n);
				case 29:
					return memmem29(h, k, n);
				case 30:
					return memmem30(h, k, n);
				case 31:
					return memmem31(h, k, n);
				case 32:
					return memmem32(h, k, n);
			}
		}
	}
	if (l < 512) {
		for (size_t i = 0; i <= k - l; i++)
			if (!memcmp(h + i, n, l))
				return (void*)(h + i);
		return NULL;
	}
	return twoway_memmem(h, h+k, n, l);
}
