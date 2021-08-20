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

#ifndef CPU_FEATURES_H
#define CPU_FEATURES_H

struct cpu_features {
	int sse2;
	int osxsave;
	int avx;
	int avx2;
	int avx512f;
	int avx512bw;
	int avx512cd;
	int avx512dq;
	int avx512er;
	int avx512vl;
	int fma;
};

__attribute__((visibility("hidden")))
extern struct cpu_features cpu_features;

static inline int has_sse2() {
#ifdef __SSE2__
	return 1;
#else
	return cpu_features.sse2;
#endif
}

static inline int has_avx() {
#ifdef __AVX__
	return 1;
#else
	return cpu_features.avx;
#endif
}

static inline int has_avx2() {
#ifdef __AVX2__
	return 1;
#else
	return cpu_features.avx2;
#endif
}

static inline int has_avx512f() {
#ifdef __AVX512F__
	return 1;
#else
	return cpu_features.avx512f;
#endif
}

static inline int has_avx512bw() {
#ifdef __AVX512BW__
	return 1;
#else
	return cpu_features.avx512bw;
#endif
}

static inline int has_fma() {
#ifdef __FMA__
	return 1;
#else
	return cpu_features.fma;
#endif
}

#endif // CPU_FEATURES_H
