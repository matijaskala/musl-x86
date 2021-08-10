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

#include <stdint.h>
#if __x86_64__ || __i386__
#include <cpuid.h>
#include <immintrin.h>
#endif

#include "cpu_features.h"

__attribute__((visibility("hidden")))
struct cpu_features cpu_features = {0};

#if __x86_64__ || __i386__
__attribute__((__target__("xsave")))
static inline uint64_t my_xgetbv(unsigned int A) { return _xgetbv(A); }
#endif

__attribute__((visibility("hidden")))
void __init_cpu_features(void) {
#if __x86_64__ || __i386__
	unsigned int eax, ebx, ecx, edx;
	if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx))
		return;
	if (edx & bit_SSE2) {
		cpu_features.sse2 = 1;
	}
	if (ecx & bit_FMA) {
		cpu_features.fma = 1;
	}
	if (ecx & bit_OSXSAVE) {
		cpu_features.osxsave = 1;
		uint64_t xcr = my_xgetbv(0);
		if ((xcr & 6) == 6) {
			if (ecx & bit_AVX)
				cpu_features.avx = 1;
			if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx))
				return;
			if (cpu_features.avx) {
				if (ebx & bit_AVX2)
					cpu_features.avx2 = 1;
			}
			if ((xcr & 224) == 224) {
				if (ebx & bit_AVX512F) {
					cpu_features.avx512f = 1;
					if (ebx & bit_AVX512BW)
						cpu_features.avx512bw = 1;
					if (ebx & bit_AVX512CD)
						cpu_features.avx512cd = 1;
					if (ebx & bit_AVX512DQ)
						cpu_features.avx512dq = 1;
					if (ebx & bit_AVX512ER)
						cpu_features.avx512er = 1;
					if (ebx & bit_AVX512VL)
						cpu_features.avx512vl = 1;
				}
			}
		}
	}
#endif
}
