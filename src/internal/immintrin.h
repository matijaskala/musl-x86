/*===---- immintrin.h - Intel intrinsics -----------------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef _MSC_VER
typedef int __v4si __attribute__((__vector_size__(16)));
typedef long long __m128i __attribute__((__vector_size__(16), __aligned__(16)));
typedef long long __m128i_u __attribute__((__vector_size__(16), __aligned__(1)));
typedef long long __v2di __attribute__ ((__vector_size__ (16)));
typedef char __v16qi __attribute__((__vector_size__(16)));
typedef unsigned long long __v2du __attribute__ ((__vector_size__ (16)));
typedef long long __v4di __attribute__ ((__vector_size__ (32)));
typedef int __v8si __attribute__ ((__vector_size__ (32)));
typedef char __v32qi __attribute__ ((__vector_size__ (32)));
typedef unsigned long long __v4du __attribute__ ((__vector_size__ (32)));
typedef long long __m256i __attribute__((__vector_size__(32), __aligned__(32)));
typedef long long __m256i_u __attribute__((__vector_size__(32), __aligned__(1)));
#define _MM_HINT_NTA 0
/// Loads one cache line of data from the specified address to a location
///    closer to the processor.
///
/// \headerfile <x86intrin.h>
///
/// \code
/// void _mm_prefetch(const void * a, const int sel);
/// \endcode
///
/// This intrinsic corresponds to the <c> PREFETCHNTA </c> instruction.
///
/// \param a
///    A pointer to a memory location containing a cache line of data.
/// \param sel
///    A predefined integer constant specifying the type of prefetch
///    operation: \n
///    _MM_HINT_NTA: Move data using the non-temporal access (NTA) hint. The
///    PREFETCHNTA instruction will be generated. \n
///    _MM_HINT_T0: Move data using the T0 hint. The PREFETCHT0 instruction will
///    be generated. \n
///    _MM_HINT_T1: Move data using the T1 hint. The PREFETCHT1 instruction will
///    be generated. \n
///    _MM_HINT_T2: Move data using the T2 hint. The PREFETCHT2 instruction will
///    be generated.
#define _mm_prefetch(a, sel) (__builtin_prefetch((const void *)(a), \
                                                 ((sel) >> 2) & 1, (sel) & 0x3))
/// Forces strong memory ordering (serialization) between store
///    instructions preceding this instruction and store instructions following
///    this instruction, ensuring the system completes all previous stores
///    before executing subsequent stores.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> SFENCE </c> instruction.
///
static inline void __attribute__((__always_inline__, __target__("sse")))
_mm_sfence(void)
{
  __builtin_ia32_sfence();
}
#ifdef __clang__
#define __DEFAULT_FN_ATTRS __attribute__((__always_inline__, __nodebug__, __target__("sse2"), __min_vector_width__(128)))
#else
#define __DEFAULT_FN_ATTRS __attribute__((__always_inline__, __target__("sse2")))
#endif
/// Performs a bitwise AND of two 128-bit integer vectors.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VPAND / PAND </c> instruction.
///
/// \param __a
///    A 128-bit integer vector containing one of the source operands.
/// \param __b
///    A 128-bit integer vector containing one of the source operands.
/// \returns A 128-bit integer vector containing the bitwise AND of the values
///    in both operands.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_and_si128(__m128i __a, __m128i __b)
{
  return (__m128i)((__v2du)__a & (__v2du)__b);
}

/// Performs a bitwise AND of two 128-bit integer vectors, using the
///    one's complement of the values contained in the first source operand.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VPANDN / PANDN </c> instruction.
///
/// \param __a
///    A 128-bit vector containing the left source operand. The one's complement
///    of this value is used in the bitwise AND.
/// \param __b
///    A 128-bit vector containing the right source operand.
/// \returns A 128-bit integer vector containing the bitwise AND of the one's
///    complement of the first operand and the values in the second operand.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_andnot_si128(__m128i __a, __m128i __b)
{
  return (__m128i)(~(__v2du)__a & (__v2du)__b);
}
/// Loads a 64-bit integer value to the low element of a 128-bit integer
///    vector and clears the upper element.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VMOVQ / MOVQ </c> instruction.
///
/// \param __a
///    A pointer to a 64-bit memory location. The address of the memory
///    location does not have to be aligned.
/// \returns A 128-bit vector of [2 x i64] containing the loaded value.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_loadu_si64(void const *__a)
{
  struct __loadu_si64 {
    long long __v;
  } __attribute__((__packed__, __may_alias__));
  long long __u = ((const struct __loadu_si64*)__a)->__v;
  return __extension__ (__m128i)(__v2di){__u, 0LL};
}
/// Performs a bitwise OR of two 128-bit integer vectors.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VPOR / POR </c> instruction.
///
/// \param __a
///    A 128-bit integer vector containing one of the source operands.
/// \param __b
///    A 128-bit integer vector containing one of the source operands.
/// \returns A 128-bit integer vector containing the bitwise OR of the values
///    in both operands.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_or_si128(__m128i __a, __m128i __b)
{
  return (__m128i)((__v2du)__a | (__v2du)__b);
}
/// Compares each of the corresponding 8-bit values of the 128-bit
///    integer vectors for equality. Each comparison yields 0x0 for false, 0xFF
///    for true.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VPCMPEQB / PCMPEQB </c> instruction.
///
/// \param __a
///    A 128-bit integer vector.
/// \param __b
///    A 128-bit integer vector.
/// \returns A 128-bit integer vector containing the comparison results.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cmpeq_epi8(__m128i __a, __m128i __b)
{
  return (__m128i)((__v16qi)__a == (__v16qi)__b);
}
/// Compares each of the corresponding 32-bit values of the 128-bit
///    integer vectors for equality. Each comparison yields 0x0 for false,
///    0xFFFFFFFF for true.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VPCMPEQD / PCMPEQD </c> instruction.
///
/// \param __a
///    A 128-bit integer vector.
/// \param __b
///    A 128-bit integer vector.
/// \returns A 128-bit integer vector containing the comparison results.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_cmpeq_epi32(__m128i __a, __m128i __b)
{
  return (__m128i)((__v4si)__a == (__v4si)__b);
}
#ifdef __clang__
#define _mm_bslli_si128(a, imm) \
  (__m128i)__builtin_ia32_pslldqi128_byteshift((__v2di)(__m128i)(a), (int)(imm))
#define _mm_bsrli_si128(a, imm) \
  (__m128i)__builtin_ia32_psrldqi128_byteshift((__v2di)(__m128i)(a), (int)(imm))
#else
#define _mm_bslli_si128(a, imm) \
  (__m128i)__builtin_ia32_pslldqi128((__v2di)(__m128i)(a), (int)(imm) * 8)
#define _mm_bsrli_si128(a, imm) \
  (__m128i)__builtin_ia32_psrldqi128((__v2di)(__m128i)(a), (int)(imm) * 8)
#endif
/// Moves packed integer values from an aligned 128-bit memory location
///    to elements in a 128-bit integer vector.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VMOVDQA / MOVDQA </c> instruction.
///
/// \param __p
///    An aligned pointer to a memory location containing integer values.
/// \returns A 128-bit integer vector containing the moved values.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_load_si128(__m128i const *__p)
{
  return *__p;
}
/// Moves packed integer values from an unaligned 128-bit memory location
///    to elements in a 128-bit integer vector.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VMOVDQU / MOVDQU </c> instruction.
///
/// \param __p
///    A pointer to a memory location containing integer values.
/// \returns A 128-bit integer vector containing the moved values.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_loadu_si128(__m128i_u const *__p)
{
  struct __loadu_si128 {
    __m128i_u __v;
  } __attribute__((__packed__, __may_alias__));
  return ((const struct __loadu_si128*)__p)->__v;
}
/// Initializes the 8-bit values in a 128-bit vector of [16 x i8] with
///    the specified 8-bit integer values.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic is a utility function and does not correspond to a specific
///    instruction.
///
/// \param __b15
///    Initializes bits [127:120] of the destination vector.
/// \param __b14
///    Initializes bits [119:112] of the destination vector.
/// \param __b13
///    Initializes bits [111:104] of the destination vector.
/// \param __b12
///    Initializes bits [103:96] of the destination vector.
/// \param __b11
///    Initializes bits [95:88] of the destination vector.
/// \param __b10
///    Initializes bits [87:80] of the destination vector.
/// \param __b9
///    Initializes bits [79:72] of the destination vector.
/// \param __b8
///    Initializes bits [71:64] of the destination vector.
/// \param __b7
///    Initializes bits [63:56] of the destination vector.
/// \param __b6
///    Initializes bits [55:48] of the destination vector.
/// \param __b5
///    Initializes bits [47:40] of the destination vector.
/// \param __b4
///    Initializes bits [39:32] of the destination vector.
/// \param __b3
///    Initializes bits [31:24] of the destination vector.
/// \param __b2
///    Initializes bits [23:16] of the destination vector.
/// \param __b1
///    Initializes bits [15:8] of the destination vector.
/// \param __b0
///    Initializes bits [7:0] of the destination vector.
/// \returns An initialized 128-bit vector of [16 x i8] containing the values
///    provided in the operands.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_set_epi8(char __b15, char __b14, char __b13, char __b12, char __b11, char __b10, char __b9, char __b8, char __b7, char __b6, char __b5, char __b4, char __b3, char __b2, char __b1, char __b0)
{
  return __extension__ (__m128i)(__v16qi){ __b0, __b1, __b2, __b3, __b4, __b5, __b6, __b7, __b8, __b9, __b10, __b11, __b12, __b13, __b14, __b15 };
}
/// Initializes the 32-bit values in a 128-bit vector of [4 x i32] with
///    the specified 32-bit integer values.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic is a utility function and does not correspond to a specific
///    instruction.
///
/// \param __i3
///    A 32-bit integer value used to initialize bits [127:96] of the
///    destination vector.
/// \param __i2
///    A 32-bit integer value used to initialize bits [95:64] of the destination
///    vector.
/// \param __i1
///    A 32-bit integer value used to initialize bits [63:32] of the destination
///    vector.
/// \param __i0
///    A 32-bit integer value used to initialize bits [31:0] of the destination
///    vector.
/// \returns An initialized 128-bit vector of [4 x i32] containing the values
///    provided in the operands.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_set_epi32(int __i3, int __i2, int __i1, int __i0)
{
  return __extension__ (__m128i)(__v4si){ __i0, __i1, __i2, __i3};
}
/// Initializes all values in a 128-bit vector of [16 x i8] with the
///    specified 8-bit value.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic is a utility function and does not correspond to a specific
///    instruction.
///
/// \param __b
///    An 8-bit value used to initialize the elements of the destination integer
///    vector.
/// \returns An initialized 128-bit vector of [16 x i8] with all elements
///    containing the value provided in the operand.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_set1_epi8(char __b)
{
  return _mm_set_epi8(__b, __b, __b, __b, __b, __b, __b, __b, __b, __b, __b, __b, __b, __b, __b, __b);
}
/// Initializes all values in a 128-bit vector of [4 x i32] with the
///    specified 32-bit value.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic is a utility function and does not correspond to a specific
///    instruction.
///
/// \param __i
///    A 32-bit value used to initialize the elements of the destination integer
///    vector.
/// \returns An initialized 128-bit vector of [4 x i32] with all elements
///    containing the value provided in the operand.
static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_set1_epi32(int __i)
{
  return _mm_set_epi32(__i, __i, __i, __i);
}
/// Copies the values of the most significant bits from each 8-bit
///    element in a 128-bit integer vector of [16 x i8] to create a 16-bit mask
///    value, zero-extends the value, and writes it to the destination.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VPMOVMSKB / PMOVMSKB </c> instruction.
///
/// \param __a
///    A 128-bit integer vector containing the values with bits to be extracted.
/// \returns The most significant bits from each 8-bit element in \a __a,
///    written to bits [15:0]. The other bits are assigned zeros.
static __inline__ int __DEFAULT_FN_ATTRS
_mm_movemask_epi8(__m128i __a)
{
  return __builtin_ia32_pmovmskb128((__v16qi)__a);
}
/// Stores a 128-bit integer vector to a memory location aligned on a
///    128-bit boundary.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VMOVAPS / MOVAPS </c> instruction.
///
/// \param __p
///    A pointer to an aligned memory location that will receive the integer
///    values.
/// \param __b
///    A 128-bit integer vector containing the values to be moved.
static __inline__ void __DEFAULT_FN_ATTRS
_mm_store_si128(__m128i *__p, __m128i __b)
{
  *__p = __b;
}
/// Stores a 128-bit integer vector to an unaligned memory location.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VMOVUPS / MOVUPS </c> instruction.
///
/// \param __p
///    A pointer to a memory location that will receive the integer values.
/// \param __b
///    A 128-bit integer vector containing the values to be moved.
static __inline__ void __DEFAULT_FN_ATTRS
_mm_storeu_si128(__m128i_u *__p, __m128i __b)
{
  struct __storeu_si128 {
    __m128i_u __v;
  } __attribute__((__packed__, __may_alias__));
  ((struct __storeu_si128*)__p)->__v = __b;
}
/// Stores a 128-bit integer vector to a 128-bit aligned memory location.
///
///    To minimize caching, the data is flagged as non-temporal (unlikely to be
///    used again soon).
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VMOVNTPS / MOVNTPS </c> instruction.
///
/// \param __p
///    A pointer to the 128-bit aligned memory location used to store the value.
/// \param __a
///    A 128-bit integer vector containing the values to be stored.
static __inline__ void __DEFAULT_FN_ATTRS
_mm_stream_si128(__m128i *__p, __m128i __a)
{
#ifdef __clang__
  __builtin_nontemporal_store((__v2di)__a, (__v2di*)__p);
#else
  __builtin_ia32_movntdq((__v2di*)__p, (__v2di)__a);
#endif
}
#undef __DEFAULT_FN_ATTRS
/// Constructs a 128-bit vector of [16 x i8] by first making a copy of
///    the 128-bit integer vector parameter, and then inserting the lower 8 bits
///    of an integer parameter \a I into an offset specified by the immediate
///    value parameter \a N.
///
/// \headerfile <x86intrin.h>
///
/// \code
/// __m128i _mm_insert_epi8(__m128i X, int I, const int N);
/// \endcode
///
/// This intrinsic corresponds to the <c> VPINSRB / PINSRB </c> instruction.
///
/// \param X
///    A 128-bit integer vector of [16 x i8]. This vector is copied to the
///    result and then one of the sixteen elements in the result vector is
///    replaced by the lower 8 bits of \a I.
/// \param I
///    An integer. The lower 8 bits of this operand are written to the result
///    beginning at the offset specified by \a N.
/// \param N
///    An immediate value. Bits [3:0] specify the bit offset in the result at
///    which the lower 8 bits of \a I are written. \n
///    0000: Bits [7:0] of the result are used for insertion. \n
///    0001: Bits [15:8] of the result are used for insertion. \n
///    0010: Bits [23:16] of the result are used for insertion. \n
///    0011: Bits [31:24] of the result are used for insertion. \n
///    0100: Bits [39:32] of the result are used for insertion. \n
///    0101: Bits [47:40] of the result are used for insertion. \n
///    0110: Bits [55:48] of the result are used for insertion. \n
///    0111: Bits [63:56] of the result are used for insertion. \n
///    1000: Bits [71:64] of the result are used for insertion. \n
///    1001: Bits [79:72] of the result are used for insertion. \n
///    1010: Bits [87:80] of the result are used for insertion. \n
///    1011: Bits [95:88] of the result are used for insertion. \n
///    1100: Bits [103:96] of the result are used for insertion. \n
///    1101: Bits [111:104] of the result are used for insertion. \n
///    1110: Bits [119:112] of the result are used for insertion. \n
///    1111: Bits [127:120] of the result are used for insertion.
/// \returns A 128-bit integer vector containing the constructed values.
#define _mm_insert_epi8(X, I, N) \
  (__m128i)__builtin_ia32_vec_set_v16qi((__v16qi)(__m128i)(X), \
                                        (int)(I), (int)(N))
#ifdef __clang__
#define __DEFAULT_FN_ATTRS __attribute__((__always_inline__, __nodebug__, __target__("avx"), __min_vector_width__(256)))
#else
#define __DEFAULT_FN_ATTRS __attribute__((__always_inline__, __target__("avx")))
#endif
/// Takes a [32 x i8] vector and replaces the vector element value
///    indexed by the immediate constant operand with a new value. Returns the
///    modified vector.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VINSERTF128+COMPOSITE </c>
///   instruction.
///
/// \param __a
///    A vector of [32 x i8] to be used by the insert operation.
/// \param __b
///    An i8 integer value. The replacement value for the insert operation.
/// \param __imm
///    An immediate integer specifying the index of the vector element to be
///    replaced.
/// \returns A copy of vector \a __a, after replacing its element indexed by
///    \a __imm with \a __b.
#ifdef __clang__
#define _mm256_insert_epi8(X, I, N) \
  (__m256i)__builtin_ia32_vec_set_v32qi((__v32qi)(__m256i)(X), \
                                        (int)(I), (int)(N))
#else
#define _mm256_insert_epi8(X, I, N) \
  _mm256_insertf128_si256((X), \
		          _mm_insert_epi8(_mm256_extractf128_si256((X), (N) >> 4), (I), (N) % 16), \
			  (N) >> 4)
#endif
/// Loads 256 bits of integer data from a 32-byte aligned memory
///    location pointed to by \a __p into elements of a 256-bit integer vector.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VMOVDQA </c> instruction.
///
/// \param __p
///    A 32-byte aligned pointer to a 256-bit integer vector containing integer
///    values.
/// \returns A 256-bit integer vector containing the moved values.
static __inline __m256i __DEFAULT_FN_ATTRS
_mm256_load_si256(__m256i const *__p)
{
  return *__p;
}
/// Loads 256 bits of integer data from an unaligned memory location
///    pointed to by \a __p into a 256-bit integer vector.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VMOVDQU </c> instruction.
///
/// \param __p
///    A pointer to a 256-bit integer vector containing integer values.
/// \returns A 256-bit integer vector containing the moved values.
static __inline __m256i __DEFAULT_FN_ATTRS
_mm256_loadu_si256(__m256i_u const *__p)
{
  struct __loadu_si256 {
    __m256i_u __v;
  } __attribute__((__packed__, __may_alias__));
  return ((const struct __loadu_si256*)__p)->__v;
}
/// Constructs a 256-bit integer vector initialized with the specified
///    8-bit integral values.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic is a utility function and does not correspond to a specific
///   instruction.
///
/// \param __b31
///    An 8-bit integral value used to initialize bits [255:248] of the result.
/// \param __b30
///    An 8-bit integral value used to initialize bits [247:240] of the result.
/// \param __b29
///    An 8-bit integral value used to initialize bits [239:232] of the result.
/// \param __b28
///    An 8-bit integral value used to initialize bits [231:224] of the result.
/// \param __b27
///    An 8-bit integral value used to initialize bits [223:216] of the result.
/// \param __b26
///    An 8-bit integral value used to initialize bits [215:208] of the result.
/// \param __b25
///    An 8-bit integral value used to initialize bits [207:200] of the result.
/// \param __b24
///    An 8-bit integral value used to initialize bits [199:192] of the result.
/// \param __b23
///    An 8-bit integral value used to initialize bits [191:184] of the result.
/// \param __b22
///    An 8-bit integral value used to initialize bits [183:176] of the result.
/// \param __b21
///    An 8-bit integral value used to initialize bits [175:168] of the result.
/// \param __b20
///    An 8-bit integral value used to initialize bits [167:160] of the result.
/// \param __b19
///    An 8-bit integral value used to initialize bits [159:152] of the result.
/// \param __b18
///    An 8-bit integral value used to initialize bits [151:144] of the result.
/// \param __b17
///    An 8-bit integral value used to initialize bits [143:136] of the result.
/// \param __b16
///    An 8-bit integral value used to initialize bits [135:128] of the result.
/// \param __b15
///    An 8-bit integral value used to initialize bits [127:120] of the result.
/// \param __b14
///    An 8-bit integral value used to initialize bits [119:112] of the result.
/// \param __b13
///    An 8-bit integral value used to initialize bits [111:104] of the result.
/// \param __b12
///    An 8-bit integral value used to initialize bits [103:96] of the result.
/// \param __b11
///    An 8-bit integral value used to initialize bits [95:88] of the result.
/// \param __b10
///    An 8-bit integral value used to initialize bits [87:80] of the result.
/// \param __b09
///    An 8-bit integral value used to initialize bits [79:72] of the result.
/// \param __b08
///    An 8-bit integral value used to initialize bits [71:64] of the result.
/// \param __b07
///    An 8-bit integral value used to initialize bits [63:56] of the result.
/// \param __b06
///    An 8-bit integral value used to initialize bits [55:48] of the result.
/// \param __b05
///    An 8-bit integral value used to initialize bits [47:40] of the result.
/// \param __b04
///    An 8-bit integral value used to initialize bits [39:32] of the result.
/// \param __b03
///    An 8-bit integral value used to initialize bits [31:24] of the result.
/// \param __b02
///    An 8-bit integral value used to initialize bits [23:16] of the result.
/// \param __b01
///    An 8-bit integral value used to initialize bits [15:8] of the result.
/// \param __b00
///    An 8-bit integral value used to initialize bits [7:0] of the result.
/// \returns An initialized 256-bit integer vector.
static __inline __m256i __DEFAULT_FN_ATTRS
_mm256_set_epi8(char __b31, char __b30, char __b29, char __b28,
                char __b27, char __b26, char __b25, char __b24,
                char __b23, char __b22, char __b21, char __b20,
                char __b19, char __b18, char __b17, char __b16,
                char __b15, char __b14, char __b13, char __b12,
                char __b11, char __b10, char __b09, char __b08,
                char __b07, char __b06, char __b05, char __b04,
                char __b03, char __b02, char __b01, char __b00)
{
  return __extension__ (__m256i)(__v32qi){
    __b00, __b01, __b02, __b03, __b04, __b05, __b06, __b07,
    __b08, __b09, __b10, __b11, __b12, __b13, __b14, __b15,
    __b16, __b17, __b18, __b19, __b20, __b21, __b22, __b23,
    __b24, __b25, __b26, __b27, __b28, __b29, __b30, __b31
  };
}
/// Constructs a 256-bit integer vector initialized with the specified
///    32-bit integral values.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic is a utility function and does not correspond to a specific
///   instruction.
///
/// \param __i0
///    A 32-bit integral value used to initialize bits [255:224] of the result.
/// \param __i1
///    A 32-bit integral value used to initialize bits [223:192] of the result.
/// \param __i2
///    A 32-bit integral value used to initialize bits [191:160] of the result.
/// \param __i3
///    A 32-bit integral value used to initialize bits [159:128] of the result.
/// \param __i4
///    A 32-bit integral value used to initialize bits [127:96] of the result.
/// \param __i5
///    A 32-bit integral value used to initialize bits [95:64] of the result.
/// \param __i6
///    A 32-bit integral value used to initialize bits [63:32] of the result.
/// \param __i7
///    A 32-bit integral value used to initialize bits [31:0] of the result.
/// \returns An initialized 256-bit integer vector.
static __inline __m256i __DEFAULT_FN_ATTRS
_mm256_set_epi32(int __i0, int __i1, int __i2, int __i3,
                 int __i4, int __i5, int __i6, int __i7)
{
  return __extension__ (__m256i)(__v8si){ __i7, __i6, __i5, __i4, __i3, __i2, __i1, __i0 };
}
/// Constructs a 256-bit integer vector of [32 x i8], with each of the
///    8-bit integral vector elements set to the specified 8-bit integral value.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VPSHUFB+VINSERTF128 </c> instruction.
///
/// \param __b
///    An 8-bit integral value used to initialize each vector element of the
///    result.
/// \returns An initialized 256-bit integer vector of [32 x i8].
static __inline __m256i __DEFAULT_FN_ATTRS
_mm256_set1_epi8(char __b)
{
  return _mm256_set_epi8(__b, __b, __b, __b, __b, __b, __b, __b,
                         __b, __b, __b, __b, __b, __b, __b, __b,
                         __b, __b, __b, __b, __b, __b, __b, __b,
                         __b, __b, __b, __b, __b, __b, __b, __b);
}
/// Constructs a 256-bit integer vector of [8 x i32], with each of the
///    32-bit integral vector elements set to the specified 32-bit integral
///    value.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VPERMILPS+VINSERTF128 </c>
///   instruction.
///
/// \param __i
///    A 32-bit integral value used to initialize each vector element of the
///    result.
/// \returns An initialized 256-bit integer vector of [8 x i32].
static __inline __m256i __DEFAULT_FN_ATTRS
_mm256_set1_epi32(int __i)
{
  return _mm256_set_epi32(__i, __i, __i, __i, __i, __i, __i, __i);
}
/// Stores integer values from a 256-bit integer vector to a 32-byte
///    aligned memory location pointed to by \a __p.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VMOVDQA </c> instruction.
///
/// \param __p
///    A 32-byte aligned pointer to a memory location that will receive the
///    integer values.
/// \param __a
///    A 256-bit integer vector containing the values to be moved.
static __inline void __DEFAULT_FN_ATTRS
_mm256_store_si256(__m256i *__p, __m256i __a)
{
  *__p = __a;
}

/// Stores integer values from a 256-bit integer vector to an unaligned
///    memory location pointed to by \a __p.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VMOVDQU </c> instruction.
///
/// \param __p
///    A pointer to a memory location that will receive the integer values.
/// \param __a
///    A 256-bit integer vector containing the values to be moved.
static __inline void __DEFAULT_FN_ATTRS
_mm256_storeu_si256(__m256i_u *__p, __m256i __a)
{
  struct __storeu_si256 {
    __m256i_u __v;
  } __attribute__((__packed__, __may_alias__));
  ((struct __storeu_si256*)__p)->__v = __a;
}
/// Moves integer data from a 256-bit integer vector to a 32-byte
///    aligned memory location. To minimize caching, the data is flagged as
///    non-temporal (unlikely to be used again soon).
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VMOVNTDQ </c> instruction.
///
/// \param __a
///    A pointer to a 32-byte aligned memory location that will receive the
///    integer values.
/// \param __b
///    A 256-bit integer vector containing the values to be moved.
static __inline void __DEFAULT_FN_ATTRS
_mm256_stream_si256(__m256i *__a, __m256i __b)
{
#ifdef __clang__
  typedef __v4di __v4di_aligned __attribute__((aligned(32)));
  __builtin_nontemporal_store((__v4di_aligned)__b, (__v4di_aligned*)__a);
#else
  __builtin_ia32_movntdq256 ((__v4di *)__a, (__v4di)__b);
#endif
}
/// Constructs a 256-bit integer vector from a 128-bit integer vector.
///
///    The lower 128 bits contain the value of the source vector. The contents
///    of the upper 128 bits are undefined.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic has no corresponding instruction.
///
/// \param __a
///    A 128-bit integer vector.
/// \returns A 256-bit integer vector. The lower 128 bits contain the value of
///    the parameter. The contents of the upper 128 bits are undefined.
static __inline __m256i __DEFAULT_FN_ATTRS
_mm256_castsi128_si256(__m128i __a)
{
#ifdef __clang__
  return __builtin_shufflevector((__v2di)__a, (__v2di)__a, 0, 1, -1, -1);
#else
  return (__m256i) __builtin_ia32_si256_si ((__v4si)__a);
#endif
}
/// Constructs a new 256-bit integer vector by first duplicating a
///    256-bit integer vector given in the first parameter, and then replacing
///    either the upper or the lower 128 bits with the contents of a 128-bit
///    integer vector in the second parameter.
///
///    The immediate integer parameter determines between the upper or the lower
///    128 bits.
///
/// \headerfile <x86intrin.h>
///
/// \code
/// __m256i _mm256_insertf128_si256(__m256i V1, __m128i V2, const int M);
/// \endcode
///
/// This intrinsic corresponds to the <c> VINSERTF128 </c> instruction.
///
/// \param V1
///    A 256-bit integer vector. This vector is copied to the result first, and
///    then either the upper or the lower 128 bits of the result will be
///    replaced by the contents of \a V2.
/// \param V2
///    A 128-bit integer vector. The contents of this parameter are written to
///    either the upper or the lower 128 bits of the result depending on the
///     value of parameter \a M.
/// \param M
///    An immediate integer. The least significant bit determines how the values
///    from the two parameters are interleaved: \n
///    If bit [0] of \a M is 0, \a V2 are copied to bits [127:0] of the result,
///    and bits [255:128] of \a V1 are copied to bits [255:128] of the
///    result. \n
///    If bit [0] of \a M is 1, \a V2 are copied to bits [255:128] of the
///    result, and bits [127:0] of \a V1 are copied to bits [127:0] of the
///    result.
/// \returns A 256-bit integer vector containing the interleaved values.
#define _mm256_insertf128_si256(V1, V2, M) \
  (__m256i)__builtin_ia32_vinsertf128_si256((__v8si)(__m256i)(V1), \
                                            (__v4si)(__m128i)(V2), (int)(M))
/// Constructs a new 256-bit integer vector by first duplicating a
///    256-bit integer vector given in the first parameter, and then replacing
///    either the upper or the lower 128 bits with the contents of a 128-bit
///    integer vector in the second parameter.
///
///    The immediate integer parameter determines between the upper or the lower
///    128 bits.
///
/// \headerfile <x86intrin.h>
///
/// \code
/// __m256i _mm256_insertf128_si256(__m256i V1, __m128i V2, const int M);
/// \endcode
///
/// This intrinsic corresponds to the <c> VINSERTF128 </c> instruction.
///
/// \param V1
///    A 256-bit integer vector. This vector is copied to the result first, and
///    then either the upper or the lower 128 bits of the result will be
///    replaced by the contents of \a V2.
/// \param V2
///    A 128-bit integer vector. The contents of this parameter are written to
///    either the upper or the lower 128 bits of the result depending on the
///     value of parameter \a M.
/// \param M
///    An immediate integer. The least significant bit determines how the values
///    from the two parameters are interleaved: \n
///    If bit [0] of \a M is 0, \a V2 are copied to bits [127:0] of the result,
///    and bits [255:128] of \a V1 are copied to bits [255:128] of the
///    result. \n
///    If bit [0] of \a M is 1, \a V2 are copied to bits [255:128] of the
///    result, and bits [127:0] of \a V1 are copied to bits [127:0] of the
///    result.
/// \returns A 256-bit integer vector containing the interleaved values.
#define _mm256_insertf128_si256(V1, V2, M) \
  (__m256i)__builtin_ia32_vinsertf128_si256((__v8si)(__m256i)(V1), \
                                            (__v4si)(__m128i)(V2), (int)(M))
/// Extracts either the upper or the lower 128 bits from a 256-bit
///    integer vector, as determined by the immediate integer parameter, and
///    returns the extracted bits as a 128-bit integer vector.
///
/// \headerfile <x86intrin.h>
///
/// \code
/// __m128i _mm256_extractf128_si256(__m256i V, const int M);
/// \endcode
///
/// This intrinsic corresponds to the <c> VEXTRACTF128 </c> instruction.
///
/// \param V
///    A 256-bit integer vector.
/// \param M
///    An immediate integer. The least significant bit determines which bits are
///    extracted from the first parameter:  \n
///    If bit [0] of \a M is 0, bits [127:0] of \a V are copied to the
///    result. \n
///    If bit [0] of \a M is 1, bits [255:128] of \a V are copied to the result.
/// \returns A 128-bit integer vector containing the extracted bits.
#define _mm256_extractf128_si256(V, M) \
  (__m128i)__builtin_ia32_vextractf128_si256((__v8si)(__m256i)(V), (int)(M))
#undef __DEFAULT_FN_ATTRS
#ifdef __clang__
#define __DEFAULT_FN_ATTRS256 __attribute__((__always_inline__, __nodebug__, __target__("avx2"), __min_vector_width__(256)))
#else
#define __DEFAULT_FN_ATTRS256 __attribute__((__always_inline__, __target__("avx2")))
#endif
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_and_si256(__m256i __a, __m256i __b)
{
  return (__m256i)((__v4du)__a & (__v4du)__b);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_andnot_si256(__m256i __a, __m256i __b)
{
  return (__m256i)(~(__v4du)__a & (__v4du)__b);
}
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_cmpeq_epi8(__m256i __a, __m256i __b)
{
  return (__m256i)((__v32qi)__a == (__v32qi)__b);
}
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_cmpeq_epi32(__m256i __a, __m256i __b)
{
  return (__m256i)((__v8si)__a == (__v8si)__b);
}
static __inline__ int __DEFAULT_FN_ATTRS256
_mm256_movemask_epi8(__m256i __a)
{
  return __builtin_ia32_pmovmskb256((__v32qi)__a);
}
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_or_si256(__m256i __a, __m256i __b)
{
  return (__m256i)((__v4du)__a | (__v4du)__b);
}
#ifdef __clang__
#define _mm256_bslli_epi128(a, imm) \
  (__m256i)__builtin_ia32_pslldqi256_byteshift((__v4di)(__m256i)(a), (int)(imm))
#define _mm256_bsrli_epi128(a, imm) \
  (__m256i)__builtin_ia32_psrldqi256_byteshift((__m256i)(a), (int)(imm))
#else
#define _mm256_bslli_epi128(a, imm) \
  (__m256i)__builtin_ia32_pslldqi256((__v4di)(__m256i)(a), (int)(imm) * 8)
#define _mm256_bsrli_epi128(a, imm) \
  (__m256i)__builtin_ia32_psrldqi256((__m256i)(a), (int)(imm) * 8)
#endif
#undef __DEFAULT_FN_ATTRS256
#define _xgetbv(A) __builtin_ia32_xgetbv((long long)(A))
#define _xsetbv(A, B) __builtin_ia32_xsetbv((unsigned int)(A), (unsigned long long)(B))
#else
#ifdef __cplusplus
extern "C" {
#endif
unsigned __int64 __cdecl _xgetbv(unsigned int);
void __cdecl _xsetbv(unsigned int, unsigned __int64);
#ifdef __cplusplus
}
#endif
#endif /* _MSC_VER */
