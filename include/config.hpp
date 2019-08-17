// Copyright (c) 2018, Tom Westerhout
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#if defined(__clang__)
#    define TCM_CLANG                                                          \
        (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#    define TCM_ASSUME(cond) __builtin_assume(cond)
#elif defined(__GNUC__)
#    define TCM_GCC                                                            \
        (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#    define TCM_ASSUME(cond)                                                   \
        do {                                                                   \
            if (!(cond)) __builtin_unreachable();                              \
        } while (false)
#elif defined(_MSV_VER)
#    define TCM_MSVC _MSV_VER
#    define TCM_ASSUME(cond)                                                   \
        do {                                                                   \
        } while (false)
#else
// clang-format off
#error "Unsupported compiler. Please, submit a request to https://github.com/twesterhout/percolation/issues."
// clang-format on
#endif

#if defined(WIN32) || defined(_WIN32)
#    define DA_EXPORT __declspec(dllexport)
#    define DA_NOINLINE __declspec(noinline)
#    define DA_FORCEINLINE __forceinline inline
#else
#    define DA_EXPORT __attribute__((visibility("default")))
#    define DA_NOINLINE __attribute__((noinline))
#    define DA_FORCEINLINE __attribute__((always_inline)) inline
#endif

#if defined(NDEBUG)
#    define TCM_CONSTEXPR constexpr
#else
#    define TCM_CONSTEXPR
#endif

#define TCM_NOEXCEPT noexcept

#define DA_NAMESPACE dual_annealing
#define DA_NAMESPACE_BEGIN namespace dual_annealing {
#define DA_NAMESPACE_END } // namespace dual_annealing

#include <cstdio>
#define DUAL_ANNEALING_TRACE(fmt, ...)                                         \
    do {                                                                       \
        ::std::fprintf(                                                        \
            stderr, "\x1b[1m\x1b[97m%s:%i:\x1b[0m \x1b[90mtrace:\x1b[0m " fmt, \
            __FILE__, __LINE__, __VA_ARGS__);                                  \
    } while (false)

#if !defined(NDEBUG)
#    if defined(__cplusplus)
#        include <cassert>
#    else
#        include <assert.h>
#    endif
#    define TCM_ASSERT(cond, msg) assert((cond) && (msg)) /* NOLINT */
#else
#    define TCM_ASSERT(cond, msg)
#endif
