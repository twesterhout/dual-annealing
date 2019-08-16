// Copyright (c) 2019, Tom Westerhout
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

/// \file assert.hpp
///

// Project configuration
// ========================================================================= {{{
#define TCM_ASSERT_NAMESPACE dual_annealing::detail
#define TCM_ASSERT_NAMESPACE_BEGIN                                             \
    namespace dual_annealing {                                                 \
    namespace detail {
#define TCM_ASSERT_NAMESPACE_END                                               \
    } /* namespace detail */                                                   \
    } /* namespace dual_annealing */
// clang-format off
#define TCM_ASSERT_MESSAGE_HEADER                                              \
    "╔═════════════════════════════════════════════════════════════════╗\n"    \
    "║     Congratulations, you have found a bug in dual-annealing!    ║\n"    \
    "║              Please, be so kind to submit it here               ║\n"    \
    "║      https://github.com/twesterhout/dual-annealing/issues       ║\n"    \
    "╚═════════════════════════════════════════════════════════════════╝\n"
#if defined(DUAL_ANNEALING_DEBUG)
#define DUAL_ANNEALING_ASSERT(cond, ...) TCM_ASSERT_ASSERT(cond, __VA_ARGS__)
#else
#define DUAL_ANNEALING_ASSERT(cond, ...) static_cast<void>(0)
#endif
// clang-format on
// ========================================================================= }}}

/// \macro TCM_ASSERT_NAMESPACE
///
/// Defines in which namespace assert_fail functions should be located. You
/// should define it to something like `your_project_namespace` or
/// `your_project_namespace::detail` __before__ including this header
/// (assert.hpp). If you use a nested namespace, then you also need to define
/// #TCM_ASSERT_NAMESPACE_BEGIN and #TCM_ASSERT_NAMESPACE_END macros.
#if !defined(TCM_ASSERT_NAMESPACE)
// clang-format off
#    error "You need to define TCM_ASSERT_NAMESPACE before including assert.hpp"
// clang-format on
#endif
/// \macro TCM_ASSERT_NAMESPACE_BEGIN
///
/// Should expand to a statement which enters #TCM_ASSERT_NAMESPACE namespace.
/// By default, it expands to `namespace TCM_ASSERT_NAMESPACE {`. If your
/// #TCM_ASSERT_NAMESPACE is a nested namespace, you should override
/// #TCM_NAMESPACE_BEGIN because it will lead to compilation errors on C++ prior
/// to `C++17`.
#if !defined(TCM_ASSERT_NAMESPACE_BEGIN)
#    define TCM_ASSERT_NAMESPACE_BEGIN namespace TCM_ASSERT_NAMESPACE {
#endif
/// \macro TCM_ASSERT_NAMESPACE_END
///
/// Should expand to a statement which enters #TCM_ASSERT_NAMESPACE namespace.
/// By default, it expands to `}` (just closing braces). If you overwrite
/// #TCM_ASSERT_NAMESPACE_BEGIN, you should probably overwrite this macro too to
/// avoid weird compilation errors.
#if !defined(TCM_ASSERT_NAMESPACE_END)
#    define TCM_ASSERT_NAMESPACE_END } /* TCM_ASSERT_NAMESPACE */
#endif
/// \macro TCM_ASSERT_MESSAGE_HEADER
///
/// A string literal which will be printed before the actual "Assertion
/// failed..." message. Expands to an empty string by default, but you can use
/// #TCM_ASSERT_MESSAGE_HEADER to print show project name and instructions how
/// to submit a bug report. I like to use something like this
///
/// ```cpp
/// "╔═════════════════════════════════════════════════════════════════╗\n"
/// "║       Congratulations, you have found a bug in <project>!       ║\n"
/// "║              Please, be so kind to submit it here               ║\n"
/// "║         https://github.com/<username>/<project>/issues          ║\n"
/// "╚═════════════════════════════════════════════════════════════════╝\n"
/// ```
#if !defined(TCM_ASSERT_MESSAGE_HEADER)
#    define TCM_ASSERT_MESSAGE_HEADER ""
#endif

#if defined(WIN32) || defined(_WIN32)
#    define TCM_ASSERT_EXPORT __declspec(dllexport)
#    define TCM_ASSERT_FORCEINLINE __forceinline inline
#    define TCM_ASSERT_LIKELY(cond) (cond)
#    define TCM_ASSERT_CURRENT_FUNCTION __FUNCTION__
#else
#    define TCM_ASSERT_EXPORT __attribute__((visibility("default")))
#    define TCM_ASSERT_FORCEINLINE __attribute__((always_inline)) inline
#    define TCM_ASSERT_LIKELY(cond) __builtin_expect(!!(cond), 1)
#    define TCM_ASSERT_CURRENT_FUNCTION __PRETTY_FUNCTION__
#endif

/// \brief A slightly nicer alternative to `assert` macro from `<cassert>`.
///
/// This macro can be used in `constexpr` and `noexcept` functions.
#define TCM_ASSERT_ASSERT(cond, ...)                                           \
    (TCM_ASSERT_LIKELY(cond)                                                   \
         ? static_cast<void>(0)                                                \
         : ::TCM_ASSERT_NAMESPACE::assert_fail(                                \
             __FILE__, static_cast<unsigned>(__LINE__),                        \
             static_cast<char const*>(TCM_ASSERT_CURRENT_FUNCTION), #cond,     \
             __VA_ARGS__))

TCM_ASSERT_NAMESPACE_BEGIN

    [[noreturn]] auto
    assert_fail(char const* file, unsigned line, char const* function,
                char const* expression, char const* message) noexcept -> void;

/// \overload assert_fail
///
/// Allows one to work with `std::string`-like types without dragging in the
/// heavy `<string>` by default. Catching a string by const reference prolongs
/// its lifetime so calling `c_str` is safe even if you used an rvalue in
/// #TCM_ASSERT_ASSERT macro.
///
/// With this function you can use `{fmt}` lib to produce beautiful error
/// messages without reimplementing the whole functionality. On the other hand,
/// if you don't want to use `{fmt}`, you don't pay anything in compile- or
/// run-time.
template <class String>
[[noreturn]] inline auto
assert_fail(char const* file, unsigned line, char const* function,
            char const* expression, String const& message) noexcept
    -> decltype(assert_fail(
        "", 0U, "", "", (*reinterpret_cast<String const*>(nullptr)).c_str()))
{
    assert_fail(file, line, function, expression, message.c_str());
}

TCM_ASSERT_NAMESPACE_END
