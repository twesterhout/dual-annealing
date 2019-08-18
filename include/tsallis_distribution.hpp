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

#include "assert.hpp"
#include "config.hpp"

#include <cmath>  // std::sqrt, std::pow
#include <random> // std::normal_distribution, std::gamma_distribution

DUAL_ANNEALING_NAMESPACE_BEGIN

/// \brief Tsallis distribution.
///
/// \warning It __does not__ model the `RandomNumberDistribution` concept.
struct tsallis_distribution_t {
    using real_type   = float;
    using result_type = real_type;

  private:
    /// \brief Calculation of \f$p\f$ given \f$q_V\f$ (see `Tsallis_RNG`
    /// function in [@Schanze2006] for an explanation).
    ///
    /// [@Schanze2006]: Thomas Schanze, "An exact D-dimensional Tsallis random
    ///                 number generator for generalized simulated annealing",
    ///                 2006.
    static constexpr auto get_p(real_type const q_V) noexcept -> real_type
    {
        DUAL_ANNEALING_ASSERT(real_type{1} < q_V && q_V < real_type{3},
                              "`q_V` must be in (1, 3)");
        return (real_type{3} - q_V) / (real_type{2} * (q_V - real_type{1}));
    }

    /// \brief Calculation of \f$s\f$ given \f$q_V\f$ and \f$t_V\f$ (see
    /// `Tsallis_RNG` function in [@Schanze2006] for an explanation).
    ///
    /// [@Schanze2006]: Thomas Schanze, "An exact D-dimensional Tsallis random
    ///                 number generator for generalized simulated annealing",
    ///                 2006.
    static /*constexpr*/ auto get_s(real_type const q_V,
                                    real_type const t_V) noexcept -> real_type
    {
        return std::sqrt(real_type{2} * (q_V - real_type{1}))
               / std::pow(t_V, real_type{1} / (real_type{3} - q_V));
    }

  public:
    class param_type {
        real_type _q_V; ///< Visiting distribution shape parameter
        real_type _t_V; ///< Visiting temperature
        real_type _s;   ///< Variable `s` from `Tsallis_RNG` in [@Schanze2006].

        // This is a hack to persuade Clang to generate constexpr `param()`
        // function for tsallis_distribution
        constexpr param_type() noexcept : _q_V{0}, _t_V{0}, _s{0} {}

      public:
        using distribution_type = tsallis_distribution_t;

        explicit /*constexpr*/ param_type(real_type const q_V,
                                          real_type const t_V) noexcept
            : _q_V{q_V}, _t_V{t_V}
        {
            DUAL_ANNEALING_ASSERT(real_type{1} < q_V && q_V < real_type{3},
                                  "`q_V` must be in (1, 3)");
            DUAL_ANNEALING_ASSERT(t_V > real_type{0}, "`t_V` must be positive");
            _s = get_s(q_V, t_V);
        }

        constexpr param_type(param_type const&) noexcept = default;
        constexpr param_type(param_type&&) noexcept      = default;
        constexpr auto operator=(param_type const& other) noexcept
            -> param_type&     = default;
        constexpr auto operator=(param_type&& other) noexcept
            -> param_type&     = default;

        [[nodiscard]] constexpr auto q_V() const noexcept { return _q_V; }
        [[nodiscard]] constexpr auto t_V() const noexcept { return _t_V; }
        [[nodiscard]] constexpr auto s() const noexcept { return _s; }

        friend constexpr auto operator==(param_type const& lhs,
                                         param_type const& rhs) noexcept -> bool
        {
            return lhs._q_V == rhs._q_V && lhs._t_V == rhs._t_V;
        }

        friend constexpr auto operator!=(param_type const& lhs,
                                         param_type const& rhs) noexcept -> bool
        {
            return !(lhs == rhs);
        }
    };

    static_assert(std::is_trivially_copyable_v<param_type>);

    explicit tsallis_distribution_t(real_type const q_V,
                                    real_type const t_V) noexcept
        : _gamma_dist{get_p(q_V), real_type{1}}
        , _normal_dist{real_type{0}, real_type{1}}
        , _params{q_V, t_V}
    {}

    tsallis_distribution_t(tsallis_distribution_t const&) noexcept = default;
    tsallis_distribution_t(tsallis_distribution_t&&) noexcept      = default;
    auto operator                  =(tsallis_distribution_t const&) noexcept
        -> tsallis_distribution_t& = default;
    auto operator                  =(tsallis_distribution_t&&) noexcept
        -> tsallis_distribution_t& = default;

    [[nodiscard]] constexpr auto q_V() const noexcept { return _params.q_V(); }
    [[nodiscard]] constexpr auto t_V() const noexcept { return _params.t_V(); }
    [[nodiscard]] constexpr auto s() const noexcept { return _params.s(); }

    [[nodiscard]] constexpr auto param() const noexcept -> param_type
    {
        return _params;
    }

    constexpr auto param(param_type const& params) noexcept -> void
    {
        // NOTE: this is an optimisation since gamma_distribution has some
        // internal state which we don't want to screw up needlessly.
        if (_params.q_V() != params.q_V()) {
            _gamma_dist.param(
                typename std::gamma_distribution<real_type>::param_type{
                    get_p(params.q_V()), real_type{1}});
        }
        _params = params;
    }

    /// Draws a single #real_type from the 1D distribution
    template <class Generator>
    [[nodiscard]] auto operator()(Generator& generator) noexcept -> real_type
    {
        auto const u = _gamma_dist(generator);
        auto const y = _params.s() * std::sqrt(u);
        auto const x = _normal_dist(generator);
        return x / y;
    }

    /// Returns a function which can be used to draw a vector from N-D
    /// distribution.
    template <class Generator> auto many(Generator& generator) noexcept
    {
        auto const u = _gamma_dist(generator);
        auto const y = _params.s() * std::sqrt(u);
        return [normal_dist =
                    std::normal_distribution<real_type>{real_type{0},
                                                        real_type{1} / y},
                &generator]() mutable { return normal_dist(generator); };
    }

    template <int64_t D = -1> auto exact() const noexcept
    {
        static_assert(D > 0 || D == -1, "Invalid dimension");
        auto const q        = static_cast<double>(q_V());
        auto const t        = static_cast<double>(t_V());
        auto const a        = (q - 1.0) * std::pow(t, -2.0 / (3.0 - q));
        auto const scale_fn = [q, t](auto const d) {
            auto const term_1 = std::pow((q - 1.0) / M_PI, d / 2.0);
            auto const term_2 = std::tgamma(1.0 / (q - 1.0) + (d - 1.0) / 2.0)
                                / std::tgamma(1.0 / (q - 1.0) - 0.5);
            auto const term_3 = std::pow(t, d / (q - 3.0));
            return term_1 * term_2 * term_3;
        };
        auto const b_fn = [q](auto const d) {
            // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers, readability-magic-numbers)
            return 1.0 / (1.0 - q) + (1.0 - d) / 2.0;
        };

        if constexpr (D > 0) {
            auto const _D    = static_cast<double>(D);
            auto const scale = scale_fn(_D);
            auto const b     = b_fn(_D);
            return [scale, a, b](auto const x) {
                return scale * std::pow(1.0 + a * x * x, b);
            };
        }
        else {
            return [a, scale_fn, b_fn](auto const x) {
                using std::begin, std::end;
                auto const _D    = static_cast<double>(x.size());
                auto const scale = scale_fn(_D);
                auto const b     = b_fn(_D);
                auto const x_2   = std::accumulate(
                    begin(x), end(x), 0.0,
                    [](auto acc, auto const y) { return acc + y * y; });
                return scale * std::pow(1.0 + a * x_2, b);
            };
        }
    }

  private:
    std::gamma_distribution<real_type>  _gamma_dist;
    std::normal_distribution<real_type> _normal_dist;
    param_type                          _params;
};

DUAL_ANNEALING_NAMESPACE_END
