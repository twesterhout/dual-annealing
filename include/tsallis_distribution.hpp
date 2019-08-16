#pragma once

#include "assert.hpp"
#include "config.hpp"

#include <cmath> // std::sqrt, std::pow
#include <cstddef>
#include <random>

DA_NAMESPACE_BEGIN

struct tsallis_distribution_t {
  private:
    /// \brief Calculation of \f$p\f$ given \f$q_V\f$ (see `Tsallis_RNG`
    /// function in [@Schanze2006] for an explanation).
    ///
    /// [@Schanze2006]: Thomas Schanze, "An exact D-dimensional Tsallis random
    ///                 number generator for generalized simulated annealing",
    ///                 2006.
    static constexpr auto get_p(double const q_V) noexcept -> double
    {
        DUAL_ANNEALING_ASSERT(1.0 < q_V && q_V < 3.0,
                              "`q_V` must be in (1, 3)");
        return (3.0 - q_V) / (2.0 * (q_V - 1.0));
    }

    /// \brief Calculation of \f$s\f$ given \f$q_V\f$ and \f$t_V\f$ (see
    /// `Tsallis_RNG` function in [@Schanze2006] for an explanation).
    ///
    /// [@Schanze2006]: Thomas Schanze, "An exact D-dimensional Tsallis random
    ///                 number generator for generalized simulated annealing",
    ///                 2006.
    static /*constexpr*/ auto get_s(double const q_V, double const t_V) noexcept
        -> double
    {
        return std::sqrt(2.0 * (q_V - 1.0)) / std::pow(t_V, 1.0 / (3.0 - q_V));
    }

  public:
    class param_type {
        double _q_V; ///< Visiting distribution shape parameter
        double _t_V; ///< Visiting temperature
        double _s;   ///< Variable `s` from `Tsallis_RNG` in [@Schanze2006].

        // This is a hack to persuade Clang to generate constexpr `param()`
        // function for tsallis_distribution
        constexpr param_type() noexcept : _q_V{0.0}, _t_V{0.0}, _s{0.0} {}

      public:
        explicit /*constexpr*/ param_type(double const q_V,
                                          double const t_V) noexcept
            : _q_V{q_V}, _t_V{t_V}
        {
            DUAL_ANNEALING_ASSERT(1.0 < q_V && q_V < 3.0,
                                  "`q_V` must be in (1, 3)");
            DUAL_ANNEALING_ASSERT(t_V > 0, "`t_V` must be positive");
            _s = get_s(q_V, t_V);
        }

        constexpr param_type(param_type const&) noexcept = default;
        constexpr param_type(param_type&&) noexcept      = default;

        constexpr auto operator=(param_type const& other) noexcept
            -> param_type&
        {
            _q_V = other._q_V;
            _t_V = other._t_V;
            _s   = other._s;
            return *this;
        }

        constexpr auto operator=(param_type&& other) noexcept -> param_type&
        {
            _q_V = other._q_V;
            _t_V = other._t_V;
            _s   = other._s;
            return *this;
        }

        constexpr auto q_V() const noexcept { return _q_V; }
        constexpr auto t_V() const noexcept { return _t_V; }
        constexpr auto s() const noexcept { return _s; }
    };

    explicit tsallis_distribution_t(double const q_V, double const t_V) noexcept
        : _gamma_dist{static_cast<float>(get_p(q_V)), 1.0f}
        , _normal_dist{0.0f, 1.0f}
        , _params{q_V, t_V}
    {}

    tsallis_distribution_t(tsallis_distribution_t const&) noexcept = default;
    tsallis_distribution_t(tsallis_distribution_t&&) noexcept      = default;
    auto operator                  =(tsallis_distribution_t const&) noexcept
        -> tsallis_distribution_t& = default;
    auto operator                  =(tsallis_distribution_t&&) noexcept
        -> tsallis_distribution_t& = default;

    [[nodiscard]] constexpr auto param() const noexcept -> param_type
    {
        return _params;
    }

    constexpr auto param(param_type const& params) noexcept -> void
    {
        _gamma_dist.param(typename std::gamma_distribution<float>::param_type{
            static_cast<float>(get_p(params.q_V())), 1.0});
        _params = params;
    }

    template <class Generator> auto one(Generator& generator) noexcept -> float
    {
        auto const u = _gamma_dist(generator);
        auto const y = static_cast<float>(_params.s()) * std::sqrt(u);
        auto const x = _normal_dist(generator);
        return x / y;
    }

    template <class Generator> auto many(Generator& generator) const noexcept
    {
        auto const u = _gamma_dist(generator);
        auto const y = static_cast<float>(_params.s()) * std::sqrt(u);
        return [normal_dist = std::normal_distribution<float>{0.0f, 1.0f / y},
                &generator]() { return normal_dist(generator); };
    }

  private:
    std::gamma_distribution<float>  _gamma_dist;
    std::normal_distribution<float> _normal_dist;
    param_type                      _params;
};

DA_NAMESPACE_END
