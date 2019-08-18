#pragma once

#include "assert.hpp"
#include "buffers.hpp"
#include "config.hpp"
#include "tsallis_distribution.hpp"

#include <gsl/gsl-lite.hpp>
#include <lbfgs/lbfgs.hpp>

#include <cmath> // std::sqrt, std::pow
#include <cstddef>
#include <cstring> // std::memcpy
#include <random>
#include <tuple>
#include <type_traits>

DA_NAMESPACE_BEGIN

struct param_t {
    float  q_V;
    float  q_A;
    float  t_0;
    size_t num_iter;
    size_t patience;
};

struct result_t {
    double func;
    size_t num_iter;
    size_t num_f_evals;
    double acceptance;
};

namespace detail {

/// \brief Determines whether `T` has a member function `wrap` which can be
/// called with a single argument of type `X`.
///
/// In other words, if we have an object `t` of type `T` and `x` of type `X`,
/// this trait determines whether `t.wrap(x)` is a valid expression.
template <class T, class X, class = void>
struct has_wrap_mem_fn : std::false_type {};

template <class T, class X>
struct has_wrap_mem_fn<
    T, X, std::void_t<decltype(std::declval<T>().wrap(std::declval<X>()))>>
    : std::true_type {};

template <class T, class X>
inline constexpr auto has_wrap_mem_fn_v = has_wrap_mem_fn<T, X>::value;

/// \brief Determines whether `T` has a member function `value` which can be
/// called with a single argument of type `gsl::span<float const>`.
///
/// In other words, if we have an object `t` of type `T` and `x` of type
/// `gsl::span<float const>`, this trait determines whether `t.value(x)` is a
/// valid expression.
template <class T, class = void> struct has_value_mem_fn : std::false_type {};

template <class T>
struct has_value_mem_fn<T, std::void_t<decltype(std::declval<T>().value(
                               std::declval<gsl::span<float const>>()))>>
    : std::true_type {};

template <class T>
inline constexpr auto has_value_mem_fn_v = has_value_mem_fn<T>::value;

/// \brief Determines whether `T` has a member function `value_from_diff` which
/// can be called with `std::pair<gsl::span<float const>, double>` and
/// `std::pair<size_t, float>`.
///
/// In other words, if we have an object `t` of type `T`, `x` of type
/// `std::pair<gsl::span<float const>, double>`, and `y` of type
/// `std::pair<size_t, float>`, this trait determines whether
/// `t.value_from_diff(x, y)` is a valid expression.
template <class T, class = void>
struct has_value_from_diff_mem_fn : std::false_type {};

template <class T>
struct has_value_from_diff_mem_fn<
    T, std::void_t<decltype(std::declval<T>().value_from_diff(
           std::declval<std::pair<gsl::span<float const>, double>>(),
           std::declval<std::pair<size_t, float>>()))>> : std::true_type {};

template <class T>
inline constexpr auto has_value_from_diff_mem_fn_v =
    has_value_from_diff_mem_fn<T>::value;

template <class Objective, class = void,
          class = std::enable_if_t<!has_wrap_mem_fn_v<Objective&, float>>>
auto do_wrap(Objective& /*unused*/, float const /*unused*/) noexcept -> void
{
    constexpr auto always_false = !std::is_same_v<Objective, Objective>;
    static_assert(always_false, "Objective is missing 'wrap' member function.");
}

template <class Objective,
          class = std::enable_if_t<has_wrap_mem_fn_v<Objective&, float>>>
DA_FORCEINLINE auto do_wrap(Objective& obj, float const x) noexcept(
    noexcept(std::declval<Objective&>().wrap(std::declval<float>()))) -> float
{
    return obj.wrap(x);
}

template <class Objective>
DA_FORCEINLINE auto
do_wrap(std::reference_wrapper<Objective> obj,
        float const x) noexcept(noexcept(do_wrap(std::declval<Objective&>(),
                                                 std::declval<float>())))
    -> float
{
    return do_wrap(obj.get(), x);
}

template <class Objective, class = void,
          class = std::enable_if_t<!has_value_mem_fn_v<Objective&>>>
auto do_value(Objective& /*unused*/, gsl::span<float const> /*unused*/) noexcept
    -> void
{
    constexpr auto always_false = !std::is_same_v<Objective, Objective>;
    static_assert(always_false,
                  "Objective is missing 'value' member function.");
}

template <class Objective,
          class = std::enable_if_t<has_value_mem_fn_v<Objective&>>>
DA_FORCEINLINE auto
do_value(Objective& obj, gsl::span<float const> x) noexcept(noexcept(
    std::declval<Objective&>().value(std::declval<gsl::span<float const>>())))
    -> double
{
    return obj.value(x);
}

template <class Objective>
DA_FORCEINLINE auto do_value(
    std::reference_wrapper<Objective> obj,
    gsl::span<float const>
        x) noexcept(noexcept(do_value(std::declval<Objective&>(),
                                      std::declval<gsl::span<float const>>())))
    -> double
{
    return do_value(obj.get(), x);
}

template <class Objective, class = void,
          class = std::enable_if_t<!has_value_from_diff_mem_fn_v<Objective&>>>
DA_FORCEINLINE auto do_value_from_diff(
    Objective& obj, std::pair<gsl::span<float const>, double> current,
    std::pair<size_t, float>
        diff) noexcept(noexcept(do_value(std::declval<Objective&>(),
                                         std::declval<
                                             gsl::span<float const>>())))
    -> double
{
    auto undo = gsl::finally(
        [p = current.first.data(), i = diff.first,
         x = current.first[diff.first]]() { const_cast<float*>(p)[i] = x; });
    const_cast<float*>(current.first.data())[diff.first] = diff.second;
    return do_value(obj, current.first);
}

template <class Objective,
          class = std::enable_if_t<has_value_from_diff_mem_fn_v<Objective&>>>
DA_FORCEINLINE auto do_value_from_diff(
    Objective& obj, std::pair<gsl::span<float const>, double> current,
    std::pair<size_t, float>
        diff) noexcept(noexcept(std::declval<Objective&>()
                                    .value_from_diff(
                                        std::declval<std::pair<
                                            gsl::span<float const>, double>>(),
                                        std::declval<
                                            std::pair<size_t, float>>())))
    -> double
{
    return obj.value_from_diff(current, diff);
}

template <class Objective>
DA_FORCEINLINE auto do_value_from_diff(
    std::reference_wrapper<Objective>         obj,
    std::pair<gsl::span<float const>, double> current,
    std::pair<size_t, float>
        diff) noexcept(noexcept(do_value_from_diff(std::declval<Objective&>(),
                                                   std::declval<std::pair<
                                                       gsl::span<float const>,
                                                       double>>(),
                                                   std::declval<std::pair<
                                                       size_t, float>>())))
    -> double
{
    return do_value_from_diff(obj.get(), current, diff);
}

} // namespace detail

template <class TargetFn, class Generator> class sa_chain_t { // {{{
  public:
    using target_fn_type = TargetFn;
    using urnbg_type     = Generator;

  private:
    target_fn_type         _target_fn;
    workspace_t&           _workspace;
    tsallis_distribution_t _tsallis_dist;
    urnbg_type&            _generator;    ///< Random number generator
    param_t const&         _params;       ///< Algorithm hyper-parameters
    size_t                 _i;            ///< Current iteration.
    size_t                 _num_accepted; ///< Number of moves accepted so far
    size_t _num_f_evals; ///< Number of function evaluations till now

  public:
    sa_chain_t(target_fn_type target_fn, workspace_t& workspace,
               param_t const& params, urnbg_type& generator) noexcept
        : _target_fn{target_fn}
        , _workspace{workspace}
        // t_0 here is arbitrary since we'll update it in operator() anyway
        , _tsallis_dist{params.q_V, params.t_0}
        , _generator{generator}
        , _params{params}
        , _i{0}
        , _num_accepted{0}
        , _num_f_evals{0}
    {
        // We only rely on `_workspace.current.x` being properly initialised.
        _workspace.current.func = value(_workspace.current.x);
        _workspace.best         = _workspace.current;
        std::memset(_workspace.proposed.x.data(), 0,
                    _workspace.proposed.x.size() * sizeof(float));
        _workspace.proposed.func = std::numeric_limits<double>::quiet_NaN();
    }

    sa_chain_t(sa_chain_t const&) = delete;
    sa_chain_t(sa_chain_t&&)      = delete;
    sa_chain_t& operator=(sa_chain_t const&) = delete;
    sa_chain_t& operator=(sa_chain_t&&) = delete;

    inline auto operator()() -> void;
    inline auto local_search(tcm::lbfgs::lbfgs_param_t const&)
        -> tcm::lbfgs::status_t;

    constexpr auto iteration() const noexcept { return _i; }
    constexpr auto num_f_evals() const noexcept { return _num_f_evals; }
    constexpr auto acceptance() const noexcept
    {
        if (_i == 0) { return std::numeric_limits<double>::quiet_NaN(); }
        // During one iteration we run a Markov Chain of length 2 * dim()
        return static_cast<double>(_num_accepted)
               / static_cast<double>(2U * _i * dim());
    }

  private:
    constexpr auto t_0() const noexcept { return _params.t_0; }
    constexpr auto q_V() const noexcept { return _params.q_V; }
    constexpr auto q_A() const noexcept { return _params.q_A; }
    /// Returns the dimension of the parameter space.
    constexpr auto dim() const noexcept { return _workspace.current.x.size(); }

    /// Calculates the visiting temperature `t_V` for iteration \p i.
    [[nodiscard]] auto temperature(size_t i) const noexcept -> float
    {
        auto const num = t_0() * (std::pow(2.0f, q_V() - 1.0f) - 1.0f);
        auto const den =
            std::pow(static_cast<float>(2 + i), q_V() - 1.0f) - 1.0f;
        return num / den;
    }

    auto value(gsl::span<float const> x) -> double
    {
        ++_num_f_evals;
        return detail::do_value(_target_fn, x);
    }

    auto value_from_diff(std::pair<gsl::span<float const>, double> current,
                         std::pair<size_t, float> diff) -> double
    {
        ++_num_f_evals;
        return detail::do_value_from_diff(_target_fn, current, diff);
    }

    template <class Accept, class Reject>
    auto accept_or_reject(float const dE, float const t_A, Accept&& accept,
                          Reject&& reject)
    {
        // Always accept moves that reduce the energy
        if (dE < 0.0f) { return std::forward<Accept>(accept)(); }

        // Eq. (5)
        //
        // pqv_temp = (q_A - 1.0) * (e - self.energy_state.current_energy) / (
        //   self.temperature_step + 1.)
        //
        auto const factor = 1.0f + (q_A() - 1.0f) * dE / t_A;
        auto const P_qA =
            factor <= 0.0f ? 0.0f : std::pow(factor, 1.0f / (1.0f - q_A()));
        if (std::uniform_real_distribution<float>{}(_generator) <= P_qA) {
            return std::forward<Accept>(accept)();
        }
        else {
            return std::forward<Reject>(reject)();
        }
    }

    inline auto generate_full() -> void
    {
        using std::begin, std::end;
        auto g = _tsallis_dist.many(_generator);
        std::transform(begin(_workspace.current.x), end(_workspace.current.x),
                       begin(_workspace.proposed.x), [this, &g](auto const x) {
                           return detail::do_wrap(_target_fn, x + g());
                       });
        _workspace.proposed.func = value(_workspace.proposed.x);
    }

    inline auto generate_one(size_t const i) -> std::tuple<float, double>
    {
        auto const x = detail::do_wrap(_target_fn, _tsallis_dist(_generator));
        auto const func = value_from_diff(
            std::make_pair(_workspace.current.x, _workspace.current.func),
            std::make_pair(i, x));
        return {x, func};
    }
};

template <class TargetFn, class Generator>
auto sa_chain_t<TargetFn, Generator>::operator()() -> void
{
    // (iv) Calculate new temperature...
    auto const t_V = temperature(_i);
    auto const t_A = t_V / static_cast<float>(_i + 1);
    _tsallis_dist.param(tsallis_distribution_t::param_type{q_V(), t_V});

    // Markov chain at constant temperature
    for (auto j = 0U; j < dim(); ++j) {
        auto const accept = [this]() {
            using std::swap;
            ++_num_accepted;
            swap(_workspace.current, _workspace.proposed);
            if (_workspace.current.func < _workspace.best.func) {
                _workspace.best = _workspace.current;
                DUAL_ANNEALING_TRACE("updating best: func=%.5e\n",
                                     _workspace.best.func);
            }
        };
        auto const reject = []() {};
        generate_full(); // In-place updates _workspace.proposed
        accept_or_reject(static_cast<float>(_workspace.proposed.func
                                            - _workspace.current.func),
                         t_A, accept, reject);
    }
    for (auto j = 0U; j < dim(); ++j) {
        auto const [x, func] = generate_one(j);
        auto const accept    = [this, j, x = x, func = func]() {
            ++_num_accepted;
            _workspace.current.x[j] = x;
            _workspace.current.func = func;
            if (_workspace.current.func < _workspace.best.func) {
                _workspace.best = _workspace.current;
                DUAL_ANNEALING_TRACE("updating best: func=%.5e\n",
                                     _workspace.best.func);
            }
        };
        auto const reject = []() {};
        accept_or_reject(static_cast<float>(func - _workspace.current.func),
                         t_A, accept, reject);
    }

    // NOTE: Don't forget this!
    ++_i;
}

template <class TargetFn, class Generator>
auto sa_chain_t<TargetFn, Generator>::local_search(
    tcm::lbfgs::lbfgs_param_t const& params) -> tcm::lbfgs::status_t
{
    _workspace.proposed = _workspace.current;
    auto r              = tcm::lbfgs::minimize(
        [this](gsl::span<float const> x, gsl::span<float> g) {
            // NOTE: We keep track of the number of function evaluations
            ++_num_f_evals;
            return _target_fn.value_and_gradient(x, g);
        },
        params, _workspace.proposed.x);
    _workspace.proposed.func = r.func;

    switch (r.status) {
    case tcm::lbfgs::status_t::too_many_iterations:
    case tcm::lbfgs::status_t::maximum_step_reached:
    case tcm::lbfgs::status_t::minimum_step_reached:
    case tcm::lbfgs::status_t::too_many_function_evaluations:
    case tcm::lbfgs::status_t::interval_too_small:
    case tcm::lbfgs::status_t::rounding_errors_prevent_progress:
        // These are all questionable cases. We proceed only if L-BFGS managed
        // to reduce the loss
        if (_workspace.proposed.func >= _workspace.current.func) {
            return tcm::lbfgs::status_t::success;
            break;
        }
    case tcm::lbfgs::status_t::success:
        using std::swap;
        swap(_workspace.proposed, _workspace.current);
        if (_workspace.current.func < _workspace.best.func) {
            _workspace.best = _workspace.current;
            DUAL_ANNEALING_TRACE(
                "updating best after local search: func=%.5e\n",
                _workspace.best.func);
        }
        return tcm::lbfgs::status_t::success;
        break;
    default:
        // All other cases are "real" errors, so we don't update
        // `_workspace.current`.
        break;
    } // end switch
    return r.status;
}
// }}}

template <class Objective, class Generator>
DA_NOINLINE auto minimize(Objective&& obj, gsl::span<float> x,
                          param_t const& parameters, Generator& generator)
    -> result_t
{
    auto workspace = thread_local_workspace(x.size());
    if (!workspace.has_value()) {
        // Memory allocation failed
        // TODO(twesterhout): Convert this to error_codes
        throw std::bad_alloc{};
    }

    std::memcpy(workspace->current.x.data(), x.data(),
                x.size() * sizeof(float));
    sa_chain_t<Objective&, Generator> chain{obj, *workspace, parameters,
                                            generator};
    auto best     = std::numeric_limits<double>::infinity();
    auto patience = parameters.patience;
    for (; chain.iteration() < parameters.num_iter && patience != 0;
         --patience) {
        chain();
        if (workspace->best.func < best) { patience = parameters.patience; }
    }
    std::memcpy(x.data(), workspace->best.x.data(), x.size() * sizeof(float));
    return result_t{/*func=*/workspace->best.func,
                    /*num_iter=*/chain.iteration(),
                    /*num_f_evals=*/chain.num_f_evals(),
                    /*acceptance=*/chain.acceptance()};
}

template <class Objective, class Generator>
DA_NOINLINE auto
minimize(Objective&& obj, gsl::span<float> x, param_t const& parameters,
         tcm::lbfgs::lbfgs_param_t const& local_search_parameters,
         Generator&                       generator) -> result_t
{
    auto workspace = thread_local_workspace(x.size());
    if (!workspace.has_value()) {
        // Memory allocation failed
        // TODO(twesterhout): Convert this to error_codes
        throw std::bad_alloc{};
    }

    std::memcpy(workspace->current.x.data(), x.data(),
                x.size() * sizeof(float));
    sa_chain_t<Objective&, Generator> chain{obj, *workspace, parameters,
                                            generator};
    auto best     = std::numeric_limits<double>::infinity();
    auto patience = parameters.patience;
    auto finalise = [x, &workspace, &chain]() {
        std::memcpy(x.data(), workspace->best.x.data(),
                    x.size() * sizeof(float));
        return result_t{/*func=*/workspace->best.func,
                        /*num_iter=*/chain.iteration(),
                        /*num_f_evals=*/chain.num_f_evals(),
                        /*acceptance=*/chain.acceptance()};
    };

    if (auto status = chain.local_search(local_search_parameters);
        status != tcm::lbfgs::status_t::success) {
        return finalise();
    }
    for (; chain.iteration() < parameters.num_iter && patience != 0;
         --patience) {
        chain();
        if (workspace->best.func < best) {
            best     = workspace->best.func;
            patience = parameters.patience;
            if (auto status = chain.local_search(local_search_parameters);
                status != tcm::lbfgs::status_t::success) {
                return finalise();
            }
        }
    }
    return finalise();
}

DA_NAMESPACE_END
