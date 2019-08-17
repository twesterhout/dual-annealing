#pragma once

#include "assert.hpp"
#include "config.hpp"
#include <gsl/gsl-lite.hpp>
#include <cstddef>     // size_t
#include <cstring>     // std::memcpy
#include <type_traits> // std::aligned_storage

DA_NAMESPACE_BEGIN

struct workspace_t {
    struct point_t {
        double           func;
        gsl::span<float> x;

        explicit constexpr point_t(double const           _func,
                                   gsl::span<float> const _x) noexcept
            : func{_func}, x{_x}
        {}

        explicit constexpr point_t(gsl::span<float> const _x) noexcept
            : point_t{std::numeric_limits<double>::quiet_NaN(), _x}
        {}

        constexpr point_t(point_t const& other) noexcept = default;
        auto operator=(point_t const& other) noexcept -> point_t&
        {
            DUAL_ANNEALING_ASSERT(this != &other);
            DUAL_ANNEALING_ASSERT(x.size() == other.x.size(),
                                  "incompatible dimensions");
            func = other.func;
            std::memcpy(x.data(), other.x.data(), x.size() * sizeof(float));
            return *this;
        }
    };

    point_t current;
    point_t proposed;
    point_t best;
};

struct sa_buffers_t {
  private:
    struct impl_t;
    using storage_type = std::aligned_storage_t<64, 8>;
    storage_type _storage;

    inline auto impl() noexcept -> impl_t&;

  public:
    sa_buffers_t() noexcept;
    explicit sa_buffers_t(size_t size);

    sa_buffers_t(sa_buffers_t const&) = delete;
    sa_buffers_t(sa_buffers_t&&) noexcept;
    auto operator=(sa_buffers_t const&) -> sa_buffers_t& = delete;
    auto operator=(sa_buffers_t&&) noexcept -> sa_buffers_t&;

    auto resize(size_t size) -> void;
    auto workspace() noexcept -> workspace_t;
};

DA_NAMESPACE_END
