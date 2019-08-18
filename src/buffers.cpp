#include "buffers.hpp"

#include <gsl/gsl-lite.hpp>

#include <cstring>   // std::memset
#include <memory>    // std::unique_ptr, std::aligned_alloc, etc.
#include <optional>  // std::optional
#include <stdexcept> // std::overflow_error, std::bad_alloc

DA_NAMESPACE_BEGIN

namespace detail {
template <size_t N> struct buffers_base_t {

    static constexpr auto cache_line_size = 64UL;

  private:
    struct Deleter {
        template <class T> auto operator()(T* p) const noexcept -> void
        {
            // NOLINTNEXTLINE(cppcoreguidelines-owning-memory, cppcoreguidelines-no-malloc, hicpp-no-malloc)
            std::free(p);
        }
    };

    std::unique_ptr<float[], Deleter> _data;
    size_t                            _capacity;
    size_t                            _buffer_size;

    template <size_t Alignment>
    static constexpr auto align_up(size_t const value) noexcept -> size_t
    {
        static_assert(Alignment != 0 && (Alignment & (Alignment - 1)) == 0,
                      "Invalid alignment");
        return (value + (Alignment - 1)) & ~(Alignment - 1);
    }

    static auto allocate_buffer(size_t size)
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays, hicpp-avoid-c-arrays, modernize-avoid-c-arrays)
        -> std::unique_ptr<float[], Deleter>
    {
        if (size > std::numeric_limits<size_t>::max() / sizeof(float)) {
            throw std::overflow_error{
                "integer overflow in allocate_buffer(size_t)"};
        }
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast, cppcoreguidelines-owning-memory)
        auto* p = reinterpret_cast<float*>(
            std::aligned_alloc(cache_line_size, size * sizeof(float)));
        if (p == nullptr) { throw std::bad_alloc{}; }
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays, hicpp-avoid-c-arrays, modernize-avoid-c-arrays)
        return std::unique_ptr<float[], Deleter>{p};
    }

  public:
    constexpr buffers_base_t() noexcept : _data{}, _capacity{0}, _buffer_size{0}
    {}

    explicit buffers_base_t(size_t const size)
        : _data{}, _capacity{0}, _buffer_size{0}
    {
        resize(size);
    }

    buffers_base_t(buffers_base_t const&)     = delete;
    buffers_base_t(buffers_base_t&&) noexcept = default;
    auto operator=(buffers_base_t const&) -> buffers_base_t& = delete;
    auto operator=(buffers_base_t&&) noexcept -> buffers_base_t& = default;

    auto resize(size_t const size) -> void
    {
        auto const required_capacity = align_up<cache_line_size>(size) * N;
        if (required_capacity > _capacity) { // Need to reallocate
            using std::swap;
            auto new_data = allocate_buffer(required_capacity);
            swap(_data, new_data);
            _capacity = required_capacity;
        }

        _buffer_size = size;
        // TODO(twesterhout): Profile and check whether we need to optimise this
        std::memset(_data.get(), 0, _capacity * sizeof(float));
    }

    [[nodiscard]] constexpr auto buffer_size() const noexcept -> size_t
    {
        return _buffer_size;
    }

    [[nodiscard]] constexpr auto buffer_capacity() const noexcept -> size_t
    {
        return align_up<cache_line_size>(buffer_size());
    }

    template <size_t I> constexpr auto get() noexcept -> gsl::span<float>
    {
        static_assert(I < N, "index out of bounds");
        return {_data.get() + I * buffer_capacity(), buffer_size()};
    }
};
} // namespace detail

struct sa_buffers_t::impl_t : public detail::buffers_base_t<3> {
    using detail::buffers_base_t<3>::buffers_base_t;
};

auto sa_buffers_t::impl() noexcept -> impl_t&
{
    return *reinterpret_cast<impl_t*>(&_storage);
}

DA_EXPORT sa_buffers_t::sa_buffers_t() noexcept
{
    static_assert(sizeof(impl_t) <= sizeof(storage_type));
    static_assert(alignof(impl_t) <= alignof(storage_type));
    ::new (static_cast<void*>(&_storage)) impl_t{};
}

DA_EXPORT sa_buffers_t::sa_buffers_t(size_t const size)
{
    static_assert(sizeof(impl_t) <= sizeof(storage_type));
    static_assert(alignof(impl_t) <= alignof(storage_type));
    ::new (static_cast<void*>(&_storage)) impl_t{size};
}

DA_EXPORT sa_buffers_t::sa_buffers_t(sa_buffers_t&& other) noexcept
{
    ::new (static_cast<void*>(&_storage)) impl_t{std::move(other.impl())};
}

DA_EXPORT auto sa_buffers_t::operator=(sa_buffers_t&& other) noexcept
    -> sa_buffers_t&
{
    impl() = std::move(other.impl());
    return *this;
}

DA_EXPORT auto sa_buffers_t::resize(size_t const size) -> void
{
    impl().resize(size);
}

DA_EXPORT auto sa_buffers_t::workspace() noexcept -> workspace_t
{
    using point_t = workspace_t::point_t;
    return workspace_t{point_t{impl().get<0>()}, point_t{impl().get<1>()},
                       point_t{impl().get<2>()}};
}

DA_EXPORT auto thread_local_workspace(size_t const size) noexcept
    -> std::optional<workspace_t>
{
    static thread_local sa_buffers_t buffers{};
    try {
        buffers.resize(size); // May throw
        return buffers.workspace();
    }
    // Yes, catching exceptions is not very efficient, but we expect these catch
    // blocks to never ever be reached in practice.
    catch (std::overflow_error const& e) {
        DUAL_ANNEALING_TRACE("%s: caught std::overflow_error: %s\n",
                             DUAL_ANNEALING_CURRENT_FUNCTION, e.what());
        return std::nullopt;
    }
    catch (std::bad_alloc const& e) {
        DUAL_ANNEALING_TRACE("%s: caught std::bad_alloc: %s\n",
                             DUAL_ANNEALING_CURRENT_FUNCTION, e.what());
        return std::nullopt;
    }
    catch (...) {
        DUAL_ANNEALING_ASSERT(
            false, "unexpected exception in 'thread_local_workspace(size_t)'");
        return std::nullopt;
    }
}

DA_NAMESPACE_END
