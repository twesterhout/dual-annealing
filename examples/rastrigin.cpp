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

#include "chain.hpp"
#include <pcg_random.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <random>

struct to_range_t {
    float min;
    float max;

    auto operator()(float const x) const -> float
    {
        auto const length = max - min;
        auto const delta =
            std::fmod(std::fmod(x - min, length) + length, length);
        return min + delta;
    }
};

struct rastrigin_t {
    to_range_t _wrap{-5.12f, 5.12f};

    auto value(gsl::span<float const> x) const -> float
    {
        constexpr auto A = 10.0;
        auto           E = 0.0;
        for (auto i = size_t{0}; i < x.size(); ++i) {
            auto const a = static_cast<double>(x[i]);
            E += a * a - A * std::cos(2.0 * M_PI * a);
        }
        E += A * x.size();
        return E;
    }

    auto wrap(float const x) const -> float { return _wrap(x); }

    auto value_and_gradient(gsl::span<float const> x, gsl::span<float> g) const
        -> double
    {
        constexpr auto A = 10.0;
        for (auto i = size_t{0}; i < x.size(); ++i) {
            auto const a = static_cast<double>(x[i]);
            g[i]         = static_cast<float>(
                2 * a + 2 * M_PI * A * std::sin(2.0 * M_PI * a));
        }
        return value(x);
    }
#if 0
    auto operator()(size_t const i, float const x, float const value,
                    gsl::span<float const> xs) const -> float
    {
        auto const shift = 0.0f;
        auto       E     = value;
        E -=
            std::pow(xs[i] - shift, 2.0f)
            - 10.0f
                  * std::cos(2.0f * static_cast<float>(M_PI) * (xs[i] - shift));
        E += std::pow(x - shift, 2.0f)
             - 10.0f * std::cos(2.0f * static_cast<float>(M_PI) * (x - shift));
        return E;
    }
#endif
};

int main(int argc, char* argv[])
{
    // if (argc != 2) {
    //     std::fprintf(stderr, "Expected 1 argument: <L>\n");
    //     return EXIT_FAILURE;
    // }
    // auto const length = std::atol(argv[1]);

    using std::begin, std::end;
    auto const params = dual_annealing::param_t{/*q_V=*/2.67,
                                                /*q_A=*/-5.0,
                                                /*t_0=*/10.0,
                                                // /*t_0=*/5300.0,
                                                /*num_iter=*/1000,
                                                /*patience=*/20};

    pcg32 generator{1230045};
    auto  energy_fn = rastrigin_t{};

    std::vector<float> xs(100);
    for (auto& x : xs) {
        x = std::uniform_real_distribution<float>{-1.0f, 3.0f}(generator);
    }

    std::cout << "Before: f([";
    std::copy(begin(xs), end(xs),
              std::ostream_iterator<float>{std::cout, ", "});
    std::cout << "]) = " << energy_fn.value(xs) << '\n';

    auto local_search_parameters  = tcm::lbfgs::lbfgs_param_t{};
    local_search_parameters.x_tol = 1e-5;
    auto const result             = dual_annealing::minimize(
        energy_fn, xs, params, local_search_parameters, generator);

    std::cout << "After : f([";
    std::copy(begin(xs), end(xs),
              std::ostream_iterator<float>{std::cout, ", "});
    std::cout << "]) = " << result.func << '\n';
    std::cout << "Number iterations: " << result.num_iter << '\n'
              << "Number function evaluations: " << result.num_f_evals << '\n'
              << "Acceptance: " << result.acceptance << '\n';
    return EXIT_SUCCESS;
}
