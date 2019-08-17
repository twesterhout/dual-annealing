#include "tsallis_distribution.hpp"
#include <pcg_random.hpp>

#include <cerrno>
#include <charconv>
#include <cstdio>
#include <string>

using file_handler_t = std::unique_ptr<std::FILE, void (*)(std::FILE*)>;

namespace {
/// Opens `filename` for writing. "-" is handled separately and means standard
/// output.
auto open_output_file(std::string const& filename) -> file_handler_t
{
    if (filename == "-") {
        return file_handler_t{stdout, [](auto* /*unused*/) {}};
    }
    auto* fp = std::fopen(filename.c_str(), "w");
    if (fp == nullptr) {
        std::perror("open_output_file(std::string const&):\n"
                    "    std::fopen(char const*, char const*)");
        std::exit(1);
    }
    return file_handler_t{fp, [](auto* p) { std::fclose(p); }};
}

/// Reads q_V, t_V, and output file from `argv`.
auto parse_arguments(int argc, char* argv[])
    -> std::tuple<float, float, file_handler_t>
{
    if (argc != 4) {
        std::fprintf(stderr, "Usage: %s <q_V> <t_V> <filename>\n", argv[0]);
        std::exit(1);
    }
    auto const read_real = [](auto* s) {
        char*      p = nullptr;
        auto const x = std::strtof(s, &p);
        if (p == s) {
            std::fprintf(stderr, "Failed to interpret \"%s\" as double\n", s);
            std::exit(1);
        }
        return x;
    };
    auto const q_V = read_real(argv[1]);
    auto const t_V = read_real(argv[2]);
    auto       out = open_output_file(argv[3]);
    return {q_V, t_V, std::move(out)};
}
} // namespace

auto main(int argc, char* argv[]) -> int
{
    auto [q_V, t_V, out] = parse_arguments(argc, argv);
    if (q_V <= 1.0f || q_V >= 3.0f) {
        std::fprintf(stderr, "Invalid q_V: %f; expected 1.0 < q_V < 3.0\n",
                     static_cast<double>(q_V));
        std::exit(1);
    }
    if (t_V <= 0.0f) {
        std::fprintf(stderr, "Invalid t_V: %f; expected t_V > 0.0\n",
                     static_cast<double>(t_V));
        std::exit(1);
    }

    pcg32                                  generator{12349827UL};
    dual_annealing::tsallis_distribution_t dist{q_V, t_V};
    auto const generate_sample = [&generator, &dist]() {
        return static_cast<double>(dist(generator));
    };

    constexpr auto number_bins = 400;
    constexpr auto min         = -100.0;
    constexpr auto max         = 100.0;
    constexpr auto bin_size    = (max - min) / static_cast<double>(number_bins);
    std::vector<size_t> bins(number_bins, size_t{0});
    auto const          process_sample = [&bins](auto const x) {
        if (x < min || x > max) { return; }
        ++bins[static_cast<size_t>((x - min) / bin_size)];
    };
    constexpr auto number_samples = size_t{1000000};

    for (auto i = size_t{0}; i < number_samples; ++i) {
        process_sample(generate_sample());
    }

    auto exact = [p = dist.exact<1>()](auto const x) { return std::log(p(x)); };
    for (auto i = size_t{0}; i < number_bins; ++i) {
        auto const x = min + bin_size * (static_cast<double>(i) + 0.5);
        std::fprintf(out.get(), "%.5e\t%.5e\t%.5e\n", x,
                     std::log(static_cast<double>(bins[i])
                              / static_cast<double>(number_samples)),
                     exact(x));
    }
    return 0;
}
