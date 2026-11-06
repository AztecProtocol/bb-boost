#include "barretenberg/vm2/constraining/flavor.hpp"
#include <gtest/gtest.h>

namespace bb::avm2 {
namespace {
template <size_t Width> struct CompressionSettings {
    static constexpr auto SRC_COLUMNS = [] {
        std::array<ColumnAndShifts, Width> columns{};
        for (size_t i = 0; i < Width; ++i) {
            columns[i] = static_cast<ColumnAndShifts>(i);
        }
        return columns;
    }();
    static constexpr auto DST_COLUMNS = [] {
        std::array<ColumnAndShifts, Width> columns{};
        for (size_t i = 0; i < Width; ++i) {
            columns[i] = static_cast<ColumnAndShifts>(Width + i);
        }
        return columns;
    }();
};

template <size_t Width, size_t Length> void check_compression()
{
    using FF = AvmFlavor::FF;
    using Settings = CompressionSettings<Width>;
    using Accumulator = bb::Univariate<FF, Length>;
    struct Parameters {
        FF beta;
        FF gamma;
    };
    AvmFlavor::ProverPolynomials polynomials;
    for (size_t c = 0; c < 2 * Width; ++c) {
        auto& polynomial = polynomials.get(static_cast<ColumnAndShifts>(c));
        const size_t start = c % 3;
        polynomial = AvmFlavor::Polynomial(37 - start, 64, start);
        for (size_t row = start; row < 37; ++row) {
            polynomial.at(row) = (row + c) % 5 < 2 ? FF(0) : FF(row * 31 + c);
        }
    }
    AvmFlavor::LazilyExtendedProverUnivariates edges(polynomials);
    for (FF beta : { FF(0), FF(1), -FF(1), FF(17), -FF(97) }) {
        const Parameters params{ beta, -FF(13) };
        for (size_t edge : std::array<size_t, 7>{ 0, 2, 14, 36, 38, 62, 0 }) {
            edges.set_current_edge(edge);
            const auto actual = edges.template compress_lookup_pair<Accumulator, Settings>(params);
            for (size_t side = 0; side < 2; ++side) {
                Accumulator expected(FF(0));
                for (size_t i = 0; i < Width; ++i) {
                    const auto column = side == 0 ? Settings::SRC_COLUMNS[i] : Settings::DST_COLUMNS[i];
                    expected = expected * beta + typename Accumulator::View(edges.get(column));
                }
                expected += params.gamma;
                // Extending after compression can change coarse field representatives.
                // The serialization must contain exactly the same canonical field bytes.
                EXPECT_EQ(::to_buffer(actual[side].evaluations), ::to_buffer(expected.evaluations));
            }
        }
    }
}
} // namespace

TEST(AvmLazyEdges, PairedLookupCompressionMatchesFullEvaluation)
{
    check_compression<2, 4>();
    check_compression<3, 5>();
    check_compression<8, 5>();
    check_compression<40, 5>();
    check_compression<8, 7>();
}
} // namespace bb::avm2
