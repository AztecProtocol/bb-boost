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

#ifndef __wasm__
namespace bb::avm2 {
namespace {
struct ZeroAccumulationSettings {
    [[maybe_unused]] static constexpr size_t LOOKUP_TUPLE_SIZE = 2;
    [[maybe_unused]] static constexpr Column INVERSES = static_cast<Column>(0);
    [[maybe_unused]] static constexpr Column COUNTS = static_cast<Column>(1);
    [[maybe_unused]] static constexpr Column SRC_SELECTOR = static_cast<Column>(2);
    [[maybe_unused]] static constexpr Column DST_SELECTOR = static_cast<Column>(3);
    [[maybe_unused]] static constexpr std::array SRC_COLUMNS{ static_cast<ColumnAndShifts>(4),
                                                            static_cast<ColumnAndShifts>(5) };
    [[maybe_unused]] static constexpr std::array DST_COLUMNS{ static_cast<ColumnAndShifts>(6),
                                                            static_cast<ColumnAndShifts>(7) };
};
template <typename Settings>
concept HasZeroAccumulationProof = requires(const AvmFlavor::LazilyExtendedProverUnivariates& edges) {
    edges.lookup_accumulation_is_zero(std::type_identity<Settings>{});
};
static_assert(HasZeroAccumulationProof<lookup_settings<ZeroAccumulationSettings>>);
static_assert(!HasZeroAccumulationProof<ZeroAccumulationSettings>);
static_assert(!HasZeroAccumulationProof<permutation_settings<ZeroAccumulationSettings>>);
} // namespace

TEST(AvmZeroAccumulation, EndpointProofAndBothCanonicalSubrelations)
{
    using FF = AvmFlavor::FF;
    using A = bb::Univariate<FF, 5>;
    using Settings = lookup_settings<ZeroAccumulationSettings>;
    using Relation = GenericLookupRelationImpl<Settings, FF>;
    struct Parameters { FF beta; FF gamma; };
    for (size_t mask = 0; mask < 8; ++mask) {
        AvmFlavor::ProverPolynomials polynomials;
        for (size_t c = 0; c < 8; ++c) {
            auto& poly = polynomials.get(static_cast<ColumnAndShifts>(c));
            poly = AvmFlavor::Polynomial(4, 8);
            const bool zero = (c == 2 && (mask & 1) != 0) || (c == 3 && (mask & 2) != 0) ||
                              (c == 1 && (mask & 4) != 0);
            for (size_t row = 0; row < 4; ++row) {
                poly.at(row) = zero ? FF(0) : FF(3 + c * 11 + row * 7);
            }
            if (zero) {
                // p is a permitted coarse representation of zero in [0, 2p).
                for (size_t limb = 0; limb < 4; ++limb) {
                    poly.at(0).data[limb] = FF::modulus.data[limb];
                }
            }
        }
        AvmFlavor::LazilyExtendedProverUnivariates edges(polynomials);
        for (size_t edge : std::array<size_t, 4>{ 0, 2, 4, 6 }) {
            edges.set_current_edge(edge);
            EXPECT_EQ(edges.lookup_accumulation_is_zero(std::type_identity<Settings>{}),
                      edge >= 4 || ((mask & 1) != 0 && (mask & 6) != 0));
            for (const FF beta : { FF(0), FF(1), FF(17) }) {
                const Parameters params{ beta, -FF(19) };
                std::array<A, 2> actual{ A(FF(23)), A(-FF(29)) };
                auto expected = actual;
                Relation::accumulate(actual, edges, params, FF(31));
                const auto lookup = Relation::compute_lookup_term<A, 0>(edges, params);
                const auto table = Relation::compute_table_term<A, 0>(edges, params);
                const auto inverse = A(typename A::View(Relation::get_inverse_polynomial(edges)));
                const auto exists = Relation::compute_inverse_exists<A>(edges);
                const auto source_predicate = Relation::get_lookup_term_predicate<A, 0>(edges);
                const auto table_predicate = Relation::get_table_term_predicate<A, 0>(edges);
                const auto count = Relation::lookup_read_counts<A, 0>(edges);
                // Original general algorithm: arbitrary selectors/counts, no valid-witness assumption.
                expected[0] += ((table * lookup) * inverse - exists) * FF(31);
                expected[1] += source_predicate * (inverse * table);
                expected[1] -= (table_predicate * (lookup * inverse)) * count;
                EXPECT_EQ(::to_buffer(actual[0].evaluations), ::to_buffer(expected[0].evaluations));
                EXPECT_EQ(::to_buffer(actual[1].evaluations), ::to_buffer(expected[1].evaluations));
                auto cached = std::array<A, 2>{ A(FF(23)), A(-FF(29)) };
                Relation::accumulate(cached, edges, params, FF(31));
                EXPECT_EQ(::to_buffer(cached[0].evaluations), ::to_buffer(expected[0].evaluations));
                EXPECT_EQ(::to_buffer(cached[1].evaluations), ::to_buffer(expected[1].evaluations));
            }
        }
    }
}
} // namespace bb::avm2
#endif
