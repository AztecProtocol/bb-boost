#include "barretenberg/vm2/constraining/flavor.hpp"
#include <gtest/gtest.h>

namespace bb::avm2 {
TEST(AvmLazyEdges, ReusesStorageAcrossChangingColumnOrders)
{
    using FF = AvmFlavor::FF;
    constexpr size_t length = AvmFlavor::MAX_PARTIAL_RELATION_LENGTH;
    AvmFlavor::ProverPolynomials polynomials;
    for (size_t c = 0; c < 97; ++c) {
        auto& polynomial = polynomials.get(static_cast<ColumnAndShifts>(c));
        const size_t offset = c % 3;
        polynomial = AvmFlavor::Polynomial(37 - offset, 64, offset);
        for (size_t row = offset; row < 37; ++row) {
            polynomial.at(row) = (row + c) % 5 == 0 ? FF(0) : FF((row * 31 + c) % 11);
        }
    }
    AvmFlavor::LazilyExtendedProverUnivariates edges(polynomials);
    for (size_t edge : std::array<size_t, 9>{ 0, 14, 36, 38, 62, 2, 0, 0, 34 }) {
        edges.set_current_edge(edge);
        std::vector<std::pair<const bb::Univariate<FF, length>*, bb::Univariate<FF, length>>> previous;
        for (size_t j = 0; j < 131; ++j) {
            const auto column = static_cast<ColumnAndShifts>((j * 73) % 131);
            const auto& polynomial = polynomials.get(column);
            const auto expected = polynomial.is_empty()
                                      ? bb::Univariate<FF, length>::zero()
                                      : bb::Univariate<FF, 2>({ polynomial[edge], polynomial[edge + 1] })
                                            .template extend_to<length>();
            const auto& actual = edges.get(column);
            EXPECT_EQ(actual, expected);
            EXPECT_EQ(&actual, &edges.get(column));
            previous.emplace_back(&actual, expected);
        }
        // Growing or reusing the pool must not invalidate other columns on this edge.
        for (const auto& [actual, expected] : previous) {
            EXPECT_EQ(*actual, expected);
        }
    }
}

TEST(AvmLazyEdges, CachesZeroAndOneEdgesWithoutStaleValues)
{
    using FF = AvmFlavor::FF;
    AvmFlavor::ProverPolynomials polynomials;
    const auto column = static_cast<ColumnAndShifts>(0);
    auto& polynomial = polynomials.get(column);
    polynomial = AvmFlavor::Polynomial(8);
    for (size_t row = 0; row < 8; ++row) {
        polynomial.at(row) = row >= 2 && row < 4 ? FF(1) : FF(0);
    }
    AvmFlavor::LazilyExtendedProverUnivariates edges(polynomials);
    for (size_t edge : std::array<size_t, 6>{ 0, 2, 4, 2, 6, 0 }) {
        edges.set_current_edge(edge);
        const auto expected = bb::Univariate<FF, AvmFlavor::MAX_PARTIAL_RELATION_LENGTH>(polynomial[edge]);
        EXPECT_EQ(edges.get(column), expected);
        EXPECT_EQ(edges.get(column), expected);
    }
}
} // namespace bb::avm2
