#include "barretenberg/vm2/constraining/flavor.hpp"
#include <gtest/gtest.h>

namespace bb::avm2 {
TEST(AvmLazyEdges, EndpointZeroMatchesExtendedEdge)
{
    using FF = AvmFlavor::FF;
    AvmFlavor::ProverPolynomials polynomials;
    for (size_t column = 0; column < 97; ++column) {
        auto& polynomial = polynomials.get(static_cast<ColumnAndShifts>(column));
        const size_t offset = column % 3;
        polynomial = AvmFlavor::Polynomial(37 - offset, 64, offset);
        for (size_t row = offset; row < 37; ++row) {
            polynomial.at(row) = (column + row) % 5 < 3 ? FF(0) : FF(column * 31 + row);
        }
    }
    AvmFlavor::LazilyExtendedProverUnivariates edges(polynomials);
    for (size_t edge : std::array<size_t, 9>{ 0, 2, 36, 38, 62, 14, 0, 0, 34 }) {
        edges.set_current_edge(edge);
        for (size_t column = 0; column < 131; ++column) {
            const auto c = static_cast<ColumnAndShifts>(column);
            const bool direct = edges.edge_is_zero(c);
            EXPECT_EQ(direct, edges.get(c).is_zero());
            EXPECT_EQ(direct, edges.edge_is_zero(c));
        }
    }
}
} // namespace bb::avm2
