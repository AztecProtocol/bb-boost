#include "barretenberg/vm2/constraining/flavor.hpp"
#include "barretenberg/polynomials/fold_stride2.hpp"
#include <gtest/gtest.h>

namespace bb::avm2 {
TEST(AvmProverPolynomials, FirstFoldWritesEveryUninitializedSlot)
{
    using FF = AvmFlavor::FF;
    using Polynomial = AvmFlavor::Polynomial;
    AvmFlavor::ProverPolynomials source;
    size_t column = 0;
    for (size_t start : std::array<size_t, 5>{ 0, 1, 2, 5, 16 }) {
        for (size_t size : std::array<size_t, 8>{ 0, 1, 2, 3, 17, 32, 63, 72 }) {
            auto& polynomial = source.get(static_cast<ColumnAndShifts>(column++));
            polynomial = Polynomial(size, 128, start);
            for (size_t i = start; i < start + size; ++i) {
                polynomial.at(i) = i % 3 == 0 ? FF(0) : FF(i * 29 + size);
            }
        }
    }
    AvmFlavor::ProverPolynomials actual(source, 128);
    for (size_t c = 0; c < column; ++c) {
        const auto& input = source.get(static_cast<ColumnAndShifts>(c));
        auto& output = actual.get(static_cast<ColumnAndShifts>(c));
        const size_t desired_size = input.end_index() / 2 + input.end_index() % 2;
        Polynomial expected(desired_size, 64);
        fold_stride2_backed(input, expected, FF(13));
        fold_stride2_backed(input, output, FF(13));
        EXPECT_EQ(output.start_index(), expected.start_index());
        EXPECT_EQ(output.end_index(), expected.end_index());
        for (size_t row = 0; row < 64; ++row) {
            EXPECT_EQ(output[row], expected[row]);
        }
        // Subsequent in-place folds must also observe exactly the prior table.
        fold_stride2_backed(expected, expected, FF(19));
        fold_stride2_backed(output, output, FF(19));
        for (size_t row = 0; row < 32; ++row) {
            EXPECT_EQ(output[row], expected[row]);
        }
    }
}
} // namespace bb::avm2
