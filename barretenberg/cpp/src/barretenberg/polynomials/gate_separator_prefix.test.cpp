#include "barretenberg/ecc/curves/bn254/fr.hpp"
#include "barretenberg/numeric/bitop/get_msb.hpp"
#include "gate_separator.hpp"

#include <algorithm>
#include <array>
#include <gtest/gtest.h>

using namespace bb;

// A table of 2^k entries, k = ceil(log2 E_0), returns the full table's value at every index the sumcheck rounds read.
// Round r reads the even edges below E_r = min(2^(d-r), W_r rounded up to even), where W_r is the largest polynomial
// end index in that round and folding takes W_{r+1} = ceil(W_r / 2).
TEST(GateSeparatorPolynomial, PrefixTableMatchesFullTableOnEveryRoundRead)
{
    constexpr size_t d = 10;
    std::vector<fr> betas(d);
    for (auto& beta : betas) {
        beta = fr::random_element();
    }
    const GateSeparatorPolynomial<fr> full_table(betas, d);

    constexpr std::array<size_t, 12> largest_ends{ 0, 1, 2, 5, 6, 64, 65, 66, 129, 1000, 1023, 1024 };
    for (const size_t largest_end : largest_ends) {
        const size_t initial_rows = largest_end + (largest_end % 2);
        const size_t log_size = initial_rows <= 1 ? 0 : numeric::get_msb(initial_rows - 1) + 1;
        GateSeparatorPolynomial<fr> full(betas, d);
        GateSeparatorPolynomial<fr> prefix(betas, log_size);
        ASSERT_EQ(prefix.beta_products.size(), size_t{ 1 } << log_size);

        size_t end = largest_end;
        for (size_t round = 0; round < d; ++round) {
            const size_t rows = std::min(size_t{ 1 } << (d - round), end + (end % 2));
            ASSERT_TRUE(rows == 0 || ((rows / 2) - 1) * prefix.periodicity < prefix.beta_products.size())
                << "largest end " << largest_end << ", round " << round;
            for (size_t idx = 0; idx < rows; idx += 2) {
                ASSERT_EQ(prefix[idx], full[idx]) << "largest end " << largest_end << ", round " << round;
            }
            const fr challenge = fr::random_element();
            full.partially_evaluate(challenge);
            prefix.partially_evaluate(challenge);
            ASSERT_EQ(prefix.partial_evaluation_result, full.partial_evaluation_result);
            end = (end + 1) / 2;
        }
    }

    // The prefix is the full table's first entries, so no index below it depends on the betas it leaves out.
    const GateSeparatorPolynomial<fr> prefix(betas, 4);
    for (size_t i = 0; i < prefix.beta_products.size(); ++i) {
        EXPECT_EQ(prefix.beta_products.at(i), full_table.beta_products.at(i));
    }
}
