#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

#include "barretenberg/ecc/curves/bn254/fr.hpp"
#include "barretenberg/numeric/random/engine.hpp"

namespace bb::avm2::constraining {
namespace {

using FF = bb::fr;

// The inverses round batch-inverts each relation's rows in independent chunks. batch_invert leaves a zero entry zero
// and replaces every other entry by its inverse, so the result must not depend on where the chunks end.
std::vector<FF> invert_whole(std::vector<FF> values)
{
    FF::batch_invert(std::span<FF>(values));
    return values;
}

std::vector<FF> invert_chunked(std::vector<FF> values, size_t rows_per_chunk)
{
    for (size_t start = 0; start < values.size();) {
        const size_t end = start + std::min(rows_per_chunk, values.size() - start);
        FF::batch_invert(std::span<FF>(values.data() + start, end - start));
        start = end;
    }
    return values;
}

void expect_chunked_matches_whole(const std::vector<FF>& values, size_t rows_per_chunk)
{
    const auto whole = invert_whole(values);
    const auto chunked = invert_chunked(values, rows_per_chunk);
    ASSERT_EQ(whole.size(), chunked.size());
    for (size_t i = 0; i < values.size(); ++i) {
        ASSERT_EQ(chunked[i], whole[i]) << "row " << i << " of " << values.size() << ", chunk " << rows_per_chunk;
        if (values[i].is_zero()) {
            ASSERT_TRUE(chunked[i].is_zero()) << "row " << i;
        } else {
            ASSERT_EQ(chunked[i] * values[i], FF::one()) << "row " << i;
        }
    }
}

TEST(AvmInverseChunksTest, ChunkedBatchInversionMatchesWholeRange)
{
    auto& engine = numeric::get_debug_randomness();
    for (const size_t chunk : { size_t{ 1 } << 12, size_t{ 1 } << 13 }) {
        for (const size_t length :
             { size_t{ 1 }, chunk - 1, chunk, chunk + 1, 2 * chunk - 1, 2 * chunk, 2 * chunk + 1 }) {
            std::vector<FF> values(length);
            for (auto& value : values) {
                value = FF::random_element(&engine);
            }
            expect_chunked_matches_whole(values, chunk);

            // Zeros at the start and at the end.
            auto edges = values;
            for (size_t i = 0; i < std::min<size_t>(3, length); ++i) {
                edges[i] = FF::zero();
                edges[length - 1 - i] = FF::zero();
            }
            expect_chunked_matches_whole(edges, chunk);

            // Zeros on both sides of the first chunk boundary.
            if (length > chunk) {
                auto across = values;
                for (size_t i = chunk - 2; i < std::min(length, chunk + 2); ++i) {
                    across[i] = FF::zero();
                }
                expect_chunked_matches_whole(across, chunk);
            }

            // A chunk that holds only zeros, between two that hold none.
            if (length > 2 * chunk) {
                auto empty_chunk = values;
                std::fill(empty_chunk.begin() + static_cast<std::ptrdiff_t>(chunk),
                          empty_chunk.begin() + static_cast<std::ptrdiff_t>(2 * chunk),
                          FF::zero());
                expect_chunked_matches_whole(empty_chunk, chunk);
            }
        }
    }
}

} // namespace
} // namespace bb::avm2::constraining
