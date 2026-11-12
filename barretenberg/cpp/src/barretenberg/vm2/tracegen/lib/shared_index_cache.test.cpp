#include "barretenberg/vm2/tracegen/lib/shared_index_cache.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

namespace bb::avm2::tracegen {
namespace {

TEST(SharedIndexCacheTest, ReleaseInParallelEmptiesTheCache)
{
    const TraceContainer trace;
    SharedIndexCache cache;
    const std::array<ColumnAndShifts, 1> columns = { ColumnAndShifts::execution_pc };
    // More row lists than one release chunk holds.
    constexpr size_t NUM_TUPLES = 40000;
    size_t builds = 0;
    const auto build = [&](const TraceContainer&) {
        ++builds;
        DstIndex index;
        for (size_t k = 0; k < NUM_TUPLES; ++k) {
            index[k] = { static_cast<uint32_t>(k), static_cast<uint32_t>(k + 1) };
        }
        return index;
    };

    const DstIndex& first = cache.get_or_build(Column::execution_sel, columns, trace, build);
    EXPECT_EQ(first.size(), NUM_TUPLES);
    EXPECT_EQ(&cache.get_or_build(Column::execution_sel, columns, trace, build), &first);
    EXPECT_EQ(builds, 1U);

    cache.release_in_parallel();

    // The cache is empty: the next request builds the index again.
    const DstIndex& rebuilt = cache.get_or_build(Column::execution_sel, columns, trace, build);
    EXPECT_EQ(builds, 2U);
    EXPECT_EQ(rebuilt.size(), NUM_TUPLES);
    EXPECT_EQ(rebuilt.at(7), (std::vector<uint32_t>{ 7, 8 }));

    cache.release_in_parallel();
    cache.release_in_parallel(); // Releasing an empty cache does nothing.
    EXPECT_EQ(cache.get_or_build(Column::execution_sel, columns, trace, build).size(), NUM_TUPLES);
    EXPECT_EQ(builds, 3U);
}

} // namespace
} // namespace bb::avm2::tracegen
