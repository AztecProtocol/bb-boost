#include "barretenberg/stdlib_circuit_builders/ultra_circuit_builder.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

using namespace bb;

namespace {
/**
 * @brief Processes a range list holding one fresh variable per value and
 * returns the values of the sorted-list variables the builder adds, which
 * process_range_list appends in sorted order.
 */
std::vector<uint64_t> sorted_by_builder(uint64_t target_range, const std::vector<uint32_t>& values)
{
    UltraCircuitBuilder builder;
    UltraCircuitBuilder::RangeList list{ .target_range = target_range,
                                         .range_tag = builder.get_new_tag(),
                                         .tau_tag = builder.get_new_tag(),
                                         .variable_indices = {} };
    for (const uint32_t value : values) {
        list.variable_indices.push_back(builder.add_variable(fr(value)));
    }
    const size_t first_sorted = builder.get_num_variables();
    builder.process_range_list(list);
    std::vector<uint64_t> sorted;
    for (size_t i = 0; i < values.size(); ++i) {
        sorted.push_back(uint256_t(builder.get_variable(static_cast<uint32_t>(first_sorted + i))).data[0]);
    }
    return sorted;
}

std::vector<uint64_t> reference_sort(const std::vector<uint32_t>& values)
{
    std::vector<uint64_t> sorted(values.begin(), values.end());
    std::sort(sorted.begin(), sorted.end());
    return sorted;
}
} // namespace

// Each case is a list with duplicates; the sorted variables must equal an
// independent sort of the values whichever way the builder sorts them.
TEST(RangeListSort, SortedVariablesMatchAReferenceSort)
{
    std::vector<uint32_t> many;
    for (uint32_t i = 0; i < 500; ++i) {
        many.push_back((i * 37) % 101);
    }
    const struct {
        const char* name;
        uint64_t target_range;
        std::vector<uint32_t> values;
    } cases[] = {
        { "small range", 50, { 7, 0, 50, 7, 13, 13, 49, 0 } },
        { "small range, many duplicates", 100, many },
        { "single element", 10, { 4 } },
        { "value above the range", 20, { 3, 25, 3, 0, 20, 25 } },
        { "range of 2^16", uint64_t(1) << 16, { 65536, 1, 40000, 1, 0, 40000 } },
        { "range above 2^16", (uint64_t(1) << 16) + 5, { 65540, 2, 65536, 2, 9 } },
    };
    for (const auto& c : cases) {
        EXPECT_EQ(sorted_by_builder(c.target_range, c.values), reference_sort(c.values)) << c.name;
    }
}
