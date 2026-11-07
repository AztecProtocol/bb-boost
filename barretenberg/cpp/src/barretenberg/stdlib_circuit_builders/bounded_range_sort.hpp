#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace bb::range_sort {
/**
 * Sort a large primitive range-value list using its small declared alphabet.
 * Return false without changing values when the general sort should handle it.
 * Counts use size_t: the number of equal values need not fit in uint32_t.
 */
inline bool try_counting_sort(std::vector<uint32_t>& values, uint64_t maximum)
{
    constexpr size_t MAX_ALPHABET = 1U << 16;
    constexpr size_t MIN_VALUES = 256;
    if (maximum >= MAX_ALPHABET || values.size() < MIN_VALUES) {
        return false;
    }
    const size_t alphabet = static_cast<size_t>(maximum) + 1;
    if (values.size() < alphabet) {
        return false;
    }
    // Avoid allocating counters for an already sorted valid list.
    if (std::is_sorted(values.begin(), values.end())) {
        return values.back() <= maximum;
    }
    std::vector<size_t> counts(alphabet, 0);
    for (const uint32_t value : values) {
        if (value > maximum) {
            return false; // No output writes have occurred: preserve the original invalid-input path.
        }
        ++counts[value];
    }
    size_t position = 0;
    for (size_t value = 0; value < alphabet; ++value) {
        std::fill_n(values.data() + position, counts[value], static_cast<uint32_t>(value));
        position += counts[value];
    }
    return true;
}
} // namespace bb::range_sort
