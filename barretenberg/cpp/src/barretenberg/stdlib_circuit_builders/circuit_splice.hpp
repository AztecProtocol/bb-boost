#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bb {

/**
 * @brief Selects the builder constructor whose builder records what UltraCircuitBuilder_::splice must replay of it.
 */
struct SpliceRecording {};

/**
 * @brief What a builder constructed with SpliceRecording records of its own construction: the changes to copy cycles
 * and tags in order, and where each new constant and range list put the rows a splice may drop.
 */
struct SpliceLog {
    struct Event {
        enum class Kind : uint8_t {
            AssertEqual, // assert_equal(a, b)
            Tag,         // the class of variable a takes tag b
        };
        Kind kind;
        uint32_t a;
        uint32_t b;
    };
    // A constant created on a cache miss, with the position of its fixing row and of its used-witness entry.
    struct Constant {
        uint32_t variable;
        size_t arithmetic_row;
        size_t used_witness;
    };
    // A range list: its first `staircase_size` entries are its staircase, whose rows start at `arithmetic_row`.
    struct RangeList {
        uint64_t target_range;
        uint32_t range_tag;
        uint32_t staircase_size;
        size_t arithmetic_row;
    };

    // The state the builder had when recording started: seeds, then the zero constant and its fixing row.
    uint32_t num_seeds = 0;
    uint32_t num_prefix_variables = 0;
    size_t num_prefix_arithmetic_rows = 0;
    size_t num_prefix_gates = 0;
    size_t num_prefix_used_witnesses = 0;

    std::vector<Event> events;
    std::vector<Constant> constants;
    std::vector<RangeList> range_lists;
};

} // namespace bb
