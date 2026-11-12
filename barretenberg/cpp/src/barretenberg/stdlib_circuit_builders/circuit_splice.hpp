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

/**
 * @brief Why UltraCircuitBuilder_::splice left the target builder unchanged.
 */
enum class SpliceRefusal : uint8_t {
    None,
    BuilderState, // source not recording or seed count mismatch, target recording, a builder finalized or in write-VK
                  // mode
    Failed,       // a builder has failed
    PublicInput,  // the source added a public input or finalized its public inputs
    InputValue,   // a seed's value differs from the value of its target variable
    InputCycle,   // two input variables share a copy cycle, or an input index is the index of another shared variable
    InputTag,     // an input variable, or a shared constant a source tag reaches, carries a tag in the target
    RowForm,      // a row the splice cannot copy: elliptic (creation may edit an earlier row) or lookup
    UnknownForm,  // state the splice does not translate (lookup tables, IPA proof, names, unrecorded tags, tag
                  // permutation entries or constants)
};

/**
 * @brief The outcome of UltraCircuitBuilder_::splice.
 */
struct SpliceResult {
    SpliceRefusal refusal = SpliceRefusal::None;
    // When spliced: the index in the target of every source variable, indexed by source index.
    std::vector<uint32_t> index_map;
    // When spliced: added to a source pairing-point tag index, gives its index in the target.
    uint32_t pairing_tag_offset = 0;

    bool spliced() const { return refusal == SpliceRefusal::None; }
};

} // namespace bb
