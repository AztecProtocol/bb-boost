#pragma once

#include "barretenberg/stdlib_circuit_builders/ultra_circuit_builder.hpp"

#include <cstddef>
#include <gtest/gtest.h>

namespace bb::test_utils {

enum class CompareValues : bool { No, Yes };

/**
 * @brief Expect two Ultra builders to be equal in every member.
 * @details With CompareValues::No, only the variable values are skipped, for builds whose draws differ.
 */
inline void expect_same_builder_state(const UltraCircuitBuilder& actual,
                                      const UltraCircuitBuilder& expected,
                                      const CompareValues values = CompareValues::Yes)
{
    const auto& actual_base = static_cast<const CircuitBuilderBase<fr>&>(actual);
    const auto& expected_base = static_cast<const CircuitBuilderBase<fr>&>(expected);
    if (values == CompareValues::Yes) {
        EXPECT_TRUE(actual_base == expected_base);
        EXPECT_TRUE(actual.get_variables() == expected.get_variables());
    } else {
        EXPECT_TRUE(actual_base.equal_except_values(expected_base));
    }
    EXPECT_EQ(actual.real_variable_index, expected.real_variable_index);
    EXPECT_EQ(actual.real_variable_tags, expected.real_variable_tags);
    EXPECT_EQ(actual.num_gates(), expected.num_gates());
    EXPECT_EQ(actual.failed(), expected.failed());
    EXPECT_EQ(actual.err(), expected.err());
    EXPECT_TRUE(actual.range_lists == expected.range_lists);
    EXPECT_TRUE(actual.blocks == expected.blocks);
    EXPECT_EQ(actual.memory_read_records, expected.memory_read_records);
    EXPECT_EQ(actual.memory_write_records, expected.memory_write_records);
    EXPECT_EQ(actual.rom_logup_records, expected.rom_logup_records);
    EXPECT_TRUE(actual.constant_variable_indices == expected.constant_variable_indices);
    EXPECT_TRUE(actual.rom_ram_logic == expected.rom_ram_logic);
    EXPECT_TRUE(actual.get_lookup_tables() == expected.get_lookup_tables());
    // The entries' own equality ignores their outputs.
    const auto& actual_nnf = actual.cached_partial_non_native_field_multiplications;
    const auto& expected_nnf = expected.cached_partial_non_native_field_multiplications;
    ASSERT_EQ(actual_nnf.size(), expected_nnf.size());
    for (size_t i = 0; i < actual_nnf.size(); ++i) {
        EXPECT_EQ(actual_nnf[i].a, expected_nnf[i].a);
        EXPECT_EQ(actual_nnf[i].b, expected_nnf[i].b);
        EXPECT_EQ(actual_nnf[i].lo_0, expected_nnf[i].lo_0);
        EXPECT_EQ(actual_nnf[i].hi_0, expected_nnf[i].hi_0);
        EXPECT_EQ(actual_nnf[i].hi_1, expected_nnf[i].hi_1);
    }
    EXPECT_TRUE(actual.ipa_proof == expected.ipa_proof);
    EXPECT_EQ(actual.pending_ipa_proof.valid(), expected.pending_ipa_proof.valid());
    EXPECT_EQ(actual.get_used_witnesses(), expected.get_used_witnesses());
    EXPECT_EQ(actual.get_finalize_witnesses(), expected.get_finalize_witnesses());
    EXPECT_EQ(actual.is_recording(), expected.is_recording());
}

/**
 * @brief Expect two Ultra builders to be equal as they are and, on copies, after finalizing both.
 */
inline void expect_same_builder(const UltraCircuitBuilder& actual,
                                const UltraCircuitBuilder& expected,
                                const CompareValues values = CompareValues::Yes)
{
    expect_same_builder_state(actual, expected, values);
    UltraCircuitBuilder actual_finalized = actual;
    UltraCircuitBuilder expected_finalized = expected;
    actual_finalized.finalize_circuit();
    expected_finalized.finalize_circuit();
    expect_same_builder_state(actual_finalized, expected_finalized, values);
}

} // namespace bb::test_utils
