#include "barretenberg/stdlib_circuit_builders/ultra_circuit_builder.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <gtest/gtest.h>
#include <vector>

/**
 * A circuit built on a recording builder seeded with the values of some variables of a target builder, then spliced
 * into the target, must leave the target exactly as building the same circuit on the target itself does; a refused
 * splice must leave the target unchanged.
 */

namespace bb {
namespace {

using Builder = UltraCircuitBuilder;

// Compares every member of the two builders.
void expect_same_builder_state(const Builder& actual, const Builder& expected)
{
    // The base comparison covers variables, copy cycles, tags, tau, gate count, public inputs, failure state, variable
    // names and pairing-point tags.
    EXPECT_TRUE(static_cast<const CircuitBuilderBase<fr>&>(actual) ==
                static_cast<const CircuitBuilderBase<fr>&>(expected));
    EXPECT_TRUE(actual.get_variables() == expected.get_variables());
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

// Compares the builders as they are and, on copies, after finalizing both.
void expect_same_builder(const Builder& actual, const Builder& expected)
{
    expect_same_builder_state(actual, expected);
    Builder actual_finalized = actual;
    Builder expected_finalized = expected;
    actual_finalized.finalize_circuit();
    expected_finalized.finalize_circuit();
    expect_same_builder_state(actual_finalized, expected_finalized);
}

// Variables a circuit returns, as indices in the builder it ran on, and the pairing-point tags it created.
struct Outputs {
    std::vector<uint32_t> variables;
    std::vector<uint32_t> pairing_tags;
};

using Circuit = std::function<Outputs(Builder&, const std::vector<uint32_t>&)>;

// A new variable that appears in a wire, as a range constraint requires.
uint32_t add_witness(Builder& builder, const fr& value)
{
    const uint32_t index = builder.add_variable(value);
    builder.create_unconstrained_gate(
        builder.blocks.arithmetic, index, builder.zero_idx(), builder.zero_idx(), builder.zero_idx());
    return index;
}

// A target builder that already holds constants, range lists, ROM and RAM arrays, cached multiplications and
// pairing-point tags that the circuits below meet again.
struct Target {
    Builder builder;
    // Input variables: inputs[0] and inputs[3] have equal values.
    std::vector<uint32_t> inputs;

    Target()
    {
        for (const uint64_t value : { 5ULL, 1000ULL, 77ULL, 5ULL, 123456ULL, 9ULL }) {
            inputs.push_back(builder.add_variable(fr(value)));
        }
        builder.put_constant_variable(fr(3));
        builder.put_constant_variable(fr(7));
        const uint32_t small = add_witness(builder, fr(200));
        builder.create_small_range_constraint(small, 255);
        const uint32_t wide = builder.add_variable(fr(40000));
        builder.create_limbed_range_constraint(wide, 20);
        const size_t rom = builder.create_ROM_array(2);
        builder.set_ROM_element(rom, 0, small);
        builder.set_ROM_element(rom, 1, wide);
        const size_t ram = builder.create_RAM_array(1);
        builder.init_RAM_element(ram, 0, small);
        builder.queue_partial_non_native_field_multiplication(
            { .a = { small, wide, small, wide }, .b = { wide, wide, small, small } });
        builder.pairing_points_tagging.create_pairing_point_tag();
    }
};

// Constants cached in the target and new ones, existing and new range lists, ROM, RAM, non-native multiplication,
// assert_equal on seeds, and enough arithmetic rows for more than one copy task. Expects six inputs, the fourth equal
// to the first.
Outputs mixed_circuit(Builder& builder, const std::vector<uint32_t>& in)
{
    const uint32_t zero = builder.zero_idx();
    const uint32_t three = builder.put_constant_variable(fr(3));
    const uint32_t eleven = builder.put_constant_variable(fr(11));
    BB_ASSERT_EQ(builder.put_constant_variable(fr(11)), eleven);

    const fr x_value = builder.get_variable(in[0]) * builder.get_variable(in[1]) + fr(3);
    const uint32_t x = builder.add_variable(x_value);
    builder.create_big_mul_add_gate({ .a = in[0],
                                      .b = in[1],
                                      .c = x,
                                      .d = three,
                                      .mul_scaling = 1,
                                      .a_scaling = 0,
                                      .b_scaling = 0,
                                      .c_scaling = -1,
                                      .d_scaling = 1,
                                      .const_scaling = 0 });
    const uint32_t y = builder.add_variable(x_value + fr(11));
    builder.create_add_gate(
        { .a = x, .b = eleven, .c = y, .a_scaling = 1, .b_scaling = 1, .c_scaling = -1, .const_scaling = 0 });
    // The fifth input is the second again.
    builder.create_add_gate(
        { .a = in[1], .b = in[4], .c = zero, .a_scaling = 1, .b_scaling = -1, .c_scaling = 0, .const_scaling = 0 });

    // Lists of ranges 255, 63 and 2^14 - 1 exist in the target; 1023 and 4095 do not.
    builder.create_small_range_constraint(add_witness(builder, fr(200)), 255);
    builder.create_small_range_constraint(in[2], 1023);
    builder.create_limbed_range_constraint(in[1], 20);
    const uint32_t lo = add_witness(builder, fr(uint256_t(1) << 60));
    const uint32_t hi = add_witness(builder, fr(12345));
    builder.range_constrain_two_limbs(lo, hi);

    const size_t rom = builder.create_ROM_array(3);
    builder.set_ROM_element(rom, 0, in[0]);
    builder.set_ROM_element(rom, 1, x);
    builder.set_ROM_element(rom, 2, eleven);
    const uint32_t one = builder.add_variable(fr(1));
    const uint32_t rom_read = builder.read_ROM_array(rom, one);

    const size_t ram = builder.create_RAM_array(2);
    builder.init_RAM_element(ram, 0, in[0]);
    builder.init_RAM_element(ram, 1, in[2]);
    builder.write_RAM_array(ram, one, x);
    const uint32_t ram_read = builder.read_RAM_array(ram, one);

    const auto [nnf_lo, nnf_hi] = builder.queue_partial_non_native_field_multiplication(
        { .a = { in[0], in[1], x, three }, .b = { y, in[2], three, in[5] } });

    builder.assert_equal(in[0], in[3]);
    builder.assert_equal_constant(in[0], builder.get_variable(in[0]));
    const uint32_t copy = add_witness(builder, builder.get_variable(in[2]));
    builder.assert_equal(copy, in[2]);
    builder.create_small_range_constraint(copy, 1023);
    builder.update_used_witnesses(x);

    fr accumulator = builder.get_variable(y);
    uint32_t previous = y;
    for (size_t i = 0; i < 5000; ++i) {
        accumulator += builder.get_variable(in[5]);
        const uint32_t next = builder.add_variable(accumulator);
        builder.create_add_gate({ .a = previous,
                                  .b = in[5],
                                  .c = next,
                                  .a_scaling = 1,
                                  .b_scaling = 1,
                                  .c_scaling = -1,
                                  .const_scaling = 0 });
        previous = next;
    }

    auto& tagging = builder.pairing_points_tagging;
    const uint32_t first_tag = tagging.create_pairing_point_tag();
    const uint32_t second_tag = tagging.create_pairing_point_tag();
    const std::array<uint32_t, 2> aggregated{ first_tag, second_tag };
    const uint32_t result_tag = tagging.aggregate_pairing_point_tags(aggregated);

    return { .variables = { x, y, rom_read, ram_read, nnf_lo, nnf_hi, copy, previous, eleven },
             .pairing_tags = { first_tag, second_tag, result_tag } };
}

// Meets the constant 11 and the range 1023 again, and tags the cached constant 11. Expects two inputs.
Outputs follow_up_circuit(Builder& builder, const std::vector<uint32_t>& in)
{
    const uint32_t eleven = builder.put_constant_variable(fr(11));
    const uint32_t thirteen = builder.put_constant_variable(fr(13));
    builder.create_small_range_constraint(eleven, 1023);
    const uint32_t sum = add_witness(builder, builder.get_variable(in[0]) + builder.get_variable(in[1]));
    builder.create_add_gate(
        { .a = in[0], .b = in[1], .c = sum, .a_scaling = 1, .b_scaling = 1, .c_scaling = -1, .const_scaling = 0 });
    builder.create_small_range_constraint(add_witness(builder, fr(1000)), 1023);
    builder.create_dyadic_range_constraint(add_witness(builder, fr(100)), 7, "follow up");
    builder.assert_equal(sum, add_witness(builder, builder.get_variable(sum)));
    return { .variables = { eleven, thirteen, sum }, .pairing_tags = {} };
}

// The seeds of `inputs`, one per distinct index in order of first appearance, and each input's seed position.
struct Seeds {
    std::vector<uint32_t> indices;
    std::vector<fr> values;
    std::vector<uint32_t> positions;

    Seeds(const Builder& builder, const std::vector<uint32_t>& inputs)
    {
        for (const uint32_t input : inputs) {
            const auto it = std::ranges::find(indices, input);
            positions.push_back(static_cast<uint32_t>(it - indices.begin()));
            if (it == indices.end()) {
                indices.push_back(input);
                values.push_back(builder.get_variable(input));
            }
        }
    }
};

// Builds `circuit` on `serial` and, through a recording builder, splices it into `target`; checks the outputs map.
SpliceResult build_both_ways(Builder& serial,
                             Builder& target,
                             const std::vector<uint32_t>& inputs,
                             const Circuit& circuit)
{
    const Outputs serial_outputs = circuit(serial, inputs);
    const Seeds seeds(target, inputs);
    Builder recording(SpliceRecording{}, seeds.values);
    const Outputs recorded_outputs = circuit(recording, seeds.positions);
    SpliceResult result = target.splice(recording, seeds.indices);
    if (result.spliced()) {
        EXPECT_EQ(result.index_map.size(), recording.get_num_variables());
        for (size_t i = 0; i < serial_outputs.variables.size(); ++i) {
            EXPECT_EQ(result.index_map[recorded_outputs.variables[i]], serial_outputs.variables[i]);
        }
        for (size_t i = 0; i < serial_outputs.pairing_tags.size(); ++i) {
            EXPECT_EQ(recorded_outputs.pairing_tags[i] + result.pairing_tag_offset, serial_outputs.pairing_tags[i]);
        }
    }
    return result;
}

TEST(UltraCircuitBuilderSplice, MatchesSerialBuild)
{
    Target target;
    Builder serial = target.builder;
    const std::vector<uint32_t> inputs{
        target.inputs[0], target.inputs[1], target.inputs[2], target.inputs[3], target.inputs[1], target.inputs[5],
    };
    const SpliceResult result = build_both_ways(serial, target.builder, inputs, mixed_circuit);
    ASSERT_EQ(result.refusal, SpliceRefusal::None);
    expect_same_builder(target.builder, serial);
}

TEST(UltraCircuitBuilderSplice, TwoSplicesInARow)
{
    Target target;
    Builder serial = target.builder;
    const std::vector<uint32_t> first_inputs{
        target.inputs[0], target.inputs[1], target.inputs[2], target.inputs[3], target.inputs[1], target.inputs[5],
    };
    ASSERT_EQ(build_both_ways(serial, target.builder, first_inputs, mixed_circuit).refusal, SpliceRefusal::None);
    const std::vector<uint32_t> second_inputs{ target.inputs[4], target.inputs[5] };
    ASSERT_EQ(build_both_ways(serial, target.builder, second_inputs, follow_up_circuit).refusal, SpliceRefusal::None);
    expect_same_builder(target.builder, serial);
}

// Records `circuit` on the seeds of `inputs` with the given seed values, applies `before_splice` to a fresh target and
// the recording, then checks that the splice is refused for `expected` and changes nothing. Every Target has the same
// variable indices, so `inputs` may come from any of them.
void expect_refusal(const std::vector<uint32_t>& inputs,
                    const Circuit& circuit,
                    const std::function<void(Builder& target, Builder& recording)>& before_splice,
                    const SpliceRefusal expected,
                    const std::vector<fr>& seed_values = {})
{
    Target target;
    const Seeds seeds(target.builder, inputs);
    Builder recording(SpliceRecording{}, seed_values.empty() ? seeds.values : seed_values);
    circuit(recording, seeds.positions);
    before_splice(target.builder, recording);
    const Builder before = target.builder;
    const SpliceResult result = target.builder.splice(recording, seeds.indices);
    EXPECT_EQ(result.refusal, expected);
    EXPECT_TRUE(result.index_map.empty());
    expect_same_builder_state(target.builder, before);
}

const auto no_change = [](Builder& /*target*/, Builder& /*recording*/) {};

TEST(UltraCircuitBuilderSplice, RefusesTaggedSeed)
{
    Target target;
    const std::vector<uint32_t> inputs{ target.inputs[4], target.inputs[5] };
    expect_refusal(
        inputs,
        follow_up_circuit,
        [seed = target.inputs[5]](Builder& builder, Builder& /*recording*/) {
            builder.create_small_range_constraint(seed, 15);
        },
        SpliceRefusal::InputTag);
}

TEST(UltraCircuitBuilderSplice, RefusesSeedsSharingACycle)
{
    Target target;
    const std::vector<uint32_t> inputs{ target.inputs[0], target.inputs[3] };
    expect_refusal(
        inputs,
        follow_up_circuit,
        [&](Builder& builder, Builder& /*recording*/) { builder.assert_equal(target.inputs[0], target.inputs[3]); },
        SpliceRefusal::InputCycle);
}

TEST(UltraCircuitBuilderSplice, RefusesChangedSeedValue)
{
    Target target;
    const std::vector<uint32_t> inputs{ target.inputs[4], target.inputs[5] };
    expect_refusal(inputs, follow_up_circuit, no_change, SpliceRefusal::InputValue, { fr(123456), fr(10) });
}

TEST(UltraCircuitBuilderSplice, RefusesFailedBuilder)
{
    Target target;
    const std::vector<uint32_t> inputs{ target.inputs[4], target.inputs[5] };
    expect_refusal(
        inputs,
        follow_up_circuit,
        [](Builder& builder, Builder& /*recording*/) { builder.failure("target failed"); },
        SpliceRefusal::Failed);
    expect_refusal(
        inputs,
        follow_up_circuit,
        [](Builder& /*target*/, Builder& recording) {
            recording.create_small_range_constraint(add_witness(recording, fr(300)), 255);
        },
        SpliceRefusal::Failed);
}

TEST(UltraCircuitBuilderSplice, RefusesAddedPublicInput)
{
    Target target;
    const std::vector<uint32_t> inputs{ target.inputs[4], target.inputs[5] };
    expect_refusal(
        inputs,
        follow_up_circuit,
        [](Builder& /*target*/, Builder& recording) { recording.add_public_variable(fr(42)); },
        SpliceRefusal::PublicInput);
}

TEST(UltraCircuitBuilderSplice, RefusesFinalizedPublicInputs)
{
    Target target;
    const std::vector<uint32_t> inputs{ target.inputs[4], target.inputs[5] };
    expect_refusal(
        inputs,
        follow_up_circuit,
        [](Builder& /*target*/, Builder& recording) { recording.finalize_public_inputs(); },
        SpliceRefusal::PublicInput);
}

TEST(UltraCircuitBuilderSplice, RefusesUnrecordedTagPermutationEntry)
{
    Target target;
    const std::vector<uint32_t> inputs{ target.inputs[4], target.inputs[5] };
    expect_refusal(
        inputs,
        follow_up_circuit,
        [](Builder& /*target*/, Builder& recording) {
            // A tag past the current one, so no other count changes.
            const uint32_t tag = recording.get_current_tag() + 1;
            recording.set_tau_at_index(tag, tag);
        },
        SpliceRefusal::UnknownForm);
}

TEST(UltraCircuitBuilderSplice, RefusesEllipticRow)
{
    Target target;
    const std::vector<uint32_t> inputs{ target.inputs[4], target.inputs[5] };
    expect_refusal(
        inputs,
        follow_up_circuit,
        [](Builder& /*target*/, Builder& recording) {
            const uint32_t x3 = recording.add_variable(fr(1));
            const uint32_t y3 = recording.add_variable(fr(2));
            recording.create_ecc_dbl_gate({ .x1 = 0, .y1 = 1, .x3 = x3, .y3 = y3 });
        },
        SpliceRefusal::RowForm);
}

TEST(UltraCircuitBuilderSplice, RecordingBuilderReleasesBlocksUnderTheDefaultFloors)
{
// Released anonymous pages read back as zeros only on Linux.
#if !defined(__linux__) || defined(NO_MULTITHREADING)
    GTEST_SKIP() << "needs Linux and native threads";
#else
    // About 27 MiB of gate tiles and 4 MiB of values: every block is under the 32 MiB floor, and the total is under the
    // 64 MiB minimum, of a builder that does not record.
    constexpr size_t NUM_WITNESSES = 100'000;
    // A tile in the middle of the largest tile chunk, and a variable in the middle of every per-variable array.
    constexpr size_t PROBE_TILE = 10'000;
    constexpr uint32_t PROBE_VARIABLE = NUM_WITNESSES / 2;
    const auto fill = [](Builder& builder) {
        for (size_t i = 0; i < NUM_WITNESSES; ++i) {
            add_witness(builder, fr(i + 1));
        }
    };
    const auto probes_released = [](const Builder& builder) {
        return builder.get_variables()[PROBE_VARIABLE] == fr(0) && builder.real_variable_index[PROBE_VARIABLE] == 0 &&
               builder.blocks.arithmetic.tiles[PROBE_TILE].wires[0][0] == 0;
    };

    Builder plain;
    fill(plain);
    ASSERT_FALSE(probes_released(plain));
    plain.release_storage_pages();
    EXPECT_FALSE(probes_released(plain));

    Builder recording(SpliceRecording{}, std::vector<fr>{ fr(5) });
    fill(recording);
    ASSERT_FALSE(probes_released(recording));
    recording.release_storage_pages();
    EXPECT_TRUE(probes_released(recording));
#endif
}

} // namespace
} // namespace bb
