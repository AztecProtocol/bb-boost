#include "barretenberg/honk/proof_system/types/proof.hpp"
#include "barretenberg/special_public_inputs/special_public_inputs.hpp"
#include "barretenberg/srs/global_crs.hpp"
#include "barretenberg/stdlib/special_public_inputs/special_public_inputs.hpp"
#include "barretenberg/stdlib_circuit_builders/circuit_splice.hpp"
#include "barretenberg/stdlib_circuit_builders/plookup_tables/plookup_tables.hpp"
#include "barretenberg/stdlib_circuit_builders/ultra_circuit_builder.hpp"
#include "barretenberg/stdlib_circuit_builders/ultra_circuit_builder_test_utils.hpp"
#include "barretenberg/ultra_honk/prover_instance.hpp"
#include "barretenberg/ultra_honk/ultra_prover.hpp"
#include "barretenberg/ultra_honk/ultra_verifier.hpp"

#include <cstdint>
#include <functional>
#include <gtest/gtest.h>
#include <numeric>
#include <optional>
#include <span>
#include <vector>

/**
 * Lookup gadgets built in chunks on recording builders and spliced in order into a target must leave the target, and
 * its proof, exactly as building them on the target itself does, tables and their read records included; a refused
 * splice must leave the target unchanged.
 */

namespace bb {
namespace {

using Builder = UltraCircuitBuilder;
using Flavor = UltraFlavor;
using plookup::ColumnIdx;
using plookup::MultiTableId;
using test_utils::expect_same_builder;
using test_utils::expect_same_builder_state;

constexpr uint64_t CACHED_CONSTANT = 3;
constexpr size_t NUM_MESSAGES = 12;
constexpr size_t NUM_CHUNKS = 3;

// A new variable that appears in a wire, as a range constraint requires.
uint32_t add_witness(Builder& builder, const fr& value)
{
    const uint32_t index = builder.add_variable(value);
    builder.create_unconstrained_gate(
        builder.blocks.arithmetic, index, builder.zero_idx(), builder.zero_idx(), builder.zero_idx());
    return index;
}

// The result of a multi-table lookup on `key_a`, and on `key_b` for a two-key table.
uint32_t lookup(Builder& builder, const MultiTableId id, const uint32_t key_a, const std::optional<uint32_t> key_b)
{
    const fr b_value = key_b.has_value() ? builder.get_variable(*key_b) : fr(0);
    const auto accumulators =
        plookup::get_lookup_accumulators(id, builder.get_variable(key_a), b_value, key_b.has_value());
    return builder.create_gates_from_plookup_accumulators(id, accumulators, key_a, key_b)[ColumnIdx::C3][0];
}

// One step in the style of a SHA-256 round: a 32-bit XOR of `chain` and `message`, the majority-function input form of
// the result, then a 32-bit AND of the result and `message`, which is returned and also offset by a cached constant.
uint32_t lookup_gadget(Builder& builder, const uint32_t chain, const uint32_t message)
{
    const uint32_t mixed = lookup(builder, MultiTableId::UINT32_XOR, chain, message);
    lookup(builder, MultiTableId::SHA256_MAJ_INPUT, mixed, std::nullopt);
    const uint32_t result = lookup(builder, MultiTableId::UINT32_AND, mixed, message);
    const uint32_t constant = builder.put_constant_variable(fr(CACHED_CONSTANT));
    const uint32_t offset = builder.add_variable(builder.get_variable(result) + fr(CACHED_CONSTANT));
    builder.create_add_gate({ .a = result,
                              .b = constant,
                              .c = offset,
                              .a_scaling = 1,
                              .b_scaling = 1,
                              .c_scaling = -1,
                              .const_scaling = 0 });
    return result;
}

uint32_t run_gadgets(Builder& builder, uint32_t chain, const std::span<const uint32_t> messages)
{
    for (const uint32_t message : messages) {
        chain = lookup_gadget(builder, chain, message);
    }
    return chain;
}

// A target that already holds an AND table pair, so the tables the gadgets create here are numbered after it while a
// recording numbers them from 1; a range list with a tagged variable; and the constant the gadgets cache.
struct Target {
    Builder builder;
    uint32_t chain;
    // Messages 2i and 2i + 1 have equal values.
    std::vector<uint32_t> messages;
    uint32_t tagged;

    explicit Target(const size_t num_messages)
    {
        const uint32_t left = builder.add_variable(fr(0x12345678));
        const uint32_t right = builder.add_variable(fr(0x0f0f0f0f));
        lookup(builder, MultiTableId::UINT32_AND, left, right);
        builder.put_constant_variable(fr(CACHED_CONSTANT));
        tagged = add_witness(builder, fr(200));
        builder.create_small_range_constraint(tagged, 255);
        chain = builder.add_variable(fr(0x6a09e667));
        for (size_t i = 0; i < num_messages; ++i) {
            const uint32_t value = 0x9e3779b9U * static_cast<uint32_t>((i / 2) + 1);
            messages.push_back(builder.add_variable(fr(static_cast<uint64_t>(value))));
        }
    }
};

// A recording builder seeded with `chain` and `messages` of `target`, the gadgets built on it, and the result there.
struct Recording {
    std::vector<uint32_t> seeds;
    Builder builder;
    uint32_t result;

    Recording(const Builder& target, const uint32_t chain, const std::span<const uint32_t> messages)
        : seeds(seeds_of(chain, messages))
        , builder(SpliceRecording{}, values_of(target, seeds))
    {
        std::vector<uint32_t> positions(messages.size());
        std::iota(positions.begin(), positions.end(), uint32_t{ 1 });
        result = run_gadgets(builder, 0, positions);
    }

    static std::vector<uint32_t> seeds_of(const uint32_t chain, const std::span<const uint32_t> messages)
    {
        std::vector<uint32_t> seeds{ chain };
        seeds.insert(seeds.end(), messages.begin(), messages.end());
        return seeds;
    }

    static std::vector<fr> values_of(const Builder& target, const std::vector<uint32_t>& seeds)
    {
        std::vector<fr> values;
        for (const uint32_t seed : seeds) {
            values.push_back(target.get_variable(seed));
        }
        return values;
    }
};

// Builds the gadgets on `messages` from `chain` on `serial`, and in `num_chunks` equal chunks on recording builders
// spliced in order into `target`; checks that each chunk is spliced and that the results agree, and returns the result.
uint32_t build_both_ways(Builder& serial,
                         Builder& target,
                         const uint32_t chain,
                         const std::span<const uint32_t> messages,
                         const size_t num_chunks)
{
    const uint32_t serial_result = run_gadgets(serial, chain, messages);
    const size_t chunk_size = messages.size() / num_chunks;
    uint32_t target_chain = chain;
    for (size_t c = 0; c < num_chunks; ++c) {
        const Recording recording(target, target_chain, messages.subspan(c * chunk_size, chunk_size));
        const SpliceResult result = target.splice(recording.builder, recording.seeds);
        EXPECT_EQ(result.refusal, SpliceRefusal::None);
        if (!result.spliced()) {
            return serial_result;
        }
        target_chain = result.index_map[recording.result];
    }
    EXPECT_EQ(target_chain, serial_result);
    return serial_result;
}

// A later gate reads the result and a later variable joins its copy cycle, as the next constraint would.
void use_result(Builder& builder, const uint32_t result, const uint32_t tagged)
{
    builder.assert_equal(add_witness(builder, builder.get_variable(result)), result);
    const uint32_t sum = builder.add_variable(builder.get_variable(result) + builder.get_variable(tagged));
    builder.create_add_gate(
        { .a = result, .b = tagged, .c = sum, .a_scaling = 1, .b_scaling = 1, .c_scaling = -1, .const_scaling = 0 });
}

HonkProof prove_and_verify(Builder builder)
{
    stdlib::recursion::honk::DefaultIO<Builder>::add_default(builder);
    auto instance = std::make_shared<ProverInstance_<Flavor>>(builder);
    auto vk = std::make_shared<Flavor::VerificationKey>(instance->get_precomputed());
    UltraProver_<Flavor> prover(instance, vk);
    auto proof = prover.construct_proof();
    UltraVerifier_<Flavor, DefaultIO> verifier(std::make_shared<Flavor::VKAndHash>(vk));
    EXPECT_TRUE(verifier.verify_proof(proof).result);
    return proof;
}

class SpliceLookup : public ::testing::Test {
  protected:
    static void SetUpTestSuite() { bb::srs::init_file_crs_factory(bb::srs::bb_crs_path()); }
};

TEST_F(SpliceLookup, ChunksMatchTheSerialBuildAndItsProof)
{
    Target target(NUM_MESSAGES);
    Builder serial = target.builder;
    const uint32_t result = build_both_ways(serial, target.builder, target.chain, target.messages, NUM_CHUNKS);
    use_result(serial, result, target.tagged);
    use_result(target.builder, result, target.tagged);
    expect_same_builder(target.builder, serial);
    EXPECT_EQ(prove_and_verify(target.builder), prove_and_verify(serial));
}

TEST_F(SpliceLookup, SecondRunSplicedOntoTheResult)
{
    Target target(2 * NUM_MESSAGES);
    Builder serial = target.builder;
    const std::span<const uint32_t> messages(target.messages);
    const uint32_t first =
        build_both_ways(serial, target.builder, target.chain, messages.first(NUM_MESSAGES), NUM_CHUNKS);
    use_result(serial, first, target.tagged);
    use_result(target.builder, first, target.tagged);
    const uint32_t second = build_both_ways(serial, target.builder, first, messages.subspan(NUM_MESSAGES), NUM_CHUNKS);
    use_result(serial, second, target.tagged);
    use_result(target.builder, second, target.tagged);
    expect_same_builder(target.builder, serial);
}

// Records the gadgets on the first two messages, applies `before_splice` to the target and the recording, then checks
// that the splice is refused for `expected` and changes nothing.
void expect_refusal(const std::function<void(Target& target, Builder& recording)>& before_splice,
                    const SpliceRefusal expected)
{
    Target target(2);
    Recording recording(target.builder, target.chain, target.messages);
    ASSERT_NE(recording.builder.blocks.lookup.size(), 0U);
    before_splice(target, recording.builder);
    const Builder before = target.builder;
    const SpliceResult result = target.builder.splice(recording.builder, recording.seeds);
    EXPECT_EQ(result.refusal, expected);
    EXPECT_TRUE(result.index_map.empty());
    expect_same_builder_state(target.builder, before);
}

TEST_F(SpliceLookup, RefusalsBesideLookupRowsLeaveTheTargetUnchanged)
{
    // An elliptic row: its creation may edit an earlier row.
    expect_refusal(
        [](Target& /*target*/, Builder& recording) {
            const uint32_t x3 = recording.add_variable(fr(1));
            const uint32_t y3 = recording.add_variable(fr(2));
            recording.create_ecc_dbl_gate({ .x1 = 0, .y1 = 1, .x3 = x3, .y3 = y3 });
        },
        SpliceRefusal::RowForm);
    // A table numbered other than by its registry position, which the rows' table numbers rely on.
    expect_refusal([](Target& /*target*/, Builder& recording) { recording.get_lookup_tables()[1].table_index = 9; },
                   SpliceRefusal::UnknownForm);
    // A registered table, whose contents its id does not fix.
    expect_refusal(
        [](Target& /*target*/, Builder& recording) {
            plookup::BasicTable table{};
            table.id = plookup::BasicTableId::STRAUS_EC_POINT;
            recording.register_basic_lookup_table(std::move(table));
        },
        SpliceRefusal::UnknownForm);
    // Two seeds of equal value joined in one copy cycle here after the recording.
    const auto join_seeds = [](Target& target, Builder& /*recording*/) {
        target.builder.assert_equal(target.messages[0], target.messages[1]);
    };
    expect_refusal(join_seeds, SpliceRefusal::InputCycle);
}

} // namespace
} // namespace bb
