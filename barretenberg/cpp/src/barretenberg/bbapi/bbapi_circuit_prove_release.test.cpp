#include "barretenberg/bbapi/bbapi_prove_instance.hpp"
#include "barretenberg/special_public_inputs/special_public_inputs.hpp"
#include "barretenberg/srs/global_crs.hpp"
#include "barretenberg/stdlib_circuit_builders/plookup_tables/plookup_tables.hpp"
#include "barretenberg/stdlib_circuit_builders/ultra_circuit_builder.hpp"
#include "barretenberg/ultra_honk/prover_instance.hpp"
#include "barretenberg/ultra_honk/ultra_prover.hpp"
#include "barretenberg/ultra_honk/ultra_verifier.hpp"

#include <gtest/gtest.h>

namespace bb::bbapi {
namespace {

using Flavor = UltraFlavor;
using Builder = UltraCircuitBuilder;
using ProverInstance = ProverInstance_<Flavor>;
using VerificationKey = Flavor::VerificationKey;

/**
 * @brief An Ultra circuit with arithmetic, public-input, lookup, range and ROM/RAM rows, built without randomness so
 * that two builds are identical.
 */
Builder build_circuit()
{
    Builder builder;
    for (uint64_t i = 0; i < 16; ++i) {
        const uint32_t a_idx = builder.add_public_variable(fr(i + 1));
        const uint32_t b_idx = builder.add_variable(fr((2 * i) + 3));
        const uint32_t c_idx = builder.add_variable(fr(7));
        const uint32_t d_idx = builder.add_variable(fr(i + 1) + fr((2 * i) + 3) + fr(7));
        builder.create_big_add_gate({ a_idx, b_idx, c_idx, d_idx, fr(1), fr(1), fr(1), fr(-1), fr(0) });
    }

    constexpr auto UINT32_XOR = plookup::MultiTableId::UINT32_XOR;
    for (uint32_t i = 0; i < 8; ++i) {
        const fr left = fr{ 0x9e3779b9U * (i + 1), 0, 0, 0 }.to_montgomery_form();
        const fr right = fr{ 0x85ebca6bU ^ i, 0, 0, 0 }.to_montgomery_form();
        const uint32_t left_idx = builder.add_variable(left);
        const uint32_t right_idx = builder.add_variable(right);
        const auto accumulators = plookup::get_lookup_accumulators(UINT32_XOR, left, right, /*is_2_to_1_lookup=*/true);
        builder.create_gates_from_plookup_accumulators(UINT32_XOR, accumulators, left_idx, right_idx);
    }

    for (uint64_t i = 0; i < 16; ++i) {
        builder.create_dyadic_range_constraint(builder.add_variable(fr(i * 1000)), 16, "range");
    }

    const size_t rom_id = builder.create_ROM_array(4);
    for (size_t i = 0; i < 4; ++i) {
        builder.set_ROM_element(rom_id, i, builder.add_variable(fr((i * i) + 1)));
    }
    builder.read_ROM_array(rom_id, builder.add_variable(fr(2)));

    const size_t ram_id = builder.create_RAM_array(2);
    builder.init_RAM_element(ram_id, 0, builder.add_variable(fr(5)));
    builder.init_RAM_element(ram_id, 1, builder.add_variable(fr(9)));
    builder.write_RAM_array(ram_id, builder.add_variable(fr(1)), builder.add_variable(fr(11)));
    builder.read_RAM_array(ram_id, builder.add_variable(fr(1)));

    DefaultIO::add_default(builder);
    return builder;
}

class CircuitProveCopiedCircuitData : public ::testing::Test {
  protected:
    static void SetUpTestSuite() { srs::init_file_crs_factory(srs::bb_crs_path()); }
};

/**
 * @brief The instance CircuitProve builds frees the circuit's blocks and lookup tables, keeps their sizes and offsets,
 * and proves the same bytes as an instance that keeps them.
 */
TEST_F(CircuitProveCopiedCircuitData, FreesTheCircuitAndProvesTheSameBytes)
{
    Builder kept = build_circuit();
    Builder released = build_circuit();
    auto kept_instance = std::make_shared<ProverInstance>(kept, CopiedCircuitData::KEEP);
    auto released_instance = circuit_prove_instance<Flavor>(released);

    EXPECT_FALSE(kept.get_lookup_tables().empty());
    EXPECT_TRUE(released.get_lookup_tables().empty());
    auto kept_blocks = kept.blocks.get();
    auto released_blocks = released.blocks.get();
    size_t num_freed_blocks = 0;
    for (size_t i = 0; i < kept_blocks.size(); ++i) {
        EXPECT_FALSE(kept_blocks[i].data_freed_);
        EXPECT_EQ(released_blocks[i].size(), kept_blocks[i].size());
        EXPECT_EQ(released_blocks[i].trace_offset(), kept_blocks[i].trace_offset());
        // A block without rows has no chunk in the copying pass, so nothing frees it.
        EXPECT_EQ(released_blocks[i].data_freed_, kept_blocks[i].size() > 0);
        if (released_blocks[i].data_freed_) {
            ++num_freed_blocks;
        }
    }
    // Arithmetic, public inputs, lookup, delta range and memory at least.
    EXPECT_GE(num_freed_blocks, 5U);

    auto vk = std::make_shared<VerificationKey>(kept_instance->get_precomputed());
    EXPECT_EQ(VerificationKey(released_instance->get_precomputed()), *vk);
    UltraProver_<Flavor> kept_prover(kept_instance, vk);
    UltraProver_<Flavor> released_prover(released_instance, vk);
    const auto kept_proof = kept_prover.construct_proof();
    const auto released_proof = released_prover.construct_proof();
    EXPECT_EQ(released_proof, kept_proof);

    UltraVerifier_<Flavor, DefaultIO> verifier(std::make_shared<Flavor::VKAndHash>(vk));
    EXPECT_TRUE(verifier.verify_proof(released_proof).result);
}

} // namespace
} // namespace bb::bbapi
