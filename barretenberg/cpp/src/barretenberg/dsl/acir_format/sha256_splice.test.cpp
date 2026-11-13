#include "barretenberg/dsl/acir_format/sha256_splice.hpp"
#include "barretenberg/common/thread.hpp"
#include "barretenberg/crypto/poseidon2/poseidon2_params.hpp"
#include "barretenberg/crypto/poseidon2/poseidon2_permutation.hpp"
#include "barretenberg/crypto/sha256/sha256.hpp"
#include "barretenberg/dsl/acir_format/acir_format.hpp"
#include "barretenberg/dsl/acir_format/scratch_build.hpp"
#include "barretenberg/dsl/acir_format/test_class.hpp"
#include "barretenberg/dsl/acir_format/witness_constant.hpp"
#include "barretenberg/flavor/ultra_flavor.hpp"
#include "barretenberg/srs/global_crs.hpp"
#include "barretenberg/stdlib_circuit_builders/ultra_circuit_builder_test_utils.hpp"
#include "barretenberg/ultra_honk/prover_instance.hpp"
#include "barretenberg/ultra_honk/ultra_prover.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <vector>

/**
 * An Ultra circuit whose SHA-256 compression constraints are built in chunks on scratch builders and spliced must equal
 * the serial build of the same program in every builder member, and prove to the same bytes.
 */

using namespace acir_format;
using namespace bb;

namespace {

using Builder = UltraCircuitBuilder;
using bb::test_utils::expect_same_builder;
using Poseidon2 = crypto::Poseidon2Permutation<crypto::Poseidon2Bn254ScalarFieldParams>;

constexpr size_t NUM_COMPRESSIONS = 12;

Sha256RunOptions serial_options()
{
    return { .max_chunks = 1 };
}

// Three chunks of four compressions, on three threads or more.
Sha256RunOptions three_chunks()
{
    return { .min_compressions_per_chunk = NUM_COMPRESSIONS / 3, .max_chunks = 3 };
}

fr word(const uint32_t value)
{
    return fr(static_cast<uint64_t>(value));
}

/**
 * @brief A 32-bit XOR, then `num_compressions` SHA-256 compressions each taking the previous one's output as its state,
 * the first message word of each range-constrained to 32 bits, then a Poseidon2 permutation of the last output's first
 * four words. With `small_word`, the second message word of that compression is below 256 and range-constrained to 8
 * bits, which tags it.
 */
AcirProgram make_program(const size_t num_compressions, const std::optional<size_t> small_word = std::nullopt)
{
    WitnessVector witness;
    std::vector<Acir::Opcode> opcodes;
    const auto add = [&](const fr& value) {
        witness.push_back(value);
        return static_cast<uint32_t>(witness.size() - 1);
    };
    const auto append = [&](const auto& constraint) {
        const std::vector<Acir::Opcode> converted = constraint_to_acir_opcode(constraint);
        opcodes.insert(opcodes.end(), converted.begin(), converted.end());
    };
    const auto witness_of = [](const uint32_t index) { return WitnessOrConstant<fr>::from_index(index); };

    constexpr uint32_t LEFT = 0x12345678;
    constexpr uint32_t RIGHT = 0x0f0f0f0f;
    append(LogicConstraint{ .a = witness_of(add(word(LEFT))),
                            .b = witness_of(add(word(RIGHT))),
                            .result = add(word(LEFT ^ RIGHT)),
                            .num_bits = 32,
                            .is_xor_gate = true });

    std::array<uint32_t, 8> state = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
    std::array<uint32_t, 8> state_witnesses{};
    for (size_t j = 0; j < 8; ++j) {
        state_witnesses[j] = add(word(state[j]));
    }
    for (size_t i = 0; i < num_compressions; ++i) {
        std::array<uint32_t, 16> block{};
        for (size_t j = 0; j < 16; ++j) {
            block[j] = 0x9e3779b9U * static_cast<uint32_t>((16 * i) + j + 1);
        }
        if (small_word == i) {
            block[1] = 200;
        }
        Sha256Compression compression;
        for (size_t j = 0; j < 16; ++j) {
            compression.inputs[j] = witness_of(add(word(block[j])));
        }
        for (size_t j = 0; j < 8; ++j) {
            compression.hash_values[j] = witness_of(state_witnesses[j]);
        }
        state = crypto::sha256_block(state, block);
        for (size_t j = 0; j < 8; ++j) {
            compression.result[j] = add(word(state[j]));
        }
        state_witnesses = compression.result;
        append(RangeConstraint{ .witness = compression.inputs[0].index, .num_bits = 32 });
        if (small_word == i) {
            append(RangeConstraint{ .witness = compression.inputs[1].index, .num_bits = 8 });
        }
        append(compression);
    }

    Poseidon2::State input{};
    Poseidon2Constraint permutation;
    for (size_t j = 0; j < input.size(); ++j) {
        input[j] = word(state[j]);
        permutation.state.push_back(witness_of(state_witnesses[j]));
    }
    for (const fr& output : Poseidon2::permutation(input)) {
        permutation.result.push_back(add(output));
    }
    append(permutation);
    return AcirProgram{ .constraints = circuit_serde_to_acir_format(build_acir_circuit(opcodes), /*is_mega=*/false),
                        .witness = witness };
}

struct Build {
    Builder builder;
    Sha256RunStats stats;
};

Build build(const AcirProgram& program, Sha256RunOptions options)
{
    Build result;
    options.stats = &result.stats;
    AcirProgram copy = program;
    result.builder = create_circuit<Builder>(copy, ProgramMetadata{ .sha256_run = options });
    return result;
}

HonkProof prove(Builder builder)
{
    auto prover_instance = std::make_shared<ProverInstance_<UltraFlavor>>(builder);
    auto verification_key = std::make_shared<UltraFlavor::VerificationKey>(prover_instance->get_precomputed());
    UltraProver_<UltraFlavor> prover(prover_instance, verification_key);
    return prover.construct_proof();
}

/**
 * @brief Whether the run was built in place whole because no scratch build started, as when a thread quota makes a
 * thread fail to start. Counts that need the chunks do not hold then, but the circuit must still be the serial one.
 */
bool built_in_place_whole(const Sha256RunStats& stats)
{
    if (stats.launched != 0) {
        return false;
    }
    EXPECT_EQ(stats.chunks, 1U);
    EXPECT_EQ(stats.built_in_place, 1U);
    return true;
}

class Sha256SpliceTest : public ::testing::Test {
  protected:
    static void SetUpTestSuite() { bb::srs::init_file_crs_factory(bb::srs::bb_crs_path()); }

    void SetUp() override
    {
        if (!BB_SCRATCH_BUILDS_AVAILABLE) {
            GTEST_SKIP() << "this build starts no scratch builds";
        }
        if (bb::get_num_cpus() < 3) {
            GTEST_SKIP() << "three chunks need three threads";
        }
    }
};

TEST_F(Sha256SpliceTest, ChunksMatchTheSerialBuildAndItsProof)
{
    const AcirProgram program = make_program(NUM_COMPRESSIONS);
    const Build serial = build(program, serial_options());
    const Build spliced = build(program, three_chunks());
    EXPECT_FALSE(serial.builder.failed()) << serial.builder.err();
    EXPECT_EQ(serial.stats.launched, 0);
    expect_same_builder(spliced.builder, serial.builder);
    EXPECT_EQ(prove(spliced.builder), prove(serial.builder));
    if (built_in_place_whole(spliced.stats)) {
        GTEST_SKIP() << "no scratch thread started, so the run was built in place";
    }
    EXPECT_EQ(spliced.stats.chunks, 3);
    EXPECT_EQ(spliced.stats.spliced, 2);
    EXPECT_EQ(spliced.stats.refused, 0) << spliced.stats.first_refusal;
}

} // namespace
