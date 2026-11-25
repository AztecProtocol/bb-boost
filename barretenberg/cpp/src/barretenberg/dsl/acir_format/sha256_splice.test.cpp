#include "barretenberg/dsl/acir_format/sha256_splice.hpp"
#include "barretenberg/common/thread.hpp"
#include "barretenberg/common/throw_or_abort.hpp"
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

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <future>
#include <gtest/gtest.h>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

/**
 * An Ultra circuit whose SHA-256 compression constraints are built in chunks on scratch builders and spliced must equal
 * the serial build of the same program in every builder member, and prove to the same bytes, whatever the chunk count
 * and whichever chunks are built in place instead.
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

TEST_F(Sha256SpliceTest, ChunkCountDoesNotChangeTheBuilder)
{
    const AcirProgram program = make_program(Sha256RunOptions::MAX_CHUNKS);
    const Build serial = build(program, serial_options());
    const size_t most = std::min(Sha256RunOptions::MAX_CHUNKS, bb::get_num_cpus());
    bool any_in_place = false;
    for (const size_t num_chunks : { size_t{ 2 }, size_t{ 3 }, most }) {
        const Build spliced = build(program, { .min_compressions_per_chunk = 1, .max_chunks = num_chunks });
        expect_same_builder(spliced.builder, serial.builder);
        if (built_in_place_whole(spliced.stats)) {
            any_in_place = true;
            continue;
        }
        EXPECT_EQ(spliced.stats.chunks, num_chunks);
        EXPECT_EQ(spliced.stats.spliced, num_chunks - 1) << spliced.stats.first_refusal;
    }
    if (any_in_place) {
        GTEST_SKIP() << "a run started no scratch thread and was built in place";
    }
}

TEST_F(Sha256SpliceTest, RefusedAndThrowingMiddleChunksAreBuiltInPlace)
{
    // The 8-bit range constraint tags a message word of compression 5, in the middle chunk.
    const AcirProgram program = make_program(NUM_COMPRESSIONS, /*small_word=*/5);
    const Build serial = build(program, serial_options());

    const Build refused = build(program, three_chunks());
    expect_same_builder(refused.builder, serial.builder);
    const bool refused_in_place = built_in_place_whole(refused.stats);
    if (!refused_in_place) {
        EXPECT_EQ(refused.stats.spliced, 1);
        EXPECT_EQ(refused.stats.refused, 1);
        EXPECT_EQ(refused.stats.built_in_place, 2);
        EXPECT_EQ(refused.stats.first_refusal, "an input carries a range tag");
    }

    Sha256RunOptions throwing = three_chunks();
    throwing.after_scratch_build = [](const size_t chunk, Builder& /*scratch*/) {
        if (chunk == 1) {
            throw_or_abort("forced scratch build failure");
        }
    };
    const Build threw = build(program, throwing);
    expect_same_builder(threw.builder, serial.builder);
    const bool threw_in_place = built_in_place_whole(threw.stats);
    if (!threw_in_place) {
        EXPECT_EQ(threw.stats.spliced, 1);
        EXPECT_EQ(threw.stats.refused, 1);
        EXPECT_EQ(threw.stats.built_in_place, 2);
        EXPECT_EQ(threw.stats.first_refusal, "the scratch build threw: forced scratch build failure");
    }
    if (refused_in_place || threw_in_place) {
        GTEST_SKIP() << "a run started no scratch thread and was built in place, so nothing was refused";
    }
}

TEST_F(Sha256SpliceTest, ShortRunStartsNoScratchThread)
{
    std::atomic<bool> recorded = false;
    Sha256RunOptions options;
    options.scratch_recording = [&](size_t /*chunk*/, size_t /*count*/) { recorded.store(true); };
    // One compression short of two chunks under the production rule.
    const Build result = build(make_program((2 * Sha256RunOptions::MIN_COMPRESSIONS_PER_CHUNK) - 1), options);
    EXPECT_FALSE(result.builder.failed()) << result.builder.err();
    EXPECT_EQ(result.stats.chunks, 1);
    EXPECT_EQ(result.stats.launched, 0);
    EXPECT_FALSE(recorded.load());
}

/**
 * An error on the building thread while chunks are being built cancels every running scratch build before joining it,
 * and surfaces as the serial build's error. Chunk 2 is held at its first recording until chunk 1 has been joined and
 * discarded, which happens only after every cancellation was requested, so on release it throws at that recording.
 * Chunk 1 is joined first and so cannot be held; whether it stops early depends on timing, and it is not checked.
 */
TEST_F(Sha256SpliceTest, AnErrorOnTheBuildingThreadCancelsTheRunningChunks)
{
    const AcirProgram program = make_program(NUM_COMPRESSIONS);
    // The checks below need both scratch builds running. A thread quota can keep scratch threads from starting, and
    // the run is then built in place with nothing to cancel, which a run without hooks shows first.
    if (built_in_place_whole(build(program, three_chunks()).stats)) {
        GTEST_SKIP() << "no scratch thread started, so there is nothing to cancel";
    }
    std::mutex mutex;
    std::condition_variable recorded;
    size_t num_recording = 0;
    std::array<std::atomic<bool>, 3> finished{};
    std::promise<void> first_discarded;
    const std::shared_future<void> released = first_discarded.get_future().share();
    std::vector<size_t> discarded;

    // Throws before the second compression of chunk 0, once both scratch builds have reached a recording if asked to.
    const auto fail_in_chunk_0 = [&](const bool wait_for_chunks) {
        return [&, wait_for_chunks](Builder& /*builder*/, const size_t compression) {
            if (compression != 1) {
                return;
            }
            if (wait_for_chunks) {
                std::unique_lock<std::mutex> lock(mutex);
                EXPECT_TRUE(recorded.wait_for(lock, std::chrono::minutes(2), [&] { return num_recording == 2; }));
            }
            throw_or_abort("forced building thread failure");
        };
    };
    // Only run where scratch builds start, which needs exceptions; other builds must still compile it.
    const auto error_of = [&](const Sha256RunOptions& options) {
#if BB_SCRATCH_BUILDS_AVAILABLE
        try {
            (void)build(program, options);
        } catch (const std::exception& error) {
            return std::string(error.what());
        }
#else
        (void)build(program, options);
#endif
        return std::string();
    };

    Sha256RunOptions serial = serial_options();
    serial.before_in_place = fail_in_chunk_0(false);
    Sha256RunOptions chunked = three_chunks();
    chunked.before_in_place = fail_in_chunk_0(true);
    chunked.after_scratch_build = [&](const size_t chunk, Builder& /*scratch*/) { finished[chunk].store(true); };
    chunked.scratch_recording = [&](const size_t chunk, const size_t count) {
        if (count != 1) {
            return;
        }
        {
            const std::lock_guard<std::mutex> lock(mutex);
            ++num_recording;
        }
        recorded.notify_all();
        if (chunk == 2) {
            EXPECT_EQ(released.wait_for(std::chrono::minutes(2)), std::future_status::ready);
        }
    };
    chunked.discarded = [&](const size_t chunk) {
        discarded.push_back(chunk);
        if (chunk == 1) {
            first_discarded.set_value();
        }
    };

    const std::string serial_error = error_of(serial);
    EXPECT_EQ(serial_error, "forced building thread failure");
    EXPECT_EQ(error_of(chunked), serial_error);
    EXPECT_FALSE(finished[2].load());
    EXPECT_EQ(discarded, (std::vector<size_t>{ 1, 2 }));
}

} // namespace
