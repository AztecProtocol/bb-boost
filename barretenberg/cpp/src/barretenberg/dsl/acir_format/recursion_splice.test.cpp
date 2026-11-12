#include "barretenberg/dsl/acir_format/recursion_splice.hpp"
#include "barretenberg/common/thread.hpp"
#include "barretenberg/common/throw_or_abort.hpp"
#include "barretenberg/dsl/acir_format/acir_format.hpp"
#include "barretenberg/dsl/acir_format/test_class.hpp"
#include "barretenberg/dsl/acir_format/utils.hpp"
#include "barretenberg/dsl/acir_format/witness_constant.hpp"
#include "barretenberg/flavor/ultra_flavor.hpp"
#include "barretenberg/flavor/ultra_zk_flavor.hpp"
#include "barretenberg/numeric/random/engine.hpp"
#include "barretenberg/numeric/random/entropy_buffer_observer.hpp"
#include "barretenberg/srs/global_crs.hpp"
#include "barretenberg/stdlib/special_public_inputs/special_public_inputs.hpp"
#include "barretenberg/stdlib_circuit_builders/mock_circuits.hpp"
#include "barretenberg/stdlib_circuit_builders/ultra_circuit_builder_test_utils.hpp"
#include "barretenberg/ultra_honk/prover_instance.hpp"
#include "barretenberg/ultra_honk/ultra_prover.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <gtest/gtest.h>
#include <map>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

/**
 * A circuit whose Honk recursion constraints are built on scratch builders and spliced must equal the serial build of
 * the same program in every builder member, and a refused splice must leave the serial result. With the seeded engine
 * the variable values and the engine position must match too; with the default engine the two builds draw different
 * offset generators, so values are not compared.
 */

using namespace acir_format;
using namespace bb;

namespace {

using Builder = UltraCircuitBuilder;
using bb::test_utils::CompareValues;
using bb::test_utils::expect_same_builder;

constexpr size_t NUM_INNER_PUBLIC_INPUTS = 2;
// The largest range a small range constraint accepts.
constexpr uint64_t MAX_SMALL_RANGE = (1ULL << 16) - 1;

// A recursion constraint with a constant-true predicate and the witness vector its indices point into.
struct InnerProof {
    RecursionConstraint constraint;
    WitnessVector witness;
};

template <typename Flavor, typename IO> InnerProof prove_inner_circuit(const uint32_t proof_type)
{
    Builder builder;
    MockCircuits::add_arithmetic_gates(builder);
    for (size_t i = 0; i < NUM_INNER_PUBLIC_INPUTS; ++i) {
        builder.add_public_variable(fr::random_element());
    }
    IO::add_default(builder);
    auto prover_instance = std::make_shared<ProverInstance_<Flavor>>(builder);
    auto verification_key = std::make_shared<typename Flavor::VerificationKey>(prover_instance->get_precomputed());
    UltraProver_<Flavor> prover(prover_instance, verification_key);
    const HonkProof proof = prover.construct_proof();

    InnerProof inner;
    inner.constraint = recursion_data_to_recursion_constraint(inner.witness,
                                                              proof,
                                                              verification_key->to_field_elements(),
                                                              verification_key->hash(),
                                                              fr::one(),
                                                              builder.num_public_inputs() - IO::PUBLIC_INPUTS_SIZE,
                                                              proof_type);
    // The predicate witness is appended last; a constant predicate leaves it unused.
    BB_ASSERT_EQ(static_cast<size_t>(inner.constraint.predicate.index), inner.witness.size() - 1);
    inner.witness.pop_back();
    inner.constraint.predicate = WitnessOrConstant<fr>::from_constant(fr::one());
    return inner;
}

// One program verifying `proofs` in order, each constraint's indices moved past the witnesses before it.
AcirProgram make_program(const std::vector<const InnerProof*>& proofs)
{
    std::vector<RecursionConstraint> constraints;
    WitnessVector witness;
    for (const InnerProof* inner : proofs) {
        RecursionConstraint constraint = inner->constraint;
        const auto offset = static_cast<uint32_t>(witness.size());
        for (std::vector<uint32_t>* indices : { &constraint.key, &constraint.proof, &constraint.public_inputs }) {
            for (uint32_t& index : *indices) {
                index += offset;
            }
        }
        constraint.key_hash += offset;
        witness.insert(witness.end(), inner->witness.begin(), inner->witness.end());
        constraints.push_back(std::move(constraint));
    }
    return AcirProgram{ .constraints = constraint_to_acir_format(constraints), .witness = witness };
}

std::vector<uint32_t> inputs_of(const RecursionConstraint& constraint)
{
    std::vector<uint32_t> inputs = constraint.proof;
    inputs.insert(inputs.end(), constraint.key.begin(), constraint.key.end());
    inputs.insert(inputs.end(), constraint.public_inputs.begin(), constraint.public_inputs.end());
    inputs.push_back(constraint.key_hash);
    return inputs;
}

// Clears every hook when a test ends, whatever its outcome.
struct HookReset {
    HookReset() = default;
    HookReset(const HookReset&) = delete;
    HookReset(HookReset&&) = delete;
    HookReset& operator=(const HookReset&) = delete;
    HookReset& operator=(HookReset&&) = delete;
    ~HookReset()
    {
        RecursionSpliceTestHooks::disabled = false;
        RecursionSpliceTestHooks::before_constraint = nullptr;
        RecursionSpliceTestHooks::after_constraint = nullptr;
        RecursionSpliceTestHooks::after_scratch_build = nullptr;
        RecursionSpliceTestHooks::stats = nullptr;
    }
};

struct Build {
    Builder builder;
    RecursionSpliceStats stats;
    // The engine position after each Honk recursion constraint, then after the whole build.
    std::vector<numeric::RNG::Position> positions;
};

/**
 * @brief Build `program` with or without splicing, from the engine position `start` when the engine has positions.
 * @details `before_constraint` runs in both modes; `after_scratch_build` only matters when splicing.
 */
Build build(const AcirProgram& program,
            const ProgramMetadata& metadata,
            const bool splice,
            const numeric::RNG::Position& start,
            const RecursionSpliceTestHooks::ConstraintHook& before_constraint = {},
            const std::function<void(size_t)>& after_scratch_build = {})
{
    numeric::RNG& engine = numeric::get_randomness();
    if (!start.empty()) {
        engine.adopt(start);
    }
    Build result;
    RecursionSpliceTestHooks::disabled = !splice;
    RecursionSpliceTestHooks::before_constraint = before_constraint;
    RecursionSpliceTestHooks::after_constraint = [&](Builder& /*builder*/, const RecursionConstraint&, size_t) {
        result.positions.push_back(engine.position());
    };
    RecursionSpliceTestHooks::after_scratch_build = after_scratch_build;
    RecursionSpliceTestHooks::stats = &result.stats;
    AcirProgram copy = program;
    result.builder = create_circuit<Builder>(copy, metadata);
    result.positions.push_back(engine.position());
    return result;
}

class RecursionSpliceTest : public ::testing::Test {
  protected:
    static void SetUpTestSuite()
    {
        bb::srs::init_file_crs_factory(bb::srs::bb_crs_path());
        using DefaultIO = stdlib::recursion::honk::DefaultIO<Builder>;
        for (size_t i = 0; i < 3; ++i) {
            honk_proofs.push_back(prove_inner_circuit<UltraFlavor, DefaultIO>(HONK));
        }
        zk_proof = prove_inner_circuit<UltraZKFlavor, DefaultIO>(HONK_ZK);
        rollup_proof = prove_inner_circuit<UltraFlavor, stdlib::recursion::honk::RollupIO>(ROLLUP_HONK);
    }

    void SetUp() override
    {
        // Three Honk constraints need three scratch threads beside the building thread.
        if (bb::get_num_cpus() < 4) {
            GTEST_SKIP() << "needs at least four threads";
        }
        start = numeric::get_randomness().position();
    }

    bool seeded() const { return !start.empty(); }
    CompareValues compare_values() const { return seeded() ? CompareValues::Yes : CompareValues::No; }

    // Builds `program` serially and spliced from the same engine position and checks they agree.
    std::pair<Build, Build> build_both(const AcirProgram& program,
                                       const ProgramMetadata& metadata = {},
                                       const RecursionSpliceTestHooks::ConstraintHook& before_constraint = {},
                                       const std::function<void(size_t)>& after_scratch_build = {})
    {
        const HookReset reset;
        Build serial = build(program, metadata, /*splice=*/false, start, before_constraint);
        Build spliced = build(program, metadata, /*splice=*/true, start, before_constraint, after_scratch_build);
        expect_same_builder(spliced.builder, serial.builder, compare_values());
        EXPECT_TRUE(spliced.positions == serial.positions);
        EXPECT_EQ(serial.stats.launched, 0);
        return { std::move(serial), std::move(spliced) };
    }

    void expect_spliced(const std::vector<const InnerProof*>& proofs,
                        const size_t expected_splices,
                        const ProgramMetadata& metadata = {})
    {
        const auto [serial, spliced] = build_both(make_program(proofs), metadata);
        EXPECT_EQ(spliced.stats.spliced, expected_splices);
        EXPECT_EQ(spliced.stats.refused, 0) << spliced.stats.first_refusal;
    }

    std::vector<const InnerProof*> three_honk_proofs() const
    {
        return { &honk_proofs[0], &honk_proofs[1], &honk_proofs[2] };
    }

    static inline std::vector<InnerProof> honk_proofs;
    static inline std::optional<InnerProof> zk_proof;
    static inline std::optional<InnerProof> rollup_proof;
    numeric::RNG::Position start;
};

TEST_F(RecursionSpliceTest, OneConstraint)
{
    expect_spliced({ &honk_proofs[0] }, 1);
}

TEST_F(RecursionSpliceTest, TwoConstraints)
{
    expect_spliced({ &honk_proofs[0], &honk_proofs[1] }, 2);
}

TEST_F(RecursionSpliceTest, ThreeConstraints)
{
    expect_spliced(three_honk_proofs(), 3);
}

TEST_F(RecursionSpliceTest, RollupConstraint)
{
    expect_spliced({ &*rollup_proof }, 1, ProgramMetadata{ .has_ipa_claim = true });
}

TEST_F(RecursionSpliceTest, EligibleAfterIneligibleIsBuiltSerially)
{
    expect_spliced({ &*zk_proof, &honk_proofs[0] }, 0);
}

// The forced refusals hit the middle of three constraints; the first is spliced either way.

TEST_F(RecursionSpliceTest, RefusesWhenTheEngineMoved)
{
    const auto [serial, spliced] =
        build_both(make_program(three_honk_proofs()), {}, [](Builder&, const RecursionConstraint&, size_t index) {
            if (index == 1) {
                [[maybe_unused]] const uint64_t extra_draw = numeric::get_randomness().get_random_uint64();
            }
        });
    if (seeded()) {
        // Every later start position was predicted without the extra draw.
        EXPECT_EQ(spliced.stats.spliced, 1);
        EXPECT_EQ(spliced.stats.refused, 2);
        EXPECT_EQ(spliced.stats.first_refusal, "the engine moved since the scratch build started");
    } else {
        // An engine without an order has no position to miss.
        EXPECT_EQ(spliced.stats.spliced, 3);
        EXPECT_EQ(spliced.stats.refused, 0);
    }
}

TEST_F(RecursionSpliceTest, RefusesATaggedInput)
{
    const auto [serial, spliced] =
        build_both(make_program(three_honk_proofs()),
                   {},
                   [](Builder& builder, const RecursionConstraint& constraint, size_t index) {
                       if (index != 1) {
                           return;
                       }
                       const std::vector<uint32_t> inputs = inputs_of(constraint);
                       const auto small = std::ranges::find_if(inputs, [&](uint32_t input) {
                           return uint256_t(builder.get_variable(input)) <= MAX_SMALL_RANGE;
                       });
                       ASSERT_NE(small, inputs.end());
                       builder.create_small_range_constraint(*small, MAX_SMALL_RANGE);
                   });
    EXPECT_EQ(spliced.stats.spliced, 2);
    EXPECT_EQ(spliced.stats.refused, 1);
    EXPECT_EQ(spliced.stats.first_refusal, "an input carries a range tag");
}

TEST_F(RecursionSpliceTest, RefusesInputsSharingACycle)
{
    const auto [serial, spliced] =
        build_both(make_program(three_honk_proofs()),
                   {},
                   [](Builder& builder, const RecursionConstraint& constraint, size_t index) {
                       if (index != 1) {
                           return;
                       }
                       std::unordered_map<fr, uint32_t> first_with_value;
                       for (const uint32_t input : inputs_of(constraint)) {
                           const auto [it, inserted] = first_with_value.try_emplace(builder.get_variable(input), input);
                           if (!inserted && it->second != input) {
                               builder.assert_equal(it->second, input);
                               return;
                           }
                       }
                       FAIL() << "no two inputs share a value";
                   });
    EXPECT_EQ(spliced.stats.spliced, 2);
    EXPECT_EQ(spliced.stats.refused, 1);
    EXPECT_EQ(spliced.stats.first_refusal, "inputs share a copy cycle");
}

TEST_F(RecursionSpliceTest, RefusesAScratchBuildThatThrew)
{
    const auto [serial, spliced] = build_both(make_program(three_honk_proofs()), {}, {}, [](size_t index) {
        if (index == 1) {
            throw_or_abort("forced scratch build failure");
        }
    });
    EXPECT_EQ(spliced.stats.spliced, 2);
    EXPECT_EQ(spliced.stats.refused, 1);
    EXPECT_EQ(spliced.stats.first_refusal, "the scratch build threw");
}

/**
 * @brief Records, per thread, each refill of the default engine's entropy buffer and each range a draw consumes.
 */
class BufferLog final : public numeric::EntropyBufferObserver {
  public:
    struct Fill {
        std::thread::id thread;
        size_t size;
    };
    struct Consume {
        std::thread::id thread;
        uint64_t fill;
        size_t offset;
        size_t length;
    };

    BufferLog() = default;
    BufferLog(const BufferLog&) = delete;
    BufferLog(BufferLog&&) = delete;
    BufferLog& operator=(const BufferLog&) = delete;
    BufferLog& operator=(BufferLog&&) = delete;
    // Draws after a failed test must not reach a destroyed log.
    ~BufferLog() override { numeric::set_entropy_buffer_observer(nullptr); }

    void on_fill(const uint64_t fill, const size_t size) override
    {
        const std::lock_guard<std::mutex> lock(mutex);
        fills.emplace(fill, Fill{ std::this_thread::get_id(), size });
    }

    void on_consume(const uint64_t fill, const size_t offset, const size_t length) override
    {
        const std::lock_guard<std::mutex> lock(mutex);
        consumes.push_back({ std::this_thread::get_id(), fill, offset, length });
    }

    std::mutex mutex;
    std::unordered_map<uint64_t, Fill> fills;
    std::vector<Consume> consumes;
};

TEST_F(RecursionSpliceTest, ScratchThreadsDrawOnlyFromTheirOwnBuffers)
{
    if (seeded()) {
        GTEST_SKIP() << "the entropy buffer belongs to the default engine";
    }
    const HookReset reset;
    constexpr size_t REFUSED = 1;
    std::mutex mutex;
    std::map<size_t, std::thread::id> scratch_threads;
    BufferLog log;
    numeric::set_entropy_buffer_observer(&log);
    const Build spliced = build(make_program(three_honk_proofs()), {}, /*splice=*/true, start, {}, [&](size_t index) {
        {
            const std::lock_guard<std::mutex> lock(mutex);
            scratch_threads.emplace(index, std::this_thread::get_id());
        }
        if (index == REFUSED) {
            throw_or_abort("forced scratch build failure");
        }
    });
    numeric::set_entropy_buffer_observer(nullptr);

    EXPECT_EQ(spliced.stats.spliced, 2);
    EXPECT_EQ(spliced.stats.refused, 1);
    ASSERT_EQ(scratch_threads.size(), 3);
    std::unordered_set<std::thread::id> scratch_ids;
    for (const auto& [index, thread] : scratch_threads) {
        scratch_ids.insert(thread);
    }

    // A scratch thread starts after the observer, so each of its fills is seen and belongs to it.
    size_t refused_consumes = 0;
    for (const BufferLog::Consume& consume : log.consumes) {
        if (consume.fill == 0) {
            EXPECT_FALSE(scratch_ids.contains(consume.thread));
            continue;
        }
        const auto fill = log.fills.find(consume.fill);
        ASSERT_NE(fill, log.fills.end());
        EXPECT_EQ(fill->second.thread, consume.thread);
        EXPECT_LE(consume.offset + consume.length, fill->second.size);
        if (fill->second.thread == scratch_threads.at(REFUSED)) {
            ++refused_consumes;
        }
    }
    // The refused build drew: its draws are consumed by nothing else, because no other thread reads its fills.
    EXPECT_GT(refused_consumes, 0);

    // No two consumed ranges of one buffer overlap; an unobserved fill is told apart by its thread.
    std::map<std::pair<uint64_t, std::thread::id>, std::vector<std::pair<size_t, size_t>>> ranges;
    for (const BufferLog::Consume& consume : log.consumes) {
        const std::thread::id owner = consume.fill == 0 ? consume.thread : std::thread::id{};
        ranges[{ consume.fill, owner }].emplace_back(consume.offset, consume.length);
    }
    for (auto& [buffer, consumed] : ranges) {
        std::ranges::sort(consumed);
        for (size_t i = 1; i < consumed.size(); ++i) {
            EXPECT_LE(consumed[i - 1].first + consumed[i - 1].second, consumed[i].first);
        }
    }
}

} // namespace
