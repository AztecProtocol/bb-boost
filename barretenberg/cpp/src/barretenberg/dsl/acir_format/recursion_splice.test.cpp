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
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
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
#if defined(__GLIBC__)
#include <pthread.h>
#endif

/**
 * A circuit whose Honk recursion constraints are built on scratch builders and spliced must equal the serial build of
 * the same program in every builder member, and a refused splice or a first constraint built in place must leave the
 * serial result. With the seeded engine the variable values and the engine position must match too; with the default
 * engine the two builds draw different offset generators, so values are not compared.
 */

using namespace acir_format;
using namespace bb;

namespace {

using Builder = UltraCircuitBuilder;
using bb::test_utils::CompareValues;
using bb::test_utils::expect_same_builder;
using bb::test_utils::expect_same_builder_state;

// Mirrors the condition under which production starts scratch builds.
#if !defined(NO_MULTITHREADING) && !defined(__wasm__) && !defined(_WIN32) && !defined(OMP_MULTITHREADING) &&           \
    defined(__cpp_exceptions) && !defined(BB_NO_EXCEPTIONS)
constexpr bool SCRATCH_BUILDS_AVAILABLE = true;
#else
constexpr bool SCRATCH_BUILDS_AVAILABLE = false;
#endif

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

// Clears every hook and restores the first-take policy when a test ends, whatever its outcome.
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
        RecursionSpliceTestHooks::scratch_recording = nullptr;
        RecursionSpliceTestHooks::discarded = nullptr;
        RecursionSpliceTestHooks::stats = nullptr;
        RecursionSpliceTestHooks::first_take = first_take;
    }

    const FirstTake first_take = RecursionSpliceTestHooks::first_take;
};

struct Build {
    Builder builder;
    RecursionSpliceStats stats;
    // The engine position after each Honk recursion constraint, then after the whole build.
    std::vector<numeric::RNG::Position> positions;
};

// The hooks of one build. Unless a test says otherwise the first constraint waits for its scratch build, so that a
// spliced first constraint is covered whatever the timing.
struct Hooks {
    // Runs in serial and spliced builds.
    RecursionSpliceTestHooks::ConstraintHook before_constraint;
    // Run in spliced builds only.
    RecursionSpliceTestHooks::ConstraintHook spliced_before_constraint;
    RecursionSpliceTestHooks::ConstraintHook spliced_after_constraint;
    std::function<void(size_t)> after_scratch_build;
    std::function<void(size_t, size_t)> scratch_recording;
    std::function<void(size_t)> discarded;
    FirstTake first_take = FirstTake::Wait;
};

/**
 * @brief Build `program` with or without splicing, from the engine position `start` when the engine has positions.
 */
Build build(const AcirProgram& program,
            const ProgramMetadata& metadata,
            const bool splice,
            const numeric::RNG::Position& start,
            const Hooks& hooks = {})
{
    numeric::RNG& engine = numeric::get_randomness();
    if (!start.empty()) {
        engine.adopt(start);
    }
    Build result;
    RecursionSpliceTestHooks::disabled = !splice;
    RecursionSpliceTestHooks::before_constraint =
        [&](Builder& builder, const RecursionConstraint& constraint, const size_t index) {
            if (hooks.before_constraint) {
                hooks.before_constraint(builder, constraint, index);
            }
            if (splice && hooks.spliced_before_constraint) {
                hooks.spliced_before_constraint(builder, constraint, index);
            }
        };
    RecursionSpliceTestHooks::after_constraint =
        [&](Builder& builder, const RecursionConstraint& constraint, const size_t index) {
            result.positions.push_back(engine.position());
            if (splice && hooks.spliced_after_constraint) {
                hooks.spliced_after_constraint(builder, constraint, index);
            }
        };
    RecursionSpliceTestHooks::after_scratch_build = hooks.after_scratch_build;
    RecursionSpliceTestHooks::scratch_recording = hooks.scratch_recording;
    RecursionSpliceTestHooks::discarded = hooks.discarded;
    RecursionSpliceTestHooks::first_take = hooks.first_take;
    RecursionSpliceTestHooks::stats = &result.stats;
    AcirProgram copy = program;
    result.builder = create_circuit<Builder>(copy, metadata);
    result.positions.push_back(engine.position());
    return result;
}

// Lets one thread wait, for a bounded time, until another sets it.
class Signal {
  public:
    void set()
    {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            set_ = true;
        }
        condition_.notify_all();
    }

    [[nodiscard]] bool wait()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return condition_.wait_for(lock, std::chrono::minutes(2), [this] { return set_; });
    }

  private:
    std::mutex mutex_;
    std::condition_variable condition_;
    bool set_ = false;
};

// Owned by a thread through a thread_local, so its destructor runs when that thread exits.
struct ExitMarker {
    ExitMarker() = default;
    ExitMarker(const ExitMarker&) = delete;
    ExitMarker(ExitMarker&&) = delete;
    ExitMarker& operator=(const ExitMarker&) = delete;
    ExitMarker& operator=(ExitMarker&&) = delete;
    ~ExitMarker()
    {
        if (exited != nullptr) {
            exited->store(true);
        }
    }

    std::atomic<bool>* exited = nullptr;
};

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
        if (!SCRATCH_BUILDS_AVAILABLE) {
            GTEST_SKIP() << "this build starts no scratch builds";
        }
        // Three Honk constraints need three scratch threads beside the building thread.
        if (bb::get_num_cpus() < 4) {
            GTEST_SKIP() << "needs at least four threads";
        }
        start = numeric::get_randomness().position();
    }

    bool seeded() const { return !start.empty(); }
    CompareValues compare_values() const { return seeded() ? CompareValues::Yes : CompareValues::No; }

    // The builder, and with the seeded engine the engine position after each constraint, match the serial build.
    void expect_same_build(const Build& spliced, const Build& serial) const
    {
        expect_same_builder(spliced.builder, serial.builder, compare_values());
        EXPECT_TRUE(spliced.positions == serial.positions);
        EXPECT_EQ(serial.stats.launched, 0);
    }

    // Builds `program` serially and spliced from the same engine position and checks they agree.
    std::pair<Build, Build> build_both(const AcirProgram& program,
                                       const ProgramMetadata& metadata = {},
                                       const Hooks& hooks = {})
    {
        const HookReset reset;
        Build serial = build(program, metadata, /*splice=*/false, start, hooks);
        Build spliced = build(program, metadata, /*splice=*/true, start, hooks);
        expect_same_build(spliced, serial);
        return { std::move(serial), std::move(spliced) };
    }

    void expect_spliced(const std::vector<const InnerProof*>& proofs,
                        const size_t expected_splices,
                        const ProgramMetadata& metadata = {})
    {
        const auto [serial, spliced] = build_both(make_program(proofs), metadata);
        EXPECT_EQ(spliced.stats.spliced, expected_splices);
        EXPECT_EQ(spliced.stats.refused, 0) << spliced.stats.first_refusal;
        EXPECT_EQ(spliced.stats.built_in_place, 0);
    }

    // The first constraint is built in place and every later one spliced: the cancelled build moved neither the real
    // builder nor the engine, so each later scratch build still starts where the engine stands.
    void expect_first_in_place(const std::vector<const InnerProof*>& proofs)
    {
        const auto [serial, spliced] = build_both(make_program(proofs), {}, Hooks{ .first_take = FirstTake::InPlace });
        EXPECT_EQ(spliced.stats.launched, proofs.size());
        EXPECT_EQ(spliced.stats.built_in_place, 1);
        EXPECT_EQ(spliced.stats.spliced, proofs.size() - 1);
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

TEST_F(RecursionSpliceTest, ScratchThreadsGetAMainThreadSizedStack)
{
#if !defined(__GLIBC__)
    GTEST_SKIP() << "reads the stack size with pthread_getattr_np";
#else
    std::atomic<size_t> stack_bytes = 0;
    const auto [serial, spliced] =
        build_both(make_program({ &honk_proofs[0] }), {}, Hooks{ .after_scratch_build = [&](size_t /*index*/) {
                       pthread_attr_t attributes;
                       size_t size = 0;
                       if (pthread_getattr_np(pthread_self(), &attributes) == 0) {
                           pthread_attr_getstacksize(&attributes, &size);
                           pthread_attr_destroy(&attributes);
                       }
                       stack_bytes.store(size);
                   } });
    EXPECT_EQ(spliced.stats.spliced, 1);
    EXPECT_GE(stack_bytes.load(), size_t{ 8 } << 20);
#endif
}

// A scratch build reads its inputs when it starts, so a constraint naming a variable the builder lacks gets none.
TEST_F(RecursionSpliceTest, ConstraintWithAnInputOutOfRangeIsNotLaunched)
{
    const HookReset reset;
    const InnerProof& inner = honk_proofs[0];
    const Builder builder(inner.witness, {}, /*is_write_vk_mode=*/false);
    const auto out_of_range = static_cast<uint32_t>(builder.get_num_variables());
    EXPECT_NE(RecursionSplicer::launch(builder, { inner.constraint }), nullptr);
    const std::vector<std::function<void(RecursionConstraint&)>> breaks{
        [&](RecursionConstraint& constraint) { constraint.proof.back() = out_of_range; },
        [&](RecursionConstraint& constraint) { constraint.key.back() = out_of_range; },
        [&](RecursionConstraint& constraint) { constraint.public_inputs.back() = out_of_range; },
        [&](RecursionConstraint& constraint) { constraint.key_hash = out_of_range; },
    };
    for (size_t i = 0; i < breaks.size(); ++i) {
        RecursionConstraint constraint = inner.constraint;
        breaks[i](constraint);
        EXPECT_EQ(RecursionSplicer::launch(builder, { constraint }), nullptr) << i;
    }
}

TEST_F(RecursionSpliceTest, RefusesAConstraintOtherThanTheLaunchedOne)
{
    const HookReset reset;
    RecursionSpliceTestHooks::first_take = FirstTake::Wait;
    RecursionSpliceStats stats;
    RecursionSpliceTestHooks::stats = &stats;
    const InnerProof& inner = honk_proofs[0];
    Builder builder(inner.witness, {}, /*is_write_vk_mode=*/false);
    const std::unique_ptr<RecursionSplicer> splicer = RecursionSplicer::launch(builder, { inner.constraint });
    ASSERT_NE(splicer, nullptr);
    RecursionConstraint other = inner.constraint;
    std::swap(other.proof[0], other.proof[1]);
    const Builder before = builder;
    EXPECT_FALSE(splicer->take(0, other, builder).has_value());
    expect_same_builder_state(builder, before);
    splicer->finish();
    EXPECT_EQ(stats.spliced, 0);
    EXPECT_EQ(stats.refused, 1);
    EXPECT_EQ(stats.first_refusal, "the constraint changed since the scratch build started");
}

// A later constraint other than the launched one is refused too, once its scratch build is cancelled and joined.
TEST_F(RecursionSpliceTest, RefusesALaterConstraintOtherThanTheLaunchedOne)
{
    const HookReset reset;
    RecursionSpliceTestHooks::first_take = FirstTake::Wait;
    RecursionSpliceStats stats;
    RecursionSpliceTestHooks::stats = &stats;
    const AcirProgram program = make_program({ &honk_proofs[0], &honk_proofs[1] });
    const std::vector<RecursionConstraint>& constraints = program.constraints.honk_recursion_constraints;
    Builder builder(program.witness, {}, /*is_write_vk_mode=*/false);
    const std::unique_ptr<RecursionSplicer> splicer = RecursionSplicer::launch(builder, constraints);
    ASSERT_NE(splicer, nullptr);
    EXPECT_TRUE(splicer->take(0, constraints[0], builder).has_value());
    RecursionConstraint other = constraints[1];
    std::swap(other.proof[0], other.proof[1]);
    const Builder before = builder;
    EXPECT_FALSE(splicer->take(1, other, builder).has_value());
    expect_same_builder_state(builder, before);
    splicer->finish();
    EXPECT_EQ(stats.spliced, 1);
    EXPECT_EQ(stats.refused, 1);
    EXPECT_EQ(stats.first_refusal, "the constraint changed since the scratch build started");
}

TEST_F(RecursionSpliceTest, LoneConstraintBuiltInPlace)
{
    expect_first_in_place({ &honk_proofs[0] });
}

TEST_F(RecursionSpliceTest, TwoConstraintsFirstBuiltInPlace)
{
    expect_first_in_place({ &honk_proofs[0], &honk_proofs[1] });
}

TEST_F(RecursionSpliceTest, ThreeConstraintsFirstBuiltInPlace)
{
    expect_first_in_place(three_honk_proofs());
}

// The production policy: whichever way timing decides the first constraint, the builder is the serial one.
TEST_F(RecursionSpliceTest, FirstTakeByTiming)
{
    const auto [serial, spliced] =
        build_both(make_program(three_honk_proofs()), {}, Hooks{ .first_take = FirstTake::Auto });
    EXPECT_EQ(spliced.stats.spliced + spliced.stats.built_in_place, 3);
    EXPECT_LE(spliced.stats.built_in_place, 1);
    EXPECT_EQ(spliced.stats.refused, 0) << spliced.stats.first_refusal;
}

/**
 * The first scratch build is cancelled while the building thread builds the constraint in place, at four points: held
 * at its first, a middle and its last recording, then released, and after it finished. It stops at that very
 * recording, its thread has exited by the time its builder is discarded, which happens before the Honk recursion
 * constraints are reported and so before any later kind of constraint is built, and the real builder and engine end as
 * in the serial build.
 */
TEST_F(RecursionSpliceTest, CancelsTheFirstScratchBuildAtFourPoints)
{
    const HookReset reset;
    const AcirProgram program = make_program({ &honk_proofs[0] });
    std::atomic<size_t> num_recordings = 0;
    {
        const Build counted =
            build(program, {}, /*splice=*/true, start, Hooks{ .scratch_recording = [&](size_t index, size_t count) {
                      if (index == 0) {
                          num_recordings.store(count);
                      }
                  } });
        ASSERT_EQ(counted.stats.spliced, 1);
    }
    const size_t total = num_recordings.load();
    ASSERT_GT(total, 2);
    const Build serial = build(program, {}, /*splice=*/false, start);

    // The first, a middle and the last recording, then past the end: the build has finished when cancelled.
    for (const size_t cancel_at : { size_t{ 1 }, total / 2, total, total + 1 }) {
        SCOPED_TRACE(cancel_at);
        const bool after_finish = cancel_at > total;
        Signal held;
        Signal released;
        std::atomic<size_t> last_recording = 0;
        std::atomic<bool> finished = false;
        std::atomic<bool> exited = false;
        std::vector<size_t> discarded;
        bool exited_before_discard = false;
        bool reported_before_discard = true;
        const Hooks hooks{
            .spliced_before_constraint =
                [&](Builder&, const RecursionConstraint&, size_t index) {
                    if (index == 0) {
                        EXPECT_TRUE(held.wait());
                    }
                },
            .spliced_after_constraint =
                [&](Builder&, const RecursionConstraint&, size_t index) {
                    if (index == 0) {
                        released.set();
                    }
                },
            .after_scratch_build =
                [&](size_t index) {
                    if (index == 0) {
                        finished.store(true);
                        if (after_finish) {
                            held.set();
                        }
                    }
                },
            .scratch_recording =
                [&](size_t index, size_t count) {
                    if (index != 0) {
                        return;
                    }
                    thread_local ExitMarker marker;
                    marker.exited = &exited;
                    last_recording.store(count);
                    if (count == cancel_at) {
                        held.set();
                        EXPECT_TRUE(released.wait());
                    }
                },
            .discarded =
                [&](size_t index) {
                    discarded.push_back(index);
                    if (index == 0) {
                        exited_before_discard = exited.load();
                        reported_before_discard = RecursionSpliceTestHooks::stats->launched != 0;
                    }
                },
            .first_take = FirstTake::InPlace,
        };
        const Build spliced = build(program, {}, /*splice=*/true, start, hooks);
        expect_same_build(spliced, serial);
        EXPECT_EQ(spliced.stats.built_in_place, 1);
        EXPECT_EQ(spliced.stats.spliced, 0);
        EXPECT_EQ(spliced.stats.refused, 0) << spliced.stats.first_refusal;
        EXPECT_EQ(last_recording.load(), std::min(cancel_at, total));
        EXPECT_EQ(finished.load(), after_finish);
        EXPECT_EQ(discarded, std::vector<size_t>{ 0 });
        EXPECT_TRUE(exited_before_discard);
        EXPECT_FALSE(reported_before_discard);
        EXPECT_EQ(spliced.stats.launched, 1);
    }
}

/**
 * An error on the building thread cancels every scratch build it has not taken before joining it. Builds 1 and 2 are
 * held at their first recording until build 0 has been joined and discarded, which the splicer reaches only after
 * requesting every cancellation, so on release each throws at that recording. Were a build joined before its
 * cancellation was requested, builds 1 and 2 would run on to their end once released. Build 0 is joined first and so
 * cannot be held; whether it stops early or finishes depends on timing, and it is not checked.
 */
TEST_F(RecursionSpliceTest, AnErrorOnTheBuildingThreadCancelsTheScratchBuilds)
{
    const HookReset reset;
    std::mutex mutex;
    std::condition_variable recorded;
    size_t num_recording = 0;
    std::array<std::atomic<bool>, 3> finished{};
    Signal first_discarded;
    std::vector<size_t> discarded;
    const Hooks hooks{
        .spliced_before_constraint =
            [&](Builder&, const RecursionConstraint&, size_t index) {
                if (index != 0) {
                    return;
                }
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    EXPECT_TRUE(recorded.wait_for(lock, std::chrono::minutes(2), [&] { return num_recording == 3; }));
                }
                throw_or_abort("forced building thread failure");
            },
        .after_scratch_build = [&](size_t index) { finished[index].store(true); },
        .scratch_recording =
            [&](size_t index, size_t count) {
                if (count != 1) {
                    return;
                }
                {
                    const std::lock_guard<std::mutex> lock(mutex);
                    ++num_recording;
                }
                recorded.notify_all();
                if (index != 0) {
                    EXPECT_TRUE(first_discarded.wait());
                }
            },
        .discarded =
            [&](size_t index) {
                discarded.push_back(index);
                if (index == 0) {
                    first_discarded.set();
                }
            },
    };
    EXPECT_THROW((void)build(make_program(three_honk_proofs()), {}, /*splice=*/true, start, hooks), std::runtime_error);
    EXPECT_FALSE(finished[1].load());
    EXPECT_FALSE(finished[2].load());
    EXPECT_EQ(discarded, (std::vector<size_t>{ 0, 1, 2 }));
}

// The forced refusals hit the middle of three constraints; the first is spliced either way.

TEST_F(RecursionSpliceTest, RefusesWhenTheEngineMoved)
{
    const auto [serial, spliced] =
        build_both(make_program(three_honk_proofs()),
                   {},
                   Hooks{ .before_constraint = [](Builder&, const RecursionConstraint&, size_t index) {
                       if (index == 1) {
                           [[maybe_unused]] const uint64_t extra_draw = numeric::get_randomness().get_random_uint64();
                       }
                   } });
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
    const auto [serial, spliced] = build_both(
        make_program(three_honk_proofs()),
        {},
        Hooks{ .before_constraint = [](Builder& builder, const RecursionConstraint& constraint, size_t index) {
            if (index != 1) {
                return;
            }
            const std::vector<uint32_t> inputs = inputs_of(constraint);
            const auto small = std::ranges::find_if(
                inputs, [&](uint32_t input) { return uint256_t(builder.get_variable(input)) <= MAX_SMALL_RANGE; });
            ASSERT_NE(small, inputs.end());
            builder.create_small_range_constraint(*small, MAX_SMALL_RANGE);
        } });
    EXPECT_EQ(spliced.stats.spliced, 2);
    EXPECT_EQ(spliced.stats.refused, 1);
    EXPECT_EQ(spliced.stats.first_refusal, "an input carries a range tag");
}

TEST_F(RecursionSpliceTest, RefusesInputsSharingACycle)
{
    const auto [serial, spliced] = build_both(
        make_program(three_honk_proofs()),
        {},
        Hooks{ .before_constraint = [](Builder& builder, const RecursionConstraint& constraint, size_t index) {
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
        } });
    EXPECT_EQ(spliced.stats.spliced, 2);
    EXPECT_EQ(spliced.stats.refused, 1);
    EXPECT_EQ(spliced.stats.first_refusal, "inputs share a copy cycle");
}

TEST_F(RecursionSpliceTest, RefusesAScratchBuildThatThrew)
{
    const auto [serial, spliced] =
        build_both(make_program(three_honk_proofs()), {}, Hooks{ .after_scratch_build = [](size_t index) {
                       if (index == 1) {
                           throw_or_abort("forced scratch build failure");
                       }
                   } });
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
    // The first build is cancelled at its first recording, which follows its draws, while held there.
    constexpr size_t CANCELLED = 0;
    constexpr size_t REFUSED = 1;
    std::mutex mutex;
    std::map<size_t, std::thread::id> scratch_threads;
    Signal held;
    Signal released;
    BufferLog log;
    numeric::set_entropy_buffer_observer(&log);
    const Hooks hooks{
        .spliced_before_constraint =
            [&](Builder&, const RecursionConstraint&, size_t index) {
                if (index == CANCELLED) {
                    EXPECT_TRUE(held.wait());
                }
            },
        .spliced_after_constraint =
            [&](Builder&, const RecursionConstraint&, size_t index) {
                if (index == CANCELLED) {
                    released.set();
                }
            },
        .after_scratch_build =
            [](size_t index) {
                if (index == REFUSED) {
                    throw_or_abort("forced scratch build failure");
                }
            },
        .scratch_recording =
            [&](size_t index, size_t count) {
                if (count != 1) {
                    return;
                }
                {
                    const std::lock_guard<std::mutex> lock(mutex);
                    scratch_threads.emplace(index, std::this_thread::get_id());
                }
                if (index == CANCELLED) {
                    held.set();
                    EXPECT_TRUE(released.wait());
                }
            },
        .first_take = FirstTake::InPlace,
    };
    const Build spliced = build(make_program(three_honk_proofs()), {}, /*splice=*/true, start, hooks);
    numeric::set_entropy_buffer_observer(nullptr);

    EXPECT_EQ(spliced.stats.spliced, 1);
    EXPECT_EQ(spliced.stats.refused, 1);
    EXPECT_EQ(spliced.stats.built_in_place, 1);
    ASSERT_EQ(scratch_threads.size(), 3);
    std::unordered_set<std::thread::id> scratch_ids;
    for (const auto& [index, thread] : scratch_threads) {
        scratch_ids.insert(thread);
    }

    // A scratch thread starts after the observer, so each of its fills is seen and belongs to it.
    size_t refused_consumes = 0;
    size_t cancelled_consumes = 0;
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
        if (fill->second.thread == scratch_threads.at(CANCELLED)) {
            ++cancelled_consumes;
        }
    }
    // The refused and the cancelled build drew: their draws are consumed by nothing else, because no other thread
    // reads their fills.
    EXPECT_GT(refused_consumes, 0);
    EXPECT_GT(cancelled_consumes, 0);

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
