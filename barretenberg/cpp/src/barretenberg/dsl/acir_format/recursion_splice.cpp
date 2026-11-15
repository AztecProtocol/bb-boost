#include "barretenberg/dsl/acir_format/recursion_splice.hpp"

#include "barretenberg/common/assert.hpp"
#include "barretenberg/common/log.hpp"
#include "barretenberg/common/thread.hpp"
#include "barretenberg/common/throw_or_abort.hpp"
#include "barretenberg/dsl/acir_format/honk_recursion_constraint.hpp"
#include "barretenberg/dsl/acir_format/mock_verifier_inputs.hpp"
#include "barretenberg/dsl/acir_format/scratch_build.hpp"
#include "barretenberg/ecc/curves/bn254/g1.hpp"
#include "barretenberg/flavor/ultra_flavor.hpp"
#include "barretenberg/stdlib/primitives/builder_binding.hpp"
#include "barretenberg/stdlib/special_public_inputs/special_public_inputs.hpp"

#include <algorithm>
#include <atomic>
#include <exception>
#include <unordered_map>
#include <utility>

namespace acir_format {

#if BB_SCRATCH_BUILDS_AVAILABLE

namespace {

using scratch_build::message_of;
using scratch_build::milliseconds_between;
using scratch_build::MIN_THREADS;
using scratch_build::refusal_reason;
using scratch_build::return_freed_memory;
using scratch_build::ScratchCancellation;
using scratch_build::ScratchThread;

/**
 * @brief Whether a scratch build may start for `constraint` on a builder of `num_variables` variables.
 * @details Its inputs are read when it starts, before the serial build would reject a malformed constraint, so each
 * must be a variable of the builder.
 */
bool is_eligible(const RecursionConstraint& constraint, const size_t num_variables)
{
    const auto exists = [num_variables](const uint32_t index) { return index < num_variables; };
    return (constraint.proof_type == HONK || constraint.proof_type == ROLLUP_HONK ||
            constraint.proof_type == ROOT_ROLLUP_HONK) &&
           constraint.predicate.is_constant && constraint.predicate.value == bb::fr::one() &&
           std::ranges::all_of(constraint.proof, exists) && std::ranges::all_of(constraint.key, exists) &&
           std::ranges::all_of(constraint.public_inputs, exists) && exists(constraint.key_hash);
}

/**
 * @brief The draws an eligible constraint makes: the replay of the proof construction it skips, then the offset
 * generator of the edge-case multiplication in its KZG batch opening.
 * @details A prediction only: a splice is accepted only when the engine stands where the scratch build started, and
 * the scratch build reports where it ended.
 */
void predicted_draws(const RecursionConstraint& constraint)
{
    const size_t num_public_inputs = constraint.public_inputs.size();
    if (constraint.proof_type == ROLLUP_HONK || constraint.proof_type == ROOT_ROLLUP_HONK) {
        discard_arbitrary_valid_honk_proof_randomness<UltraFlavor, stdlib::recursion::honk::RollupIO>(
            num_public_inputs);
    } else {
        discard_arbitrary_valid_honk_proof_randomness<UltraFlavor,
                                                      stdlib::recursion::honk::DefaultIO<UltraCircuitBuilder>>(
            num_public_inputs);
    }
    [[maybe_unused]] const bb::g1::element offset_generator = bb::g1::element::random_element();
}

} // namespace

struct RecursionSplicer::Scratch {
    size_t index;
    // The constraint as launched; take() splices only for an equal one.
    RecursionConstraint original;
    // seed_indices[i] is the real index of scratch input variable i; `constraint` names scratch inputs instead.
    std::vector<uint32_t> seed_indices;
    std::vector<bb::fr> seed_values;
    RecursionConstraint constraint;
    bb::numeric::RNG::Position start;
    ScratchCancellation cancellation;

    // Written by the scratch thread, read by the building thread after the join or after reading `finished` true.
    std::unique_ptr<UltraCircuitBuilder> builder;
    std::optional<HonkRecursionConstraintOutput<UltraCircuitBuilder>> output;
    bb::numeric::RNG::Position reached;
    std::exception_ptr error;
    Clock::time_point finished_at;
    std::atomic<bool> finished = false;

    bb::numeric::RNG* engine = nullptr;
    ScratchThread thread;

    // Inputs are numbered by distinct real index at first appearance in the order proof, key, public inputs, key
    // hash, so two inputs share a scratch variable exactly when they share a real one.
    Scratch(const UltraCircuitBuilder& real, const RecursionConstraint& launched, const size_t scratch_index)
        : index(scratch_index)
        , original(launched)
        , constraint(launched)
        , cancellation("RecursionSplicer", [scratch_index, num_checks = size_t{ 0 }]() mutable {
            // Only the scratch thread counts.
            if (RecursionSpliceTestHooks::scratch_recording) {
                RecursionSpliceTestHooks::scratch_recording(scratch_index, ++num_checks);
            }
        })
    {
        std::unordered_map<uint32_t, uint32_t> seed_of;
        const auto seed = [&](uint32_t& index) {
            const auto [it, inserted] = seed_of.try_emplace(index, static_cast<uint32_t>(seed_indices.size()));
            if (inserted) {
                seed_indices.push_back(index);
                seed_values.push_back(real.get_variable(index));
            }
            index = it->second;
        };
        for (uint32_t& index : constraint.proof) {
            seed(index);
        }
        for (uint32_t& index : constraint.key) {
            seed(index);
        }
        for (uint32_t& index : constraint.public_inputs) {
            seed(index);
        }
        seed(constraint.key_hash);
    }

    /**
     * @brief Runs build() on a scratch thread of its own; throws when it cannot start.
     */
    void start_thread(bb::numeric::RNG& draw_engine)
    {
        engine = &draw_engine;
        const bool started = thread.start(
            [](void* scratch) -> void* {
                static_cast<Scratch*>(scratch)->build();
                return nullptr;
            },
            this);
        if (!started) {
            throw_or_abort("RecursionSplicer: could not start a scratch thread");
        }
    }

    void join() { thread.join(); }

    // Throws nothing: whatever the build throws is kept in `error`.
    void build() noexcept
    {
        try {
            // parallel_for's core count and pool are per thread, so at one core this thread's loops run inline
            // instead of starting a worker pool of its own.
            bb::set_parallel_for_concurrency(1);
            const std::unique_ptr<bb::numeric::RNG::DrawScope> scope = engine->draw_from(start);
            builder =
                std::make_unique<UltraCircuitBuilder>(SpliceRecording{ .cancellation = &cancellation }, seed_values);
            output = create_ultra_honk_recursion_constraint(*builder, constraint);
            reached = scope->reached();
            if (RecursionSpliceTestHooks::after_scratch_build) {
                RecursionSpliceTestHooks::after_scratch_build(index);
            }
        } catch (...) {
            error = std::current_exception();
        }
        finished_at = Clock::now();
        finished.store(true, std::memory_order_release);
    }

    // Called once the scratch thread is joined. Freed memory is returned first, while the builder still lives.
    void discard()
    {
        if (!builder) {
            return;
        }
        return_freed_memory();
        output.reset();
        builder.reset();
        if (RecursionSpliceTestHooks::discarded) {
            RecursionSpliceTestHooks::discarded(index);
        }
    }
};

RecursionSplicer::RecursionSplicer(bb::numeric::RNG& engine)
    : engine_(engine)
{}

RecursionSplicer::~RecursionSplicer()
{
    reap();
}

void RecursionSplicer::reap()
{
    // Requested of every running build first, so a join waits only until that build's next recording, or its end.
    for (const auto& scratch : scratches_) {
        if (scratch->thread.running()) {
            scratch->cancellation.request();
        }
    }
    for (const auto& scratch : scratches_) {
        scratch->join();
        scratch->discard();
    }
}

void RecursionSplicer::finish()
{
    reap();
    report();
}

std::unique_ptr<RecursionSplicer> RecursionSplicer::launch(const UltraCircuitBuilder& builder,
                                                           const std::vector<RecursionConstraint>& honk_constraints)
{
    const size_t num_variables = builder.get_num_variables();
    if (RecursionSpliceTestHooks::disabled || builder.is_write_vk_mode() || honk_constraints.empty() ||
        !is_eligible(honk_constraints.front(), num_variables)) {
        return nullptr;
    }
    const size_t num_threads = bb::get_num_cpus();
    if (num_threads < MIN_THREADS) {
        return nullptr;
    }
    std::unique_ptr<RecursionSplicer> splicer(new RecursionSplicer(bb::numeric::get_randomness()));
    bb::numeric::RNG& engine = splicer->engine_;
    try {
        // Honk recursion constraints are built first and every later kind is ineligible, so the walk ends in this
        // list. Each start position is the previous one moved by the previous constraint's predicted draws.
        bb::numeric::RNG::Position position = engine.position();
        for (const RecursionConstraint& constraint : honk_constraints) {
            if (splicer->scratches_.size() == num_threads - 1 || !is_eligible(constraint, num_variables)) {
                break;
            }
            auto scratch = std::make_unique<Scratch>(builder, constraint, splicer->scratches_.size());
            scratch->start = position;
            position =
                engine.after(position, [&constraint](bb::numeric::RNG& /*engine*/) { predicted_draws(constraint); });
            splicer->scratches_.push_back(std::move(scratch));
        }
        splicer->launched_at_ = Clock::now();
        for (const auto& scratch : splicer->scratches_) {
            scratch->start_thread(engine);
            ++splicer->stats_.launched;
        }
    } catch (...) {
        // A launch that fails splices nothing; the destructor joins the threads already started.
        return nullptr;
    }
    if (splicer->scratches_.empty()) {
        return nullptr;
    }
    return splicer;
}

void RecursionSplicer::refuse(Scratch& scratch, std::string reason)
{
    if (stats_.refused == 0) {
        stats_.first_refusal = std::move(reason);
    }
    ++stats_.refused;
    scratch.discard();
}

std::optional<HonkRecursionConstraintOutput<UltraCircuitBuilder>> RecursionSplicer::take(
    const size_t index, const RecursionConstraint& constraint, UltraCircuitBuilder& builder)
{
    if (index >= scratches_.size()) {
        return std::nullopt;
    }
    Scratch& scratch = *scratches_[index];
    if (!(constraint == scratch.original)) {
        scratch.cancellation.request();
        scratch.join();
        refuse(scratch, "the constraint changed since the scratch build started");
        return std::nullopt;
    }
    if (index == 0 && !wait_for_first(scratch)) {
        return std::nullopt;
    }
    scratch.join();
    if (index == 0) {
        first_finished_at_ = scratch.finished_at;
    }
    if (scratch.error) {
        const std::string message = message_of(scratch.error);
        vinfo("recursion splice: the scratch build of Honk recursion constraint ", index, " threw: ", message);
        refuse(scratch, "the scratch build threw: " + message);
        return std::nullopt;
    }
    // Before the splice: the scratch build drew what the serial build would draw only from the position it started at.
    if (!engine_.at(scratch.start)) {
        refuse(scratch, "the engine moved since the scratch build started");
        return std::nullopt;
    }
    const SpliceResult result = builder.splice(*scratch.builder, scratch.seed_indices);
    if (!result.spliced()) {
        refuse(scratch, refusal_reason(result.refusal));
        return std::nullopt;
    }
    engine_.adopt(scratch.reached);
    HonkRecursionConstraintOutput<UltraCircuitBuilder> output = std::move(*scratch.output);
    stdlib::rebind_to_builder(output, &builder, result.index_map, result.pairing_tag_offset);
    BB_ASSERT_DEBUG(stdlib::count_contexts_on(output, scratch.builder.get()) == 0,
                    "RecursionSplicer: a spliced output still refers to its scratch builder");
    scratch.discard();
    ++stats_.spliced;
    return output;
}

/**
 * @brief Whether take(0) waits for scratch build 0; when not, the build is cancelled, its thread left to stop at its
 * next recording, and the constraint is built in place.
 */
bool RecursionSplicer::wait_for_first(Scratch& scratch)
{
    first_reached_at_ = Clock::now();
    const bool finished = scratch.finished.load(std::memory_order_acquire);
    if (finished) {
        first_finished_at_ = scratch.finished_at;
    }
    const FirstTake policy = RecursionSpliceTestHooks::first_take;
    if (policy == FirstTake::Wait || (policy == FirstTake::Auto && finished)) {
        return true;
    }
    scratch.cancellation.request();
    ++stats_.built_in_place;
    return false;
}

std::string RecursionSplicer::summary() const
{
    std::string line = std::to_string(stats_.spliced) + " of " + std::to_string(stats_.launched) + " spliced, " +
                       std::to_string(stats_.refused) + " refused, " + std::to_string(stats_.built_in_place) +
                       " built in place";
    std::string detail;
    if (first_reached_at_.has_value()) {
        detail =
            "head start " + std::to_string(milliseconds_between(launched_at_, *first_reached_at_)) + " ms, scratch 0 ";
        if (!first_finished_at_.has_value()) {
            detail += "not finished";
        } else if (RecursionSpliceTestHooks::first_take == FirstTake::Wait) {
            // Negative when the building thread had to wait.
            detail += "margin " + std::to_string(milliseconds_between(*first_finished_at_, *first_reached_at_)) + " ms";
        } else {
            detail += "finished " + std::to_string(milliseconds_between(*first_finished_at_, *first_reached_at_)) +
                      " ms earlier";
        }
    }
    if (stats_.refused > 0) {
        detail += (detail.empty() ? "" : "; ") + std::string("first refusal: ") + stats_.first_refusal;
    }
    return detail.empty() ? line : line + " (" + detail + ")";
}

void RecursionSplicer::report() const
{
    vinfo("recursion splice: ", summary());
    if (RecursionSpliceTestHooks::stats != nullptr) {
        *RecursionSpliceTestHooks::stats = stats_;
    }
}

#else

struct RecursionSplicer::Scratch {};

RecursionSplicer::RecursionSplicer(bb::numeric::RNG& engine)
    : engine_(engine)
{}

RecursionSplicer::~RecursionSplicer() = default;

std::unique_ptr<RecursionSplicer> RecursionSplicer::launch(const UltraCircuitBuilder& /*builder*/,
                                                           const std::vector<RecursionConstraint>& /*honk_constraints*/)
{
    return nullptr;
}

void RecursionSplicer::refuse(Scratch& /*scratch*/, std::string /*reason*/) {}

std::optional<HonkRecursionConstraintOutput<UltraCircuitBuilder>> RecursionSplicer::take(
    const size_t /*index*/, const RecursionConstraint& /*constraint*/, UltraCircuitBuilder& /*builder*/)
{
    return std::nullopt;
}

void RecursionSplicer::finish() {}

void RecursionSplicer::report() const {}

#endif

} // namespace acir_format
