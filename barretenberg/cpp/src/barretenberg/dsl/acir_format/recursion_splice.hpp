#pragma once

#include "barretenberg/dsl/acir_format/recursion_constraint.hpp"
#include "barretenberg/dsl/acir_format/recursion_constraint_output.hpp"
#include "barretenberg/numeric/random/engine.hpp"
#include "barretenberg/stdlib_circuit_builders/ultra_circuit_builder.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace acir_format {

/**
 * @brief What one circuit build did with its scratch builds.
 */
struct RecursionSpliceStats {
    size_t launched = 0;
    size_t spliced = 0;
    size_t refused = 0;
    // Constraints whose scratch build was cancelled because the building thread reached them first.
    size_t built_in_place = 0;
    // Why the first refused constraint was built serially; empty when none was refused.
    std::string first_refusal;
};

/**
 * @brief How RecursionSplicer::take treats scratch build 0.
 */
enum class FirstTake : uint8_t {
    Auto,    // splice it when it has finished, otherwise cancel it and build the constraint in place
    Wait,    // wait for it and splice it
    InPlace, // cancel it and build the constraint in place
};

/**
 * @brief Test-only switches of the recursion splice, all off by default.
 * @details Set them only while no circuit is being built.
 */
struct RecursionSpliceTestHooks {
    using ConstraintHook =
        std::function<void(bb::UltraCircuitBuilder& builder, const RecursionConstraint& constraint, size_t index)>;

    // Builds every recursion constraint serially.
    static inline bool disabled = false;
    // Runs on the building thread just before Honk recursion constraint `index` is spliced or built serially.
    static inline ConstraintHook before_constraint;
    // Runs on the building thread just after Honk recursion constraint `index` was spliced or built serially.
    static inline ConstraintHook after_constraint;
    // Runs on a scratch thread when its build of constraint `index` has finished; a throw refuses that build.
    static inline std::function<void(size_t index)> after_scratch_build;
    // Runs on a scratch thread before each recording of its build of constraint `index`, `count` counting from 1,
    // before the cancellation check.
    static inline std::function<void(size_t index, size_t count)> scratch_recording;
    // Runs on the building thread once the scratch builder of constraint `index` has been destroyed.
    static inline std::function<void(size_t index)> discarded;
    static inline FirstTake first_take = FirstTake::Auto;
    // Receives the counts of each build that started scratch builds.
    static inline RecursionSpliceStats* stats = nullptr;
};

/**
 * @brief Builds the leading eligible Honk recursion constraints of an Ultra circuit on scratch builders on their own
 * threads, and splices each into the real builder when the serial order reaches it.
 * @details A constraint is eligible when it verifies a plain or rollup Honk proof under a constant-true predicate.
 * Whether a constraint is spliced or built serially never changes the circuit: a refused constraint is built serially
 * on the unchanged builder and engine.
 *
 * When the building thread reaches the first constraint before its scratch build has finished, that build is
 * cancelled and the constraint is built in place, so which path builds it depends on timing. A cancelled build only
 * ever had its own builder and its own draws: it never touches the real builder or the real engine, its draws come
 * from its draw scope under an engine with positions and from its thread's own buffer otherwise, and they are dropped
 * as a refused build's are. It stops by throwing at its next recording. Its thread is joined, never detached, before
 * its builder is destroyed, and its freed memory is returned the same way as every other scratch build's.
 */
class RecursionSplicer {
  public:
    /**
     * @brief Start one scratch build per leading eligible constraint of `honk_constraints`, at most one fewer than the
     * worker threads; null when none starts.
     * @details Call on the building thread after `builder` holds the witness and before any opcode reaches it. Starts
     * nothing for a builder without a witness, with fewer than three threads, without native threads, pthreads and
     * exceptions, under OpenMP, or when starting fails. A constraint naming a witness index the builder does not have
     * is not eligible. The engine is get_randomness() and must stay the same object until the last splice.
     */
    static std::unique_ptr<RecursionSplicer> launch(const bb::UltraCircuitBuilder& builder,
                                                    const std::vector<RecursionConstraint>& honk_constraints);

    RecursionSplicer(const RecursionSplicer&) = delete;
    RecursionSplicer(RecursionSplicer&&) = delete;
    RecursionSplicer& operator=(const RecursionSplicer&) = delete;
    RecursionSplicer& operator=(RecursionSplicer&&) = delete;

    /**
     * @brief Join every scratch thread, including those of cancelled builds, and destroy every scratch builder,
     * returning first, on glibc, the memory freed into malloc arenas to the operating system.
     */
    ~RecursionSplicer();

    /**
     * @brief Splice the scratch build of Honk recursion constraint `index`, which is `constraint`, into `builder` and
     * return its output bound to `builder`; nullopt when the constraint must be built serially.
     * @details Call on the building thread exactly where the serial build of that constraint would start, once per
     * index in increasing order. On nullopt, `builder` and the engine are unchanged. A `constraint` other than the one
     * launched at `index` is refused once its scratch build is cancelled and joined. Otherwise index 0 is spliced only
     * if its build has finished, and is else cancelled and nullopt returned at once; a later index waits for its build.
     */
    std::optional<HonkRecursionConstraintOutput<bb::UltraCircuitBuilder>> take(size_t index,
                                                                               const RecursionConstraint& constraint,
                                                                               bb::UltraCircuitBuilder& builder);

    /**
     * @brief Log how many constraints were spliced, refused and built in place, why the first refusal happened, and
     * how far scratch build 0 was ahead of the building thread.
     */
    void report() const;

  private:
    struct Scratch;
    using Clock = std::chrono::steady_clock;

    explicit RecursionSplicer(bb::numeric::RNG& engine);
    void refuse(Scratch& scratch, std::string reason);
    bool wait_for_first(Scratch& scratch);
    std::string summary() const;

    // Unread in builds without native threads, where nothing is ever launched.
    [[maybe_unused]] bb::numeric::RNG& engine_;
    std::vector<std::unique_ptr<Scratch>> scratches_;
    RecursionSpliceStats stats_;
    Clock::time_point launched_at_;
    std::optional<Clock::time_point> first_reached_at_;
    // When scratch build 0 finished, if take(0) saw it finish.
    std::optional<Clock::time_point> first_finished_at_;
};

} // namespace acir_format
