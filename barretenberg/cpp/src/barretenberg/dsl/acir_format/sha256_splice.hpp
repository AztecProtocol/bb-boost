#pragma once

#include "barretenberg/dsl/acir_format/sha256_constraint.hpp"
#include "barretenberg/stdlib_circuit_builders/ultra_circuit_builder.hpp"

#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace acir_format {

/**
 * @brief What one circuit build did with its SHA-256 compression constraints.
 */
struct Sha256RunStats {
    size_t compressions = 0;
    // The chunks the run was cut into; 1 when it was built in place whole.
    size_t chunks = 0;
    size_t launched = 0;
    size_t spliced = 0;
    size_t refused = 0;
    // Chunks the building thread built: chunk 0, chunks not launched and refused chunks.
    size_t built_in_place = 0;
    // Why the first refused chunk was built in place; empty when none was refused.
    std::string first_refusal;
};

/**
 * @brief How the SHA-256 compression constraints of an Ultra circuit are cut into chunks built on scratch threads.
 * @details Splicing a chunk saves the building thread its in-place build, less its launch, splice and scratch builder's
 * destruction. An earlier form splicing into a fresh builder on 64 threads measured 1.7 to 1.9 ms per compression in
 * place and about 1 ms per chunk plus 0.6 ms per compression for those three; on a real circuit, 16 chunks took the
 * building thread's run of 512 compressions from 886 to 537 ms. The chunk count never changes the circuit.
 */
struct Sha256RunOptions {
    // From 8 compressions a chunk's in-place build costs the building thread more than twice what splicing it does.
    static constexpr size_t MIN_COMPRESSIONS_PER_CHUNK = 8;
    // The most chunks measured: on 512 compressions 16 chunks saved about a fifth more than 4 did.
    static constexpr size_t MAX_CHUNKS = 16;

    size_t min_compressions_per_chunk = MIN_COMPRESSIONS_PER_CHUNK;
    size_t max_chunks = MAX_CHUNKS;

    // Observers for tests. They are compiled in production and native callers can set them through ProgramMetadata;
    // every caller in this repository leaves them empty, and nothing serialized exposes them.
    // Runs on the building thread before it builds compression `compression` in place.
    std::function<void(bb::UltraCircuitBuilder& builder, size_t compression)> before_in_place;
    // Runs on a scratch thread when its build of chunk `chunk` has finished; a throw refuses the chunk.
    std::function<void(size_t chunk, bb::UltraCircuitBuilder& scratch)> after_scratch_build;
    // Runs on the scratch thread at each recording of chunk `chunk`, before its cancellation check; `count` from 1.
    std::function<void(size_t chunk, size_t count)> scratch_recording;
    // Runs on the building thread once the scratch builder of chunk `chunk` has been destroyed.
    std::function<void(size_t chunk)> discarded;
    // Receives the counts of the build.
    Sha256RunStats* stats = nullptr;
};

/**
 * @brief Builds the SHA-256 compression constraints of an Ultra circuit in equal contiguous chunks: the building thread
 * builds chunk 0 in place, and each later chunk is built on a scratch builder on a thread of its own and spliced into
 * the real builder at its turn in list order.
 * @details A chunk's scratch build reads only the values of the witnesses its compressions read and write, its seeds,
 * and the splice appends to the real builder what building the chunk there would, so the circuit is the serial build's
 * whatever the chunk count, the thread count or the timing. A chunk that names a variable the builder lacks is not
 * launched; one the splice refuses, or whose scratch build threw or moved a seeded engine's position, is built in place
 * at its turn. Every scratch thread is joined, never detached: at its chunk's turn, or after every running build was
 * cancelled when the build ends early. Each scratch builder's freed memory is returned before it is destroyed.
 */
class Sha256RunSplicer {
  public:
    /**
     * @brief How many chunks a run of `num_compressions` is cut into on `num_threads` threads; under 2, none.
     */
    static size_t num_chunks(size_t num_compressions, size_t num_threads, const Sha256RunOptions& options);

    /**
     * @brief Start a scratch build of every chunk of `run` but the first; null when none starts.
     * @details Call on the building thread after `builder` holds the witness and before any opcode reaches it. The
     * seeds are witnesses, whose values the builder holds from its construction; the splice checks at each chunk's turn
     * that they still have them and that no later constraint joined or tagged them. Starts nothing for a builder
     * without a witness, with fewer than three threads, without native threads, pthreads and exceptions, or under
     * OpenMP.
     */
    static std::unique_ptr<Sha256RunSplicer> launch(const bb::UltraCircuitBuilder& builder,
                                                    const std::vector<Sha256Compression>& run,
                                                    const Sha256RunOptions& options);

    Sha256RunSplicer(const Sha256RunSplicer&) = delete;
    Sha256RunSplicer(Sha256RunSplicer&&) = delete;
    Sha256RunSplicer& operator=(const Sha256RunSplicer&) = delete;
    Sha256RunSplicer& operator=(Sha256RunSplicer&&) = delete;

    /**
     * @brief Cancel every scratch build still running, join every scratch thread and destroy every scratch builder.
     */
    ~Sha256RunSplicer();

    /**
     * @brief Build `run` on `builder` where the serial build of the compressions would start, then log how many chunks
     * were spliced, refused and built in place; `after_in_place(i)` runs once compression i is built in place.
     * @details A `run` other than the one launched is built in place whole once every scratch build is cancelled.
     */
    void build(bb::UltraCircuitBuilder& builder,
               const std::vector<Sha256Compression>& run,
               const std::function<void(size_t)>& after_in_place);

  private:
    struct Chunk;
    using Clock = std::chrono::steady_clock;

    Sha256RunSplicer(const std::vector<Sha256Compression>& run, size_t num_chunks, const Sha256RunOptions& options);
    bool take(Chunk& chunk, bb::UltraCircuitBuilder& builder, bool same_run);
    void refuse(Chunk& chunk, std::string reason);
    void reap();
    std::string summary() const;

    std::vector<Sha256Compression> run_;
    size_t num_chunks_ = 0;
    Sha256RunOptions options_;
    // Indexed by chunk; null for chunk 0 and for chunks not launched.
    std::vector<std::unique_ptr<Chunk>> chunks_;
    Sha256RunStats stats_;
    Clock::time_point launched_at_;
    Clock::time_point reached_at_;
    // When the building thread reached the first launched chunk, and when that chunk's scratch build finished.
    std::optional<size_t> first_taken_;
    Clock::time_point first_needed_at_;
    Clock::time_point first_finished_at_;
};

/**
 * @brief Build the SHA-256 compression constraints `run` of an Ultra circuit in list order: through `splicer` when it
 * is not null, otherwise each in place; `after_in_place(i)` runs once compression i is built in place.
 */
void build_sha256_compressions(bb::UltraCircuitBuilder& builder,
                               const std::vector<Sha256Compression>& run,
                               Sha256RunSplicer* splicer,
                               const Sha256RunOptions& options,
                               const std::function<void(size_t)>& after_in_place);

} // namespace acir_format
