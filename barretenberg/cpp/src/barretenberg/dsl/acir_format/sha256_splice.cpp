#include "barretenberg/dsl/acir_format/sha256_splice.hpp"

#include "barretenberg/common/log.hpp"
#include "barretenberg/common/thread.hpp"
#include "barretenberg/dsl/acir_format/scratch_build.hpp"
#include "barretenberg/numeric/random/engine.hpp"

#include <algorithm>
#include <exception>
#include <span>
#include <unordered_map>
#include <utility>

namespace acir_format {

using bb::UltraCircuitBuilder;

namespace {

void build_in_place(UltraCircuitBuilder& builder,
                    const std::vector<Sha256Compression>& run,
                    const size_t begin,
                    const size_t end,
                    const Sha256RunOptions& options,
                    const std::function<void(size_t)>& after_in_place)
{
    for (size_t i = begin; i < end; ++i) {
        if (options.before_in_place) {
            options.before_in_place(builder, i);
        }
        create_sha256_compression_constraints(builder, run[i]);
        after_in_place(i);
    }
}

} // namespace

size_t Sha256RunSplicer::num_chunks(const size_t num_compressions,
                                    const size_t num_threads,
                                    const Sha256RunOptions& options)
{
    const size_t by_size = num_compressions / std::max<size_t>(options.min_compressions_per_chunk, 1);
    return std::min({ by_size, options.max_chunks, num_threads });
}

void build_sha256_compressions(UltraCircuitBuilder& builder,
                               const std::vector<Sha256Compression>& run,
                               Sha256RunSplicer* splicer,
                               const Sha256RunOptions& options,
                               const std::function<void(size_t)>& after_in_place)
{
    if (splicer != nullptr) {
        splicer->build(builder, run, after_in_place);
        return;
    }
    build_in_place(builder, run, 0, run.size(), options, after_in_place);
    if (options.stats != nullptr) {
        *options.stats = Sha256RunStats{ .compressions = run.size(), .chunks = 1, .built_in_place = 1 };
    }
}

#if BB_SCRATCH_BUILDS_AVAILABLE

namespace {

using scratch_build::message_of;
using scratch_build::milliseconds_between;
using scratch_build::MIN_THREADS;
using scratch_build::refusal_reason;
using scratch_build::return_freed_memory;
using scratch_build::ScratchCancellation;
using scratch_build::ScratchThread;

// Chunk `chunk` of `num_chunks` equal contiguous chunks of `num_compressions` starts here.
size_t chunk_begin(const size_t chunk, const size_t num_chunks, const size_t num_compressions)
{
    return chunk * num_compressions / num_chunks;
}

// Visits every witness index a compression names: inputs, hash values, outputs.
template <typename Compression, typename Visit> void for_each_witness(Compression& compression, const Visit& visit)
{
    for (auto& input : compression.inputs) {
        if (!input.is_constant) {
            visit(input.index);
        }
    }
    for (auto& input : compression.hash_values) {
        if (!input.is_constant) {
            visit(input.index);
        }
    }
    for (auto& output : compression.result) {
        visit(output);
    }
}

bool names_only_variables_below(const std::span<const Sha256Compression> compressions, const size_t num_variables)
{
    bool all_below = true;
    for (const Sha256Compression& compression : compressions) {
        for_each_witness(compression, [&](const uint32_t index) { all_below = all_below && index < num_variables; });
    }
    return all_below;
}

} // namespace

struct Sha256RunSplicer::Chunk {
    size_t index;
    // seed_indices[i] is the real index of scratch input variable i; `compressions` name scratch inputs instead.
    std::vector<uint32_t> seed_indices;
    std::vector<bb::fr> seed_values;
    std::vector<Sha256Compression> compressions;
    const Sha256RunOptions* options;
    bb::numeric::RNG::Position start;
    ScratchCancellation cancellation;

    // Written by the scratch thread, read by the building thread after the join.
    std::unique_ptr<UltraCircuitBuilder> builder;
    bb::numeric::RNG::Position reached;
    std::exception_ptr error;
    Clock::time_point finished_at;

    bb::numeric::RNG* engine = nullptr;
    ScratchThread thread;

    // Inputs are numbered by distinct real index at first appearance, so two inputs share a scratch variable exactly
    // when they share a real one.
    Chunk(const UltraCircuitBuilder& real,
          const std::span<const Sha256Compression> launched,
          const size_t chunk_index,
          const Sha256RunOptions& run_options)
        : index(chunk_index)
        , compressions(launched.begin(), launched.end())
        , options(&run_options)
        , cancellation("Sha256RunSplicer", [&run_options, chunk_index, num_checks = size_t{ 0 }]() mutable {
            // Only the scratch thread counts.
            if (run_options.scratch_recording) {
                run_options.scratch_recording(chunk_index, ++num_checks);
            }
        })
    {
        std::unordered_map<uint32_t, uint32_t> seed_of;
        for (Sha256Compression& compression : compressions) {
            for_each_witness(compression, [&](uint32_t& variable) {
                const auto [it, inserted] = seed_of.try_emplace(variable, static_cast<uint32_t>(seed_indices.size()));
                if (inserted) {
                    seed_indices.push_back(variable);
                    seed_values.push_back(real.get_variable(variable));
                }
                variable = it->second;
            });
        }
    }

    static void* run_build(void* chunk)
    {
        static_cast<Chunk*>(chunk)->build();
        return nullptr;
    }

    // Throws nothing: whatever the build throws is kept in `error`.
    void build() noexcept
    {
        try {
            // parallel_for's core count and pool are per thread, so at one core this thread's loops run inline
            // instead of starting a worker pool of its own.
            bb::set_parallel_for_concurrency(1);
            const std::unique_ptr<bb::numeric::RNG::DrawScope> scope = engine->draw_from(start);
            builder = std::make_unique<UltraCircuitBuilder>(bb::SpliceRecording{ .cancellation = &cancellation },
                                                            seed_values);
            for (const Sha256Compression& compression : compressions) {
                create_sha256_compression_constraints(*builder, compression);
            }
            reached = scope->reached();
            if (options->after_scratch_build) {
                options->after_scratch_build(index, *builder);
            }
        } catch (...) {
            error = std::current_exception();
        }
        finished_at = Clock::now();
    }

    // Called once the scratch thread is joined. Freed memory is returned first, while the builder still lives.
    void discard()
    {
        if (!builder) {
            return;
        }
        return_freed_memory();
        builder.reset();
        if (options->discarded) {
            options->discarded(index);
        }
    }
};

Sha256RunSplicer::Sha256RunSplicer(const std::vector<Sha256Compression>& run,
                                   const size_t num_chunks,
                                   const Sha256RunOptions& options)
    : run_(run)
    , num_chunks_(num_chunks)
    , options_(options)
    , chunks_(num_chunks)
{
    stats_.compressions = run.size();
    stats_.chunks = num_chunks;
}

Sha256RunSplicer::~Sha256RunSplicer()
{
    reap();
}

void Sha256RunSplicer::reap()
{
    // Requested of every running build first, so a join waits only until that build's next recording, or its end.
    for (const auto& chunk : chunks_) {
        if (chunk != nullptr && chunk->thread.running()) {
            chunk->cancellation.request();
        }
    }
    for (const auto& chunk : chunks_) {
        if (chunk != nullptr) {
            chunk->thread.join();
            chunk->discard();
        }
    }
}

std::unique_ptr<Sha256RunSplicer> Sha256RunSplicer::launch(const UltraCircuitBuilder& builder,
                                                           const std::vector<Sha256Compression>& run,
                                                           const Sha256RunOptions& options)
{
    const size_t num_threads = bb::get_num_cpus();
    if (builder.is_write_vk_mode() || num_threads < MIN_THREADS) {
        return nullptr;
    }
    const size_t count = num_chunks(run.size(), num_threads, options);
    if (count < 2) {
        return nullptr;
    }
    std::unique_ptr<Sha256RunSplicer> splicer(new Sha256RunSplicer(run, count, options));
    bb::numeric::RNG& engine = bb::numeric::get_randomness();
    const bb::numeric::RNG::Position start = engine.position();
    const std::span<const Sha256Compression> compressions(run);
    const size_t num_variables = builder.get_num_variables();
    try {
        for (size_t c = 1; c < count; ++c) {
            const size_t begin = chunk_begin(c, count, run.size());
            const auto chunk = compressions.subspan(begin, chunk_begin(c + 1, count, run.size()) - begin);
            // Seeds are read here, before the serial build would reject a compression naming a missing variable.
            if (names_only_variables_below(chunk, num_variables)) {
                splicer->chunks_[c] = std::make_unique<Chunk>(builder, chunk, c, splicer->options_);
                splicer->chunks_[c]->start = start;
                splicer->chunks_[c]->engine = &engine;
            }
        }
        splicer->launched_at_ = Clock::now();
        for (const auto& chunk : splicer->chunks_) {
            if (chunk == nullptr) {
                continue;
            }
            // A launch that fails splices nothing; the destructor joins the threads already started.
            if (!chunk->thread.start(&Chunk::run_build, chunk.get())) {
                return nullptr;
            }
            ++splicer->stats_.launched;
        }
    } catch (...) {
        return nullptr;
    }
    if (splicer->stats_.launched == 0) {
        return nullptr;
    }
    return splicer;
}

void Sha256RunSplicer::build(UltraCircuitBuilder& builder,
                             const std::vector<Sha256Compression>& run,
                             const std::function<void(size_t)>& after_in_place)
{
    reached_at_ = Clock::now();
    const bool same_run = run == run_;
    for (size_t c = 0; c < num_chunks_; ++c) {
        if (chunks_[c] == nullptr || !take(*chunks_[c], builder, same_run)) {
            build_in_place(builder,
                           run,
                           chunk_begin(c, num_chunks_, run.size()),
                           chunk_begin(c + 1, num_chunks_, run.size()),
                           options_,
                           after_in_place);
            ++stats_.built_in_place;
        }
    }
    vinfo("sha256 splice: ", summary());
    if (options_.stats != nullptr) {
        *options_.stats = stats_;
    }
}

bool Sha256RunSplicer::take(Chunk& chunk, UltraCircuitBuilder& builder, const bool same_run)
{
    if (!same_run) {
        chunk.cancellation.request();
        chunk.thread.join();
        refuse(chunk, "the compressions changed since the scratch builds started");
        return false;
    }
    const Clock::time_point needed_at = Clock::now();
    chunk.thread.join();
    if (!first_taken_.has_value()) {
        first_taken_ = chunk.index;
        first_needed_at_ = needed_at;
        first_finished_at_ = chunk.finished_at;
    }
    if (chunk.error) {
        const std::string message = message_of(chunk.error);
        vinfo("sha256 splice: the scratch build of chunk ", chunk.index, " threw: ", message);
        refuse(chunk, "the scratch build threw: " + message);
        return false;
    }
    // The serial build would draw from wherever the engine then stands, which the scratch build could not know.
    if (!(chunk.reached == chunk.start)) {
        refuse(chunk, "the scratch build drew randomness");
        return false;
    }
    const bb::SpliceResult result = builder.splice(*chunk.builder, chunk.seed_indices);
    if (!result.spliced()) {
        refuse(chunk, refusal_reason(result.refusal));
        return false;
    }
    chunk.discard();
    ++stats_.spliced;
    return true;
}

void Sha256RunSplicer::refuse(Chunk& chunk, std::string reason)
{
    if (stats_.refused == 0) {
        stats_.first_refusal = std::move(reason);
    }
    ++stats_.refused;
    chunk.discard();
}

std::string Sha256RunSplicer::summary() const
{
    std::string line = std::to_string(stats_.compressions) + " compressions in " + std::to_string(stats_.chunks) +
                       " chunks, " + std::to_string(stats_.spliced) + " of " + std::to_string(stats_.launched) +
                       " spliced, " + std::to_string(stats_.refused) + " refused, " +
                       std::to_string(stats_.built_in_place) + " built in place (head start " +
                       std::to_string(milliseconds_between(launched_at_, reached_at_)) + " ms";
    if (first_taken_.has_value()) {
        // Negative when the building thread had to wait.
        line += ", chunk " + std::to_string(*first_taken_) + " margin " +
                std::to_string(milliseconds_between(first_finished_at_, first_needed_at_)) + " ms";
    }
    if (stats_.refused > 0) {
        line += "; first refusal: " + stats_.first_refusal;
    }
    return line + ")";
}

#else

struct Sha256RunSplicer::Chunk {};

Sha256RunSplicer::~Sha256RunSplicer() = default;

std::unique_ptr<Sha256RunSplicer> Sha256RunSplicer::launch(const UltraCircuitBuilder& /*builder*/,
                                                           const std::vector<Sha256Compression>& /*run*/,
                                                           const Sha256RunOptions& /*options*/)
{
    return nullptr;
}

void Sha256RunSplicer::build(UltraCircuitBuilder& builder,
                             const std::vector<Sha256Compression>& run,
                             const std::function<void(size_t)>& after_in_place)
{
    build_in_place(builder, run, 0, run.size(), options_, after_in_place);
}

#endif

} // namespace acir_format
