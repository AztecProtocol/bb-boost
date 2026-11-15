#pragma once

#include "barretenberg/common/thread.hpp"
#include "barretenberg/ecc/curves/bn254/bn254.hpp"
#include "barretenberg/ecc/curves/grumpkin/grumpkin.hpp"
#include "barretenberg/polynomials/polynomial.hpp"
#include <string>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace bb::scalar_multiplication {

/**
 * @brief N-dependent oversubscription factor used ONLY for `choose_window_bits`'
 *        target_load formula (not for actual thread dispatch).
 */
size_t window_bits_tuning_oversub_factor(size_t n_input);

/**
 * @brief State of the art pippenger_fast multiscalar multiplication algorithm.
 *
 * @details A traditional pippenger_fast N/logN algorithm (split scalars into windows, use window value to map scalar's
 * base point into a bucket. Accumulate buckets. Repeat for all windows (1 window = 1 round)) We add the following
 * optimizations on top of the algorithm: 1) Efficient multithreading via round-parallelism. If memory budget allows
 * each thread evaluates multiple rounds. Thread efficiency ~90% when measured. 2) Booth in-place recoding of scalars.
 * Each slice represents [-(num_buckets/2- 1), ..., (num_buckets/2 - 1)]. halves number of buckets. 3) GLV decompositon
 * of input scalars into two half-size scalars. Halves number of bucket accumulation steps. Only used for n<2^{16} due
 * to memory throughput tradeoffs. 4) Adaptive bucket range. When mixing large and small scalars (e.g. witness values),
 * low bit-ranges use a larger bucket-range. Larger bit-ranges use a bucket-range tuned to the reduced number of large
 * scalars. 5) Duplicate stripping. Witness commitments and permutation polynomials contain large numbers of duplicates.
 * When dedup flag is active these are detected and base points consolidated prior to main MSM_fast. 6) Batch-affine
 *    arithmetic. When accumulating points into buckets, independent point additions are gathered and batched, allowing
 *    for affine point arithmetic w. batch invert. Cost = 6M for addition, 7M for doubling (M=field mult).
 * 7) Batch-affine bucket accumulation. Uses novel technique from (TODO REF) to accumulate buckets in runs of
 *    independent additions.
 * 8) If n is small (num points per thread < ~24), fallback to optimised multithreaded Straus MSM_fast
 *    (windowed double-and-add with GLV endomorphism)
 *
 * In order to efficiently utilize memory and prevent WASM memory fragmentation, we use a single `arena` memory buffer
 * to allocate all temporary data structures. Currently sized at 36MB which defines the upper cap on the memory consumed
 * by this algorihtm.
 * @param scalars  Input scalars
 * @param points   Input points
 * @param dedup_hint Activates duplicate stripping if true. Off by default as dup stripping adds ~5% overhead on random
 * inputs
 * @pre Inputs are linearly independent: no point-at-infinity, no equal-x within a bucket
 *      (matches `pippenger_unsafe_fast`).
 * @note Scalars are converted out of Montgomery form internally and restored before return.
 */
// `external_glv_doubled`: optional caller-supplied [P, φP, ...] interleaved buffer
//   (length 2*n). When non-empty, every n_input is treated as GLV-eligible and the
//   doubled points are aliased instead of recomputed — the batched driver uses this
//   to share the doubled SRS prefix across MSMs in a batch.
// `external_arena`: optional caller-supplied scratch buffer ≥ this MSM_fast's required
//   bytes. When empty, allocated per-MSM_fast and freed at return. The batched driver
//   supplies a single arena sized to the largest member.
// `dedup_info`: MSM dedup pre-pass hint — 0 = off, 1 = hinted (no estimate), >=2 = a caller-measured
//   duplicate count used to discount the window-selection point count.
template <typename Curve>
typename Curve::Element pippenger_round_parallel(
    PolynomialSpan<const typename Curve::ScalarField> scalars,
    std::span<const typename Curve::AffineElement> points,
    size_t dedup_info = 0,
    std::span<const typename Curve::AffineElement> external_glv_doubled = {},
    std::span<std::byte> external_arena = {},
    size_t max_threads = 0) noexcept;

extern template curve::BN254::Element pippenger_round_parallel<curve::BN254>(
    PolynomialSpan<const curve::BN254::ScalarField> scalars,
    std::span<const curve::BN254::AffineElement> points,
    size_t dedup_info,
    std::span<const curve::BN254::AffineElement> external_glv_doubled,
    std::span<std::byte> external_arena,
    size_t max_threads) noexcept;

extern template curve::Grumpkin::Element pippenger_round_parallel<curve::Grumpkin>(
    PolynomialSpan<const curve::Grumpkin::ScalarField> scalars,
    std::span<const curve::Grumpkin::AffineElement> points,
    size_t dedup_info,
    std::span<const curve::Grumpkin::AffineElement> external_glv_doubled,
    std::span<std::byte> external_arena,
    size_t max_threads) noexcept;

// ===================================================================================
// Public API (interface-compatible with the legacy `scalar_multiplication::MSM_fast` class).
// ===================================================================================
//
// `pippenger_fast`        — handle_edge_cases routed: false → fast affine round-parallel,
//                      true → Jacobian fast path (handles point-at-infinity / equal-x
//                      bucket collisions).
// `pippenger_unsafe_fast` — always the fast path; caller asserts linear-independence of points.
// `MSM_fast<Curve>::msm`            — single-MSM_fast convenience wrapper (returns AffineElement).
// `MSM_fast<Curve>::batch_multi_scalar_mul` — multi-MSM_fast driver: runs each MSM_fast via `pippenger_fast`
//                                       and returns a vector of AffineElement results.

template <typename Curve>
typename Curve::Element pippenger_fast(PolynomialSpan<const typename Curve::ScalarField> scalars,
                                       std::span<const typename Curve::AffineElement> points,
                                       bool handle_edge_cases = true,
                                       size_t dedup_info = 0) noexcept;

template <typename Curve>
typename Curve::Element pippenger_unsafe_fast(PolynomialSpan<const typename Curve::ScalarField> scalars,
                                              std::span<const typename Curve::AffineElement> points,
                                              size_t dedup_info = 0) noexcept;

extern template curve::BN254::Element pippenger_fast<curve::BN254>(
    PolynomialSpan<const curve::BN254::ScalarField> scalars,
    std::span<const curve::BN254::AffineElement> points,
    bool handle_edge_cases,
    size_t dedup_info) noexcept;

extern template curve::Grumpkin::Element pippenger_fast<curve::Grumpkin>(
    PolynomialSpan<const curve::Grumpkin::ScalarField> scalars,
    std::span<const curve::Grumpkin::AffineElement> points,
    bool handle_edge_cases,
    size_t dedup_info) noexcept;

extern template curve::BN254::Element pippenger_unsafe_fast<curve::BN254>(
    PolynomialSpan<const curve::BN254::ScalarField> scalars,
    std::span<const curve::BN254::AffineElement> points,
    size_t dedup_info) noexcept;

extern template curve::Grumpkin::Element pippenger_unsafe_fast<curve::Grumpkin>(
    PolynomialSpan<const curve::Grumpkin::ScalarField> scalars,
    std::span<const curve::Grumpkin::AffineElement> points,
    size_t dedup_info) noexcept;

/**
 * @brief How a batch MSM schedules its small members (those run one per worker, single-threaded).
 * @details WHOLE runs each small member as one task. SPLIT_BY_COST estimates each small member's cost from a sample
 *          of its scalars and runs any member above the pool's fair share as contiguous parts, whose results are
 *          summed per member in offset order. Both produce the same group elements.
 */
enum class SmallMsmSchedule : uint8_t {
    WHOLE,
    SPLIT_BY_COST,
};

template <typename Curve> class MSM_fast {
  public:
    using Element = typename Curve::Element;
    using ScalarField = typename Curve::ScalarField;
    using AffineElement = typename Curve::AffineElement;

    /**
     * @brief Single MSM_fast convenience wrapper — returns the result as an AffineElement.
     * @param handle_edge_cases  false (default): fast affine round-parallel path.
     *                           true: Jacobian fast path (handles edge cases).
     * @param dedup_hint         When true, opts this MSM_fast into the input-scalar dedup pre-pass.
     */
    static AffineElement msm(std::span<const AffineElement> points,
                             PolynomialSpan<const ScalarField> scalars,
                             bool handle_edge_cases = false,
                             size_t dedup_info = 0) noexcept;

    /**
     * @brief Batch driver for multiple MSMs. Returns one AffineElement per input MSM_fast.
     *
     * Every MSM_fast in the batch shares a single contiguous point set (the SRS / GLV table);
     * each MSM_fast picks its own range via `scalars[m].start_index` and `scalars[m].size()`.
     * MSM_fast `m` computes Σ_i scalars[m][i] * points[scalars[m].start_index + i].
     *
     * Large members run sequentially, each round-parallel across the whole pool. Small members,
     * and wide balanced batches of large members, run concurrently, one single-threaded member
     * per worker.
     * This matches the legacy interface but the parallelisation strategy is different:
     * the legacy implementation work-balanced points across threads spanning multiple
     * MSMs; round-parallel parallelises within each MSM_fast, so one MSM_fast at a time uses the
     * full thread pool. For the typical chonk workload (commit batches of polys of size
     * 2^20), this is faster because per-MSM_fast threading dominates over inter-MSM_fast stealing.
     *
     * @param points      Shared point set (SRS prefix). Every MSM_fast indexes into this span.
     * @param scalars     Per-MSM_fast scalars carrying the start offset into `points`. Precondition
     *                    (unchecked): the spans must not overlap. Non-GLV members convert their
     *                    scalars out of Montgomery form in place and restore them before return,
     *                    and members may run concurrently (wide batches of large members, and
     *                    the small-member pool), so overlapping spans race and can produce wrong
     *                    commitments or leave the shared scalars corrupted. Committing a
     *                    polynomial together with its shifted view, or the same polynomial
     *                    twice, violates this.
     * @param dedup_hints Optional per-MSM_fast dedup opt-ins (parallel to `scalars`): a
     *                    non-zero entry opts that MSM_fast's input scalars into the
     *                    duplicate-cluster pre-pass. Empty span means no dedup anywhere.
     */
    static std::vector<AffineElement> batch_multi_scalar_mul(std::span<const AffineElement> points,
                                                             std::span<PolynomialSpan<ScalarField>> scalars,
                                                             bool handle_edge_cases = true,
                                                             std::span<const uint32_t> dedup_infos = {}) noexcept;

    /**
     * @brief The batch driver without edge-case handling, with an explicit small-member schedule.
     * @details WHOLE is exactly `batch_multi_scalar_mul(points, scalars, false, dedup_infos)`.
     */
    static std::vector<AffineElement> batch_multi_scalar_mul(std::span<const AffineElement> points,
                                                             std::span<PolynomialSpan<ScalarField>> scalars,
                                                             std::span<const uint32_t> dedup_infos,
                                                             SmallMsmSchedule schedule) noexcept;
};

extern template class MSM_fast<curve::Grumpkin>;
extern template class MSM_fast<curve::BN254>;

// `pippenger_round_parallel` falls back to `trivial_msm_threaded` when each worker
// would receive fewer than this many points (after the n_active filter). Exposed so tests
// and bench targets can pin behaviour at the boundary.
inline constexpr size_t MIN_PTS_PER_THREAD_FOR_PIPPENGER = 24;

// Points-per-worker floor below which intra-MSM multithreading loses to its parallel_for barrier
// overhead. Drives both the worker-count pick in pippenger_round_parallel and the batch driver's
// concurrent/sequential split. SIZE_MAX forces single-threaded on wasm.
#ifdef __wasm__
inline constexpr size_t MSM_MIN_PTS_PER_THREAD = SIZE_MAX;
#else
inline constexpr size_t MSM_MIN_PTS_PER_THREAD = 256;
#endif

// Point-count bound for the batch driver's concurrent/sequential split on wasm. Intra-MSM work is
// always single-threaded on wasm (MSM_MIN_PTS_PER_THREAD == SIZE_MAX), so the native split rule
// `n < MSM_MIN_PTS_PER_THREAD * pool_width` would classify every member as small and route them all
// through the concurrent pool, whose per-worker arena is sized to the largest member and caps the
// worker count by memory budget. This finite bound keeps large members on the sequential shared-arena
// path so the concurrent pool retains full worker width. Members at or below it dispatch one-per-worker.
inline constexpr size_t SMALL_MSM_BATCH_THRESHOLD = size_t{ 1 } << 13;

// Per-MSM_fast arena sizer. Returns 0 for shapes that fall back to the Jacobian-fast path
// (no affine arena). Mirrors the inline budget calc inside `pippenger_round_parallel`;
// declared here so the test suite can exercise the same sizer.
template <typename Curve>
size_t compute_arena_bytes_for_msm(size_t n_input,
                                   bool external_glv_provided,
                                   bool dedup_active = false,
                                   size_t max_threads = 0) noexcept;

namespace round_parallel_detail {

// Above this N, GLV's 2x point-count cost outweighs the windows-halved benefit.
#ifdef __wasm__
inline constexpr size_t GLV_SMALL_N_THRESHOLD = size_t{ 1 } << 16;
#else
inline constexpr size_t GLV_SMALL_N_THRESHOLD = size_t{ 1 } << 13;
#endif

// SmallMsmSchedule::SPLIT_BY_COST samples one scalar per equal segment of a member; a part never ends inside a segment.
inline constexpr size_t COST_SEGMENTS = 64;
// Below this many points a part's fixed setup outweighs what splitting saves.
inline constexpr size_t MIN_SPLIT_PART_POINTS = size_t{ 1 } << 11;

using SegmentBits = std::array<uint8_t, COST_SEGMENTS>;

/// Contiguous points [offset, offset + length) of batch member `member`, with its estimated cost.
struct MsmPart {
    size_t member;
    size_t offset;
    size_t length;
    uint64_t cost;
};

/// First point of segment j of a member of n points.
inline size_t cost_segment_begin(size_t n, size_t j) noexcept
{
    return (n * j) / COST_SEGMENTS;
}

/**
 * @brief Bit length of the scalar at the middle of each of a member's COST_SEGMENTS segments (0 for zero).
 * @details A segment's estimated cost is its sampled bit length times its point count: an MSM's work tracks its
 *          non-zero scalars times their bits, not its span. Reads only.
 */
template <typename ScalarField> SegmentBits sample_segment_bits(std::span<const ScalarField> scalars) noexcept
{
    SegmentBits bits{};
    const size_t n = scalars.size();
    for (size_t j = 0; j < COST_SEGMENTS && n != 0; ++j) {
        const ScalarField& s = scalars[(n * ((2 * j) + 1)) / (2 * COST_SEGMENTS)];
        if (!s.is_zero()) {
            bits[j] = static_cast<uint8_t>(static_cast<numeric::uint256_t>(s).get_msb() + 1);
        }
    }
    return bits;
}

/**
 * @brief Cuts members into parts of roughly equal estimated cost, ordered for costliest-first claiming.
 * @param members Batch member indices; lengths[k] and bits[k] describe members[k].
 * @details A member whose cost exceeds the batch's fair share (total / pool_width) becomes up to
 *          min(ceil(cost / fair), length / MIN_SPLIT_PART_POINTS, pool_width) parts, cut at the first segment boundary
 *          where the running cost reaches the next quantile (i / parts of the cost). Every part of a split member has
 *          at least MIN_SPLIT_PART_POINTS points and a non-zero estimated cost, so it holds a non-zero scalar. Other
 *          members are one part. The parts of a member tile [0, length) exactly. Order: cost descending, then
 *          member, then offset.
 */
inline std::vector<MsmPart> plan_cost_split_parts(std::span<const size_t> members,
                                                  std::span<const size_t> lengths,
                                                  std::span<const SegmentBits> bits,
                                                  size_t pool_width) noexcept
{
    auto segment_cost = [&](size_t k, size_t j) noexcept -> uint64_t {
        const size_t n = lengths[k];
        return uint64_t{ bits[k][j] } * (cost_segment_begin(n, j + 1) - cost_segment_begin(n, j));
    };
    std::vector<uint64_t> member_cost(members.size(), 0);
    uint64_t total = 0;
    for (size_t k = 0; k < members.size(); ++k) {
        for (size_t j = 0; j < COST_SEGMENTS; ++j) {
            member_cost[k] += segment_cost(k, j);
        }
        total += member_cost[k];
    }
    const uint64_t fair = std::max<uint64_t>(1, total / std::max<size_t>(1, pool_width));

    std::vector<MsmPart> parts;
    parts.reserve(members.size());
    for (size_t k = 0; k < members.size(); ++k) {
        const size_t n = lengths[k];
        const uint64_t cost = member_cost[k];
        const size_t max_parts = std::min({ pool_width, n / MIN_SPLIT_PART_POINTS, COST_SEGMENTS });
        const size_t wanted = cost > fair ? static_cast<size_t>((cost + fair - 1) / fair) : 1;
        const size_t num_parts = std::min(wanted, max_parts);
        if (num_parts <= 1) {
            parts.push_back({ members[k], 0, n, cost });
            continue;
        }
        size_t begin = 0;
        size_t made = 0;
        uint64_t cum = 0;
        uint64_t cum_at_begin = 0;
        for (size_t j = 0; j < COST_SEGMENTS && made + 1 < num_parts; ++j) {
            cum += segment_cost(k, j);
            const size_t end = cost_segment_begin(n, j + 1);
            const uint64_t quantile = ((cost * (made + 1)) + num_parts - 1) / num_parts;
            if (cum >= quantile && cum > cum_at_begin && cum < cost && end - begin >= MIN_SPLIT_PART_POINTS &&
                n - end >= MIN_SPLIT_PART_POINTS) {
                parts.push_back({ members[k], begin, end - begin, cum - cum_at_begin });
                begin = end;
                cum_at_begin = cum;
                ++made;
            }
        }
        parts.push_back({ members[k], begin, n - begin, cost - cum_at_begin });
    }
    std::sort(parts.begin(), parts.end(), [](const MsmPart& a, const MsmPart& b) {
        if (a.cost != b.cost) {
            return a.cost > b.cost;
        }
        return a.member != b.member ? a.member < b.member : a.offset < b.offset;
    });
    return parts;
}

/**
 * @brief Single-MSM_fast, no-affine-trick Pippenger over window_bits-wide windows.
 *
 * `min_pts_per_thread_override` lets benchmarks pin behaviour:
 *   - 0 (default) → use the internal `MIN_PTS_PER_THREAD` heuristic (256 native, single-threaded on WASM).
 *   - SIZE_MAX    → force single-threaded.
 *   - 1           → maximally multi-threaded (one worker per logical CPU).
 */
template <typename Curve>
typename Curve::Element pippenger_round_parallel_jacobian_fast(std::span<const typename Curve::ScalarField> scalars,
                                                               std::span<const typename Curve::AffineElement> points,
                                                               size_t min_pts_per_thread_override = 0,
                                                               size_t max_threads = 0) noexcept;

extern template curve::BN254::Element pippenger_round_parallel_jacobian_fast<curve::BN254>(
    std::span<const curve::BN254::ScalarField> scalars,
    std::span<const curve::BN254::AffineElement> points,
    size_t min_pts_per_thread_override,
    size_t max_threads) noexcept;

extern template curve::Grumpkin::Element pippenger_round_parallel_jacobian_fast<curve::Grumpkin>(
    std::span<const curve::Grumpkin::ScalarField> scalars,
    std::span<const curve::Grumpkin::AffineElement> points,
    size_t min_pts_per_thread_override,
    size_t max_threads) noexcept;

} // namespace round_parallel_detail

/**
 * @brief Single-threaded small-MSM_fast driver: `Element::straus_msm` over the input slice.
 */
template <typename Curve>
typename Curve::Element trivial_msm(PolynomialSpan<const typename Curve::ScalarField> scalars_span,
                                    std::span<const typename Curve::AffineElement> all_points) noexcept;

extern template curve::BN254::Element trivial_msm<curve::BN254>(
    PolynomialSpan<const curve::BN254::ScalarField> scalars_span,
    std::span<const curve::BN254::AffineElement> all_points) noexcept;

extern template curve::Grumpkin::Element trivial_msm<curve::Grumpkin>(
    PolynomialSpan<const curve::Grumpkin::ScalarField> scalars_span,
    std::span<const curve::Grumpkin::AffineElement> all_points) noexcept;

/**
 * @brief Multi-threaded small-MSM_fast driver: parallel `Element::straus_msm` over zero-skipped
 *        input slices.
 */
template <typename Curve>
typename Curve::Element trivial_msm_threaded(PolynomialSpan<const typename Curve::ScalarField> scalars_span,
                                             std::span<const typename Curve::AffineElement> all_points,
                                             size_t max_threads = 0) noexcept;

extern template curve::BN254::Element trivial_msm_threaded<curve::BN254>(
    PolynomialSpan<const curve::BN254::ScalarField> scalars_span,
    std::span<const curve::BN254::AffineElement> all_points,
    size_t max_threads) noexcept;

extern template curve::Grumpkin::Element trivial_msm_threaded<curve::Grumpkin>(
    PolynomialSpan<const curve::Grumpkin::ScalarField> scalars_span,
    std::span<const curve::Grumpkin::AffineElement> all_points,
    size_t max_threads) noexcept;

} // namespace bb::scalar_multiplication
