#pragma once

// Implementation fragment included from scalar_multiplication_fast.cpp inside
// bb::scalar_multiplication, after pippenger_round_parallel is defined.

// Multi-MSM_fast driver for `MSM_fast<>::batch_multi_scalar_mul`. The hot path
// (`CommitmentKey::batch_commit` from `commit_to_wires`) batches K MSMs sharing the same
// SRS subspan. We do NOT interleave K MSMs inside a single parallel_for body — that
// K-multiplies the per-thread working set and forces windows_in_batch=1; the single-MSM_fast
// hot path is tuned to fit ~4 MiB in L2 and we want to preserve that. The loop is just
//   for m in 0..K: run single-MSM_fast dispatch for MSM_fast m.
// The only cross-MSM_fast amortisation is the GLV-doubled point set: when every member of a
// shared-SRS-prefix group wants GLV, we double the prefix once into a shared buffer and
// each per-MSM_fast call aliases its prefix instead of doubling its own.
namespace round_parallel_detail {

// One per shared-SRS-prefix group. Membership is keyed on identical
// `point_arrays[m].data()` pointers — that is the actual sharing relation
// `commit_to_wires` exposes. Lives for one `pippenger_round_parallel_batched`
// call: the GLV-doubled buffer is recomputed per batch and freed at return.
template <typename Curve> struct BatchMsmGlvGroup {
    const typename Curve::AffineElement* base_ptr = nullptr; // SRS prefix pointer
    size_t group_max_n = 0;                                  // max n_input across MSMs in this group
    std::span<typename Curve::AffineElement> doubled;        // length 2 * group_max_n; aliases a prefix of
                                                             // the master-group buffer (computed once for
                                                             // the largest GLV-using group). Layout
                                                             // `[P_0, φP_0, P_1, φP_1, …]` — the first 2*n
                                                             // entries are the per-MSM_fast view for n ≤ Nmax.
    std::vector<size_t> member_msms;                         // indices into `scalar_arrays` of MSMs in this group
};

} // namespace round_parallel_detail

namespace {
// NOLINTNEXTLINE(readability-function-size, readability-function-cognitive-complexity,
// google-readability-function-size)
template <typename Curve>
void pippenger_round_parallel_batched(std::span<std::span<typename Curve::ScalarField>> scalar_arrays,
                                      std::span<std::span<const typename Curve::AffineElement>> point_arrays,
                                      std::vector<typename Curve::Element>& out_results,
                                      std::span<const uint32_t> dedup_infos = {}) noexcept
{
    using AffineElement = typename Curve::AffineElement;
    using ScalarField = typename Curve::ScalarField;
    using BaseField = typename Curve::BaseField;

    BB_BENCH_NAME("MSM_fast::pippenger_round_parallel_batched");

    const size_t K = scalar_arrays.size();
    BB_ASSERT_EQ(point_arrays.size(), K);
    out_results.assign(K, Curve::Group::point_at_infinity);

    auto info_for = [&](size_t m) noexcept -> size_t { return m < dedup_infos.size() ? dedup_infos[m] : 0; };

    if (K == 0) {
        return;
    }
    if (K == 1) {
        const size_t n = scalar_arrays[0].size();
        BB_ASSERT_DEBUG(n <= point_arrays[0].size());
        if (n == 0) {
            return;
        }
        PolynomialSpan<const ScalarField> sp(0, std::span<const ScalarField>(scalar_arrays[0].data(), n));
        out_results[0] = pippenger_round_parallel<Curve>(sp, point_arrays[0], info_for(0));
        return;
    }

    std::vector<size_t> n_input(K);
    for (size_t m = 0; m < K; ++m) {
        n_input[m] = scalar_arrays[m].size();
        BB_ASSERT_DEBUG(n_input[m] <= point_arrays[m].size());
    }

    // Group MSMs by shared SRS pointer; one shared GLV-doubled buffer per group, sized to
    // group_max_n. group_uses_glv is a per-group bool but the per-MSM_fast internal dispatch keeps
    // each MSM_fast's own GLV decision in case shared doubling is skipped.
    using GlvGroup = round_parallel_detail::BatchMsmGlvGroup<Curve>;
    std::vector<GlvGroup> glv_groups;

    auto find_or_create_group = [&](const AffineElement* base_ptr, size_t n) -> size_t {
        for (size_t g = 0; g < glv_groups.size(); ++g) {
            if (glv_groups[g].base_ptr == base_ptr) {
                glv_groups[g].group_max_n = std::max(glv_groups[g].group_max_n, n);
                return g;
            }
        }
        GlvGroup g{};
        g.base_ptr = base_ptr;
        g.group_max_n = n;
        glv_groups.push_back(std::move(g));
        return glv_groups.size() - 1;
    };

    std::vector<size_t> msm_to_group(K, std::numeric_limits<size_t>::max());
    for (size_t m = 0; m < K; ++m) {
        if (n_input[m] == 0) {
            continue;
        }
        const size_t g = find_or_create_group(point_arrays[m].data(), n_input[m]);
        glv_groups[g].member_msms.push_back(m);
        msm_to_group[m] = g;
    }

    std::vector<bool> group_uses_glv(glv_groups.size(), false);
    for (size_t g = 0; g < glv_groups.size(); ++g) {
        // GLV decision is per-group on group_max_n. Within a group, every MSM_fast has
        // n[m] <= group_max_n; if group_max_n is in the small-N regime, every MSM_fast
        // is too, so they all want GLV. If group_max_n is in the large-N regime,
        // no MSM_fast in the group wants GLV (they'd be slower with it).
        group_uses_glv[g] = glv_groups[g].group_max_n <= round_parallel_detail::GLV_SMALL_N_THRESHOLD;
    }

    // Build ONE shared GLV-doubled buffer covering the union of every GLV-using group's
    // SRS range, then alias each group's `doubled` into a slice of that buffer.
    //
    // Every production / test caller of batch_multi_scalar_mul is `commitment_key.batch_commit`,
    // which constructs each MSM_fast's point span as `get_monomial_points().subspan(start_index)`
    // — sub-spans of a single contiguous `std::vector<AffineElement>` SRS. So in every
    // batch every group's `base_ptr` lives in the same allocation and offsets are
    // necessarily integer multiples of `sizeof(AffineElement)`. The asserts below
    // catch a future caller that violates that contract.
    std::unique_ptr<AffineElement[]> master_doubled_owner; // NOLINT(cppcoreguidelines-avoid-c-arrays)
    {
        BB_BENCH_NAME("MSM_fast::pippenger_round_parallel_batched/glv_double_points");

        const AffineElement* min_base = nullptr;
        for (size_t g = 0; g < glv_groups.size(); ++g) {
            glv_groups[g].doubled = {};
            if (!group_uses_glv[g]) {
                continue;
            }
            if (min_base == nullptr || std::less<const AffineElement*>{}(glv_groups[g].base_ptr, min_base)) {
                min_base = glv_groups[g].base_ptr;
            }
        }

        if (min_base != nullptr) {
            const auto min_addr = reinterpret_cast<uintptr_t>(min_base);
            size_t max_extent_units = 0;
            for (size_t g = 0; g < glv_groups.size(); ++g) {
                if (!group_uses_glv[g]) {
                    continue;
                }
                const auto base_addr = reinterpret_cast<uintptr_t>(glv_groups[g].base_ptr);
                const uintptr_t offset_bytes = base_addr - min_addr;
                BB_ASSERT_EQ(offset_bytes % sizeof(AffineElement),
                             size_t{ 0 },
                             "GLV group base_ptr not aligned to AffineElement boundary "
                             "(point spans must be subranges of a contiguous AffineElement array)");
                const size_t offset_units = offset_bytes / sizeof(AffineElement);
                const size_t end_units = offset_units + glv_groups[g].group_max_n;
                max_extent_units = std::max(max_extent_units, end_units);
            }

            master_doubled_owner = std::make_unique_for_overwrite<AffineElement[]>(
                2 * max_extent_units); // NOLINT(cppcoreguidelines-avoid-c-arrays)
            AffineElement* const master_buf = master_doubled_owner.get();
            const BaseField beta = BaseField::cube_root_of_unity();
            bb::parallel_for(bb::get_num_cpus(), [&](const ThreadChunk& chunk) {
                for (size_t i : chunk.range(max_extent_units)) {
                    master_buf[2 * i] = min_base[i];
                    master_buf[(2 * i) + 1].x = min_base[i].x * beta;
                    master_buf[(2 * i) + 1].y = -min_base[i].y;
                }
            });

            for (size_t g = 0; g < glv_groups.size(); ++g) {
                if (!group_uses_glv[g]) {
                    continue;
                }
                const auto base_addr = reinterpret_cast<uintptr_t>(glv_groups[g].base_ptr);
                const size_t offset_units = (base_addr - min_addr) / sizeof(AffineElement);
                glv_groups[g].doubled =
                    std::span<AffineElement>(master_buf + (2 * offset_units), 2 * glv_groups[g].group_max_n);
            }
        }
    }

    // Batch split by n_input: members >= MSM_MIN_PTS_PER_THREAD * pool_width (large enough to clear the
    // per-worker floor even when split across all workers) run sequentially, each internally
    // multithreaded; smaller members run concurrently, one per worker, single-threaded. Keyed on
    // n_input (span), not the active/non-zero count: the deciding cost is the concurrent working-set
    // footprint, which tracks the span. A sparse-but-wide MSM (few non-zeros, large domain) must stay
    // "large" — splitting by active count instead drops it into the concurrent pool and thrashes cache.
    const size_t pool_width = bb::get_num_cpus();
#ifdef __wasm__
    // wasm runs every MSM single-threaded, so the split uses a fixed point-count bound rather than
    // MSM_MIN_PTS_PER_THREAD (SIZE_MAX here, which would classify every member as small). The loop
    // below tests `n < mt_threshold` and the intended classification is `n <= SMALL_MSM_BATCH_THRESHOLD`,
    // so the exclusive bound is one past it.
    const size_t mt_threshold = SMALL_MSM_BATCH_THRESHOLD + 1;
#else
    // overflow-safe MSM_MIN_PTS_PER_THREAD * pool_width
    const size_t mt_threshold =
        (pool_width <= 1 || MSM_MIN_PTS_PER_THREAD > std::numeric_limits<size_t>::max() / pool_width)
            ? std::numeric_limits<size_t>::max()
            : MSM_MIN_PTS_PER_THREAD * pool_width;
#endif
    std::vector<size_t> small_members;
    std::vector<size_t> large_members;
    small_members.reserve(K);
    large_members.reserve(K);
    for (size_t m = 0; m < K; ++m) {
        if (n_input[m] == 0) {
            continue;
        }
        if (pool_width > 1 && n_input[m] < mt_threshold) {
            small_members.push_back(m);
        } else {
            large_members.push_back(m);
        }
    }
    if (small_members.size() < 2) {
        // A lone small member gains nothing from the concurrent path; keep it on the
        // shared-arena sequential dispatch.
        large_members.insert(large_members.end(), small_members.begin(), small_members.end());
        std::sort(large_members.begin(), large_members.end());
        small_members.clear();
    }

    auto external_glv_for = [&](size_t m, size_t n) noexcept -> std::span<const AffineElement> {
        const size_t g = msm_to_group[m];
        if (g != std::numeric_limits<size_t>::max() && group_uses_glv[g] && !glv_groups[g].doubled.empty()) {
            // First 2*n entries of the group's interleaved doubled buffer are this MSM's GLV view,
            // valid for any n <= Nmax (see BatchMsmGlvGroup::doubled for the layout).
            return { glv_groups[g].doubled.data(), 2 * n };
        }
        return {};
    };

    // Run the large members concurrently (one per worker, single-threaded, below) instead of
    // sequentially at full width. It keeps a live arena per worker vs the sequential path's single
    // reused arena, so gate it to a large, balanced, sufficiently-threaded batch:
    //   - size > CONCURRENT_MIN_MEMBERS: only a commitment over a whole wide trace batches this many
    //     columns at once (100s-1000s); every other is <= ~86, so 100 keeps them sequential while
    //     the wide-trace case (385+) qualifies — paying the extra arenas only where it pays off.
    //   - max n <= Σn / pool_width (n is the work proxy): with largest-first ordering (below) the
    //     makespan is max(largest, Σn / pool_width), so a dominant member can't strand one worker.
    //   - pool_width >= CONCURRENT_MIN_POOL_WIDTH: fewer threads make the win too small to justify.
#ifdef __wasm__
    // wasm keeps large members on the sequential shared-arena dispatch: with single-threaded MSMs the
    // concurrent path's per-worker live arenas buy nothing over one reused arena.
    const bool large_members_concurrent = false;
#else
    static constexpr size_t CONCURRENT_MIN_MEMBERS = 100;
    static constexpr size_t CONCURRENT_MIN_POOL_WIDTH = 4;
    size_t total_large_n = 0;
    size_t max_large_n = 0;
    for (size_t m : large_members) {
        total_large_n += n_input[m];
        max_large_n = std::max(max_large_n, n_input[m]);
    }
    // The kernel converts a member's scalars in place and restores them, so members may only run concurrently
    // if their scalar spans are pairwise disjoint; the sequential path tolerates repeated or overlapping spans.
    const auto scalar_spans_disjoint = [&] {
        std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
        ranges.reserve(large_members.size());
        for (size_t m : large_members) {
            const auto begin = reinterpret_cast<uintptr_t>(scalar_arrays[m].data());
            ranges.emplace_back(begin, begin + (n_input[m] * sizeof(ScalarField)));
        }
        std::sort(ranges.begin(), ranges.end());
        for (size_t i = 1; i < ranges.size(); ++i) {
            if (ranges[i].first < ranges[i - 1].second) {
                return false;
            }
        }
        return true;
    };
    const bool large_members_concurrent = pool_width >= CONCURRENT_MIN_POOL_WIDTH &&
                                          large_members.size() > CONCURRENT_MIN_MEMBERS &&
                                          max_large_n <= total_large_n / pool_width && scalar_spans_disjoint();
#endif

    // A few independent large native MSMs can instead share the CPU budget,
    // each using a private multi-threaded arena. Keep the established wide-AVM
    // single-thread-per-member policy above unchanged.
    size_t large_job_count = 0;
    [[maybe_unused]] size_t threads_per_large_job = 0;
    [[maybe_unused]] size_t large_job_arena_bytes = 0;
#if !defined(__wasm__) && !defined(NO_MULTITHREADING) && SIZE_MAX > UINT32_MAX
    constexpr size_t MIN_LARGE_JOB_THREADS = 4;
    constexpr size_t MAX_LARGE_JOBS = 4;
    // Admission cap on canonical arena payload only, not expected allocation or
    // a process RSS bound (GLV storage, sort scratch and thread stacks are extra).
    constexpr size_t LARGE_JOB_ARENA_PAYLOAD_CAP = size_t{ 16 } << 30;
    if (!large_members_concurrent && large_members.size() >= 2 && pool_width >= 2 * MIN_LARGE_JOB_THREADS &&
        total_large_n >= (size_t{ 1 } << 18)) {
        // The kernel temporarily converts scalar backing in place and restores
        // it. Sequential large members allowed repeated/overlapping spans, so
        // preserve that path unless every concurrent mutable span is disjoint.
        bool disjoint_scalars = true;
        for (size_t i = 0; i < large_members.size() && disjoint_scalars; ++i) {
            const size_t a = large_members[i];
            if (n_input[a] > std::numeric_limits<size_t>::max() / sizeof(ScalarField)) {
                disjoint_scalars = false;
                break;
            }
            const uintptr_t first = reinterpret_cast<uintptr_t>(scalar_arrays[a].data());
            const size_t first_bytes = n_input[a] * sizeof(ScalarField);
            for (size_t j = 0; j < i; ++j) {
                const size_t b = large_members[j];
                const uintptr_t second = reinterpret_cast<uintptr_t>(scalar_arrays[b].data());
                const size_t second_bytes = n_input[b] * sizeof(ScalarField);
                if (first <= second ? second - first < first_bytes : first - second < second_bytes) {
                    disjoint_scalars = false;
                    break;
                }
            }
        }
        const size_t max_jobs = std::min({ MAX_LARGE_JOBS, large_members.size(), pool_width / MIN_LARGE_JOB_THREADS });
        for (size_t jobs = disjoint_scalars ? max_jobs : 0; jobs >= 2; --jobs) {
            const size_t average = total_large_n / jobs;
            // Avoid assigning most of the work to one independently capped job.
            if (max_large_n > average + average / 2) {
                continue;
            }
            const size_t threads = pool_width / jobs;
            size_t arena_bytes = 0;
            for (size_t m : large_members) {
                const bool ext_glv = !external_glv_for(m, n_input[m]).empty();
                arena_bytes = std::max(arena_bytes,
                                       compute_arena_bytes_for_msm<Curve>(n_input[m], ext_glv, info_for(m), threads));
            }
            if (arena_bytes <= LARGE_JOB_ARENA_PAYLOAD_CAP / jobs) {
                large_job_count = jobs;
                threads_per_large_job = threads;
                large_job_arena_bytes = arena_bytes;
                break;
            }
        }
    }
#endif

    // Shared dynamically-sized arena for the sequential (large-member) calls. Sized to
    // the max requirement across those members so each MSM_fast finds enough space; a
    // single allocation across the batch (vs one per MSM_fast if we passed {} down).
    // dedup_active varies per MSM_fast (gated by per-MSM_fast hint), so the budget query must
    // mirror the predicate used inside pippenger_round_parallel.
    size_t shared_arena_bytes = 0;
    std::unique_ptr<std::byte[]> shared_arena_owner; // NOLINT(cppcoreguidelines-avoid-c-arrays)
    std::span<std::byte> shared_arena;
    if (!large_members_concurrent && large_job_count == 0) {
        for (size_t m : large_members) {
            const bool ext_glv = !external_glv_for(m, n_input[m]).empty();
            // The internal short-circuits to trivial_msm_threaded for tiny MSMs, so the hint
            // alone is the right arena-sizing predicate (over-sizing for a path that bails
            // is harmless — under-sizing would crash).
            const size_t bytes = compute_arena_bytes_for_msm<Curve>(n_input[m], ext_glv, info_for(m));
            shared_arena_bytes = std::max(shared_arena_bytes, bytes);
        }
        if (shared_arena_bytes > 0) {
            shared_arena_owner = std::make_unique_for_overwrite<std::byte[]>(
                shared_arena_bytes); // NOLINT(cppcoreguidelines-avoid-c-arrays)
            shared_arena = std::span<std::byte>(shared_arena_owner.get(), shared_arena_bytes);
        }
    }
    // Concurrent small-member dispatch: workers pull members off an atomic cursor and run
    // each with a thread-capped pipeline (max_threads=1, so the member never re-enters the
    // pool) out of a per-worker arena sized for the capped layout. The GLV-doubled buffer
    // is read-only and shared across workers.
    if (!small_members.empty()) {
        BB_BENCH_NAME("MSM_fast::pippenger_round_parallel_batched/small_members");
        size_t small_arena_bytes = 0;
        for (size_t m : small_members) {
            const bool ext_glv = !external_glv_for(m, n_input[m]).empty();
            const size_t bytes =
                compute_arena_bytes_for_msm<Curve>(n_input[m], ext_glv, info_for(m), /*max_threads=*/1);
            small_arena_bytes = std::max(small_arena_bytes, bytes);
        }
        // The per-worker arenas are one contiguous block of `num_workers * small_arena_bytes`.
        // Native servers may keep more independent small MSMs busy at once. This aggregate
        // allowance does not change any individual MSM's 32 MiB window/layout budget.
        // 2 GiB bounds this experiment to 64 such budgets; it is not a tuned optimum.
        // Compared with the old aggregate cap, the arena block can grow by at most 2016 MiB.
        // Other worker/TLS allocations and RSS must still be measured separately.
#ifdef __wasm__
        constexpr size_t aggregate_arena_budget = round_parallel_detail::BATCH_MEM_BUDGET;
#else
        constexpr size_t aggregate_arena_budget = 2ULL * 1024ULL * 1024ULL * 1024ULL;
#endif
        const size_t workers_by_budget = small_arena_bytes > 0
                                             ? std::max<size_t>(1, aggregate_arena_budget / small_arena_bytes)
                                             : std::numeric_limits<size_t>::max();
        const size_t num_workers = std::min({ pool_width, small_members.size(), workers_by_budget });
        std::unique_ptr<std::byte[]> small_arena_owner; // NOLINT(cppcoreguidelines-avoid-c-arrays)
        if (small_arena_bytes > 0) {
            small_arena_owner = std::make_unique_for_overwrite<std::byte[]>(
                num_workers * small_arena_bytes); // NOLINT(cppcoreguidelines-avoid-c-arrays)
        }
        // A block this large is normally mapped on its own, so its free unmaps every resident page on this thread. The
        // workers discard their slices' interior pages first, which leaves that free little to unmap. glibc still
        // serves the block from the heap when free heap space holds it; the next allocation there then faults the
        // discarded pages back in. The floor is release_pages_in_parallel's.
        const bool release_slices = num_workers * small_arena_bytes >= (size_t{ 64 } << 20);
        std::atomic<size_t> next_member{ 0 };
        bb::parallel_for(num_workers, [&](size_t tid) {
            std::span<std::byte> worker_arena;
            if (small_arena_bytes > 0) {
                worker_arena = { small_arena_owner.get() + (tid * small_arena_bytes), small_arena_bytes };
            }
            while (true) {
                const size_t s = next_member.fetch_add(1, std::memory_order_relaxed);
                if (s >= small_members.size()) {
                    break;
                }
                const size_t m = small_members[s];
                const size_t n = n_input[m];
                PolynomialSpan<const ScalarField> sp(0, std::span<const ScalarField>(scalar_arrays[m].data(), n));
                out_results[m] = pippenger_round_parallel<Curve>(
                    sp, point_arrays[m], info_for(m), external_glv_for(m, n), worker_arena, /*max_threads=*/1);
            }
            if (release_slices) {
                round_parallel_detail::release_arena_slice(worker_arena);
            }
        });
    }

    if (large_members_concurrent) {
        // Workers pull work off an atomic cursor and run it single-threaded (max_threads=1). This skips the
        // member's cross-thread reduction. It also keeps the member off the pool, so there is no nested
        // parallel_for. The gate above guarantees at least pool_width members, so every worker starts busy. Each
        // call self-allocates its arena (empty span), because a shared num_workers × max-member arena would
        // exceed BATCH_MEM_BUDGET and cap num_workers below pool_width.
        //
        // A member's cost tracks its non-zero scalars, not its span, so the span-based gate admits batches in
        // which a few dense members each take longer on one thread than the rest of the batch takes on all of
        // them. Members wider than CONCURRENT_CHUNK_POINTS therefore run as contiguous chunks of at most that
        // many points: an MSM is a sum over its terms, so a member's result is the sum of its chunks' results.
        // Chunks run longest-first to bound the tail.
        BB_BENCH_NAME("MSM_fast::pippenger_round_parallel_batched/large_members_concurrent");
        constexpr size_t CONCURRENT_CHUNK_POINTS = size_t{ 1 } << 16;
        struct Chunk {
            size_t member;
            size_t offset;
            size_t length;
        };
        std::vector<Chunk> chunks;
        for (size_t m : large_members) {
            const size_t n = n_input[m];
            const size_t parts = (n + CONCURRENT_CHUNK_POINTS - 1) / CONCURRENT_CHUNK_POINTS;
            for (size_t k = 0; k < parts; ++k) {
                const size_t begin = (n * k) / parts;
                chunks.push_back({ m, begin, ((n * (k + 1)) / parts) - begin });
            }
        }
        std::sort(chunks.begin(), chunks.end(), [](const Chunk& a, const Chunk& b) {
            if (a.length != b.length) {
                return a.length > b.length;
            }
            return a.member != b.member ? a.member < b.member : a.offset < b.offset;
        });
        std::vector<typename Curve::Element> partials(chunks.size());
        const size_t num_workers = std::min(pool_width, chunks.size());
        std::atomic<size_t> next_chunk{ 0 };
        bb::parallel_for(num_workers, [&](size_t) {
            while (true) {
                const size_t c = next_chunk.fetch_add(1, std::memory_order_relaxed);
                if (c >= chunks.size()) {
                    break;
                }
                const auto [m, offset, length] = chunks[c];
                const size_t n = n_input[m];
                // A measured duplicate count describes the whole member; each chunk gets its share.
                const size_t info = info_for(m) >= 2 ? std::max<size_t>(2, (info_for(m) * length) / n) : info_for(m);
                std::span<const AffineElement> glv = external_glv_for(m, n);
                if (!glv.empty()) {
                    glv = glv.subspan(2 * offset, 2 * length);
                }
                PolynomialSpan<const ScalarField> sp(
                    0, std::span<const ScalarField>(scalar_arrays[m].data() + offset, length));
                partials[c] = pippenger_round_parallel<Curve>(
                    sp, point_arrays[m].subspan(offset), info, glv, {}, /*max_threads=*/1);
            }
        });
        // out_results starts at infinity; summing in chunk order keeps the result deterministic.
        for (size_t c = 0; c < chunks.size(); ++c) {
            out_results[chunks[c].member] += partials[c];
        }
#if !defined(__wasm__) && !defined(NO_MULTITHREADING) && SIZE_MAX > UINT32_MAX
    } else if (large_job_count != 0) {
        BB_BENCH_NAME("MSM_fast::pippenger_round_parallel_batched/large_member_jobs");
        std::sort(
            large_members.begin(), large_members.end(), [&](size_t a, size_t b) { return n_input[a] > n_input[b]; });
        std::atomic<size_t> next_large{ 0 };
        auto* const bench_parent = bb::detail::GlobalBenchStatsContainer::parent;
        std::vector<std::thread> jobs;
        jobs.reserve(large_job_count);
        for (size_t job = 0; job < large_job_count; ++job) {
            try {
                jobs.emplace_back([&] {
                    // A new caller thread owns a new TLS pool. Set its width
                    // BEFORE the first MSM dispatch; active job widths together
                    // never exceed the original caller's pool_width.
                    bb::set_parallel_for_concurrency(threads_per_large_job);
                    bb::detail::GlobalBenchStatsContainer::parent = bench_parent;
                    std::unique_ptr<std::byte[]> owner; // NOLINT(cppcoreguidelines-avoid-c-arrays)
                    std::span<std::byte> arena;
                    if (large_job_arena_bytes != 0) {
                        owner = std::make_unique_for_overwrite<std::byte[]>(large_job_arena_bytes);
                        arena = { owner.get(), large_job_arena_bytes };
                    }
                    while (true) {
                        const size_t index = next_large.fetch_add(1, std::memory_order_relaxed);
                        if (index >= large_members.size()) {
                            break;
                        }
                        const size_t m = large_members[index];
                        const size_t n = n_input[m];
                        PolynomialSpan<const ScalarField> sp(0,
                                                             std::span<const ScalarField>(scalar_arrays[m].data(), n));
                        out_results[m] = pippenger_round_parallel<Curve>(
                            sp, point_arrays[m], info_for(m), external_glv_for(m, n), arena, threads_per_large_job);
                    }
                });
            } catch (const std::system_error&) {
                // Any successfully created jobs still drain the complete queue.
                // If none could start, use the original per-member fallback below.
                break;
            }
        }
        for (auto& job : jobs) {
            job.join();
        }
        if (jobs.empty()) {
            for (size_t m : large_members) {
                const size_t n = n_input[m];
                PolynomialSpan<const ScalarField> sp(0, std::span<const ScalarField>(scalar_arrays[m].data(), n));
                out_results[m] =
                    pippenger_round_parallel<Curve>(sp, point_arrays[m], info_for(m), external_glv_for(m, n));
            }
        }
#endif
    } else {
        // Sequential large-member dispatch: one member at a time, each running the full single-
        // MSM_fast pipeline (its own from-Mont and to-Mont, schedule, Stage 1-6b) across the whole
        // pool. Taken when the concurrent gate above does not hold — too few members to fill the
        // pool, or a member large enough to warrant the full pool on its own. The only batched
        // amortisation shared is the doubled SRS prefix above.
        for (size_t m : large_members) {
            const size_t n = n_input[m];
            PolynomialSpan<const ScalarField> sp(0, std::span<const ScalarField>(scalar_arrays[m].data(), n));
            out_results[m] =
                pippenger_round_parallel<Curve>(sp, point_arrays[m], info_for(m), external_glv_for(m, n), shared_arena);
        }
    }
}
} // namespace

template <typename Curve>
std::vector<typename Curve::AffineElement> MSM_fast<Curve>::batch_multi_scalar_mul(
    std::span<const typename Curve::AffineElement> points,
    std::span<PolynomialSpan<typename Curve::ScalarField>> scalars,
    bool handle_edge_cases,
    std::span<const uint32_t> dedup_infos) noexcept
{
    BB_BENCH_NAME("MSM_fast::batch_multi_scalar_mul");
    const size_t k = scalars.size();

    // Adapt the new (single shared points span + per-MSM_fast PolynomialSpan scalars) API to
    // the internal dispatcher, which still takes one point sub-span per MSM_fast. Each MSM_fast's
    // sub-span is `points[start_index .. start_index + size)`; the dispatcher's existing
    // GLV-doubled-buffer grouping then deduplicates across MSMs that fall in the same
    // underlying allocation.
    std::vector<std::span<const AffineElement>> point_subspans;
    std::vector<std::span<ScalarField>> scalar_subspans;
    point_subspans.reserve(k);
    scalar_subspans.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        const size_t start_i = scalars[i].start_index;
        BB_ASSERT_LTE(start_i, points.size(), "scalars[m].start_index exceeds shared points span");
        point_subspans.push_back(points.subspan(start_i, points.size() - start_i));
        scalar_subspans.push_back(scalars[i].span);
    }

    auto info_for = [&](size_t m) noexcept -> size_t { return m < dedup_infos.size() ? dedup_infos[m] : 0; };

    if (handle_edge_cases) {
        std::vector<AffineElement> results(k);
        for (size_t i = 0; i < k; ++i) {
            const size_t n = scalar_subspans[i].size();
            PolynomialSpan<const ScalarField> scalar_span(0,
                                                          std::span<const ScalarField>(scalar_subspans[i].data(), n));
            results[i] =
                AffineElement(pippenger_fast<Curve>(scalar_span, point_subspans[i], handle_edge_cases, info_for(i)));
        }
        return results;
    }

    std::vector<typename Curve::Element> per_msm_jac;
    pippenger_round_parallel_batched<Curve>(scalar_subspans, point_subspans, per_msm_jac, dedup_infos);

    std::vector<AffineElement> results(k);
    for (size_t i = 0; i < k; ++i) {
        results[i] = AffineElement(per_msm_jac[i]);
    }
    return results;
}
