#include "barretenberg/common/log.hpp"
#include "barretenberg/common/serialize.hpp"
#include "barretenberg/common/thread.hpp"
#include "barretenberg/ecc/curves/bn254/bn254.hpp"
#include "barretenberg/ecc/curves/grumpkin/grumpkin.hpp"
#include "barretenberg/numeric/random/engine.hpp"
#include "barretenberg/polynomials/polynomial.hpp"
#include "scalar_multiplication.hpp"
#include <gtest/gtest.h>

#include <chrono>
#include <thread>

using namespace bb;
using scalar_multiplication::SmallMsmSchedule;
using scalar_multiplication::round_parallel_detail::COST_SEGMENTS;
using scalar_multiplication::round_parallel_detail::MIN_SPLIT_PART_POINTS;
using scalar_multiplication::round_parallel_detail::MsmPart;
using scalar_multiplication::round_parallel_detail::plan_cost_split_parts;
using scalar_multiplication::round_parallel_detail::SegmentBits;

namespace {
auto& engine = numeric::get_randomness();

// Each member's parts tile [0, length); every part of a split member has MIN_SPLIT_PART_POINTS points and a
// non-zero estimated cost.
void expect_tiling(const std::vector<MsmPart>& parts,
                   std::span<const size_t> members,
                   std::span<const size_t> lengths,
                   const std::string& context)
{
    std::vector<MsmPart> sorted = parts;
    std::sort(sorted.begin(), sorted.end(), [](const MsmPart& a, const MsmPart& b) {
        return a.member != b.member ? a.member < b.member : a.offset < b.offset;
    });
    size_t i = 0;
    for (size_t k = 0; k < members.size(); ++k) {
        size_t next = 0;
        size_t count = 0;
        while (i < sorted.size() && sorted[i].member == members[k]) {
            EXPECT_EQ(sorted[i].offset, next) << context << ": member " << members[k] << " has a gap or overlap";
            EXPECT_GT(sorted[i].length, 0U) << context;
            next = sorted[i].offset + sorted[i].length;
            ++count;
            ++i;
        }
        EXPECT_EQ(next, lengths[k]) << context << ": member " << members[k] << " not covered";
        if (count > 1) {
            for (const MsmPart& p : parts) {
                if (p.member == members[k]) {
                    EXPECT_GE(p.length, MIN_SPLIT_PART_POINTS) << context << ": member " << members[k];
                    EXPECT_GT(p.cost, 0U) << context << ": member " << members[k] << " has a zero-cost part";
                }
            }
        }
    }
    EXPECT_EQ(i, sorted.size()) << context << ": parts of unknown members";
}

size_t part_count(const std::vector<MsmPart>& parts, size_t member)
{
    return static_cast<size_t>(
        std::count_if(parts.begin(), parts.end(), [member](const MsmPart& p) noexcept { return p.member == member; }));
}

// The plan a batch makes at `pool_width` workers (empty when it makes none): its small members are those below
// MSM_MIN_PTS_PER_THREAD * pool_width points, there must be two of them, and one must fit two parts.
template <typename ScalarField>
std::vector<MsmPart> batch_plan(const std::vector<std::vector<ScalarField>>& scalars, size_t pool_width)
{
    std::vector<size_t> ids;
    std::vector<size_t> lengths;
    std::vector<SegmentBits> bits;
    bool splittable = false;
    for (size_t k = 0; k < scalars.size(); ++k) {
        const size_t n = scalars[k].size();
        if (pool_width > 1 && n != 0 && n < scalar_multiplication::MSM_MIN_PTS_PER_THREAD * pool_width) {
            ids.push_back(k);
            lengths.push_back(n);
            bits.push_back(scalar_multiplication::round_parallel_detail::sample_segment_bits<ScalarField>(scalars[k]));
            splittable |= n >= 2 * MIN_SPLIT_PART_POINTS;
        }
    }
    if (ids.size() < 2 || !splittable) {
        return {};
    }
    return plan_cost_split_parts(ids, lengths, bits, pool_width);
}

template <typename AffineElement> std::vector<AffineElement> random_points(size_t n)
{
    // Serial: the debug RNG is shared and not thread-safe.
    std::vector<AffineElement> points(n);
    for (auto& p : points) {
        p = AffineElement::random_element(&engine);
    }
    return points;
}

template <typename ScalarField> std::vector<ScalarField> random_scalars(size_t n)
{
    std::vector<ScalarField> s(n);
    for (auto& x : s) {
        x = ScalarField::random_element(&engine);
    }
    return s;
}

// Scalars of exactly the field's bit length, so every sample weighs the same and a plan's cut points are fixed.
template <typename ScalarField> std::vector<ScalarField> full_width_scalars(size_t n)
{
    constexpr size_t BITS = ScalarField::modulus.get_msb() + 1;
    std::vector<ScalarField> s(n);
    for (auto& x : s) {
        do {
            x = ScalarField::random_element(&engine);
        } while (static_cast<uint256_t>(x).get_msb() + 1 != BITS);
    }
    return s;
}

// Runs one batch on a fresh thread whose pool has `pool_width` workers (native); where the pool is process-wide, at
// its current width.
template <typename Curve>
std::vector<typename Curve::AffineElement> run_batch(std::span<const typename Curve::AffineElement> points,
                                                     std::vector<std::vector<typename Curve::ScalarField>>& scalars,
                                                     std::span<const size_t> starts,
                                                     std::span<const uint32_t> dedup_infos,
                                                     SmallMsmSchedule schedule,
                                                     [[maybe_unused]] size_t pool_width)
{
    std::vector<PolynomialSpan<typename Curve::ScalarField>> spans;
    for (size_t k = 0; k < scalars.size(); ++k) {
        spans.emplace_back(starts[k], std::span<typename Curve::ScalarField>(scalars[k]));
    }
    std::vector<typename Curve::AffineElement> out;
#if defined(NO_MULTITHREADING) || defined(__wasm__)
    out = scalar_multiplication::MSM<Curve>::batch_multi_scalar_mul(points, spans, dedup_infos, schedule);
#else
    std::thread worker([&] {
        set_parallel_for_concurrency(pool_width);
        out = scalar_multiplication::MSM<Curve>::batch_multi_scalar_mul(points, spans, dedup_infos, schedule);
    });
    worker.join();
#endif
    return out;
}
} // namespace

// The planner splits only members above the fair share, at most min(length / MIN_SPLIT_PART_POINTS, pool_width)
// ways, tiles every member exactly, and orders parts by cost, then member, then offset.
TEST(BatchMsmSplitByCostPlanner, TilesSplitsAndOrders)
{
    SegmentBits full{};
    full.fill(254);
    // Cost in the first eight segments and the last one: the only cut after the head that keeps every part
    // non-empty in cost and MIN_SPLIT_PART_POINTS long is the first boundary past MIN_SPLIT_PART_POINTS.
    SegmentBits head{};
    for (size_t j = 0; j < 8; ++j) {
        head[j] = 254;
    }
    head[COST_SEGMENTS - 1] = 254;
    SegmentBits none{};

    const std::vector<size_t> members = { 3, 5, 8, 9, 11 };
    const std::vector<size_t> lengths = { 16000, 300, 16383, 9000, 4095 };
    const std::vector<SegmentBits> bits = { full, full, head, none, full };
    for (const size_t pool_width : { size_t{ 1 }, size_t{ 2 }, size_t{ 4 }, size_t{ 64 } }) {
        const std::string context = "pool " + std::to_string(pool_width);
        const std::vector<MsmPart> parts = plan_cost_split_parts(members, lengths, bits, pool_width);
        expect_tiling(parts, members, lengths, context);
        for (size_t s = 1; s < parts.size(); ++s) {
            const MsmPart& a = parts[s - 1];
            const MsmPart& b = parts[s];
            const bool ordered =
                a.cost != b.cost ? a.cost > b.cost : (a.member != b.member ? a.member < b.member : a.offset < b.offset);
            EXPECT_TRUE(ordered) << context << ": parts " << s - 1 << " and " << s;
        }
        EXPECT_EQ(part_count(parts, 5), 1U) << context << ": a member below MIN_SPLIT_PART_POINTS splits";
        EXPECT_EQ(part_count(parts, 9), 1U) << context << ": a zero-cost member splits";
        EXPECT_EQ(part_count(parts, 11), 1U) << context << ": a member below 2 * MIN_SPLIT_PART_POINTS splits";
        EXPECT_LE(part_count(parts, 3), std::min<size_t>(pool_width, 16000 / MIN_SPLIT_PART_POINTS)) << context;
        if (pool_width == 1) {
            EXPECT_EQ(parts.size(), members.size()) << context << ": one worker splits nothing";
        }
    }

    // At 64 workers the dense member takes the most parts allowed, and the head-and-tail member is cut once, at the
    // first segment boundary past MIN_SPLIT_PART_POINTS, so its tail part carries only the last segment's cost.
    const std::vector<MsmPart> parts = plan_cost_split_parts(members, lengths, bits, 64);
    EXPECT_EQ(part_count(parts, 3), 16000 / MIN_SPLIT_PART_POINTS);
    ASSERT_EQ(part_count(parts, 8), 2U);
    for (const MsmPart& p : parts) {
        if (p.member == 8 && p.offset != 0) {
            EXPECT_EQ(p.offset, 2303U);
            EXPECT_EQ(p.cost, uint64_t{ 254 } * (16383 - ((16383 * (COST_SEGMENTS - 1)) / COST_SEGMENTS)));
        }
    }
}

template <class Curve> class BatchMsmSplitByCostTest : public ::testing::Test {};

using CurveTypes = ::testing::Types<curve::BN254, curve::Grumpkin>;
TYPED_TEST_SUITE(BatchMsmSplitByCostTest, CurveTypes);

// SPLIT_BY_COST must give every member the commitment the existing schedule and the edge-case-safe single MSM give,
// byte for byte after serialization, and restore every scalar, at 1, 4, 32 and 64 workers. The members cover dense,
// narrow, sparse, all-zero, a single non-zero, a part holding a single non-zero, duplicates with a measured count, a
// member that sums to infinity while its parts do not, a part that sums to infinity on its own, and members on the
// shared doubled-point table. Every part of a split member holds a non-zero sampled scalar, but its result can still
// be infinity (member 10).
TYPED_TEST(BatchMsmSplitByCostTest, MatchesUnsplitAndSafeMsm)
{
    using AffineElement = typename TypeParam::AffineElement;
    using ScalarField = typename TypeParam::ScalarField;

    // Region A, a shared prefix as in a wire batch: members 0-6. Region B: member 7, whose last point is chosen.
    // Region C: members 8-9, at most GLV_SMALL_N_THRESHOLD points, so their group uses the shared doubled-point table.
    // Region D: member 10, whose first part's last point is chosen.
    constexpr size_t REGION_A = 16383;
    constexpr size_t CANCEL_N = 15000;
    constexpr size_t GLV_N = scalar_multiplication::round_parallel_detail::GLV_SMALL_N_THRESHOLD;
    constexpr size_t PART_CANCEL_N = 6000;
    constexpr size_t START_B = REGION_A;
    constexpr size_t START_C = START_B + CANCEL_N;
    constexpr size_t START_D = START_C + GLV_N;
    std::vector<AffineElement> points = random_points<AffineElement>(START_D + PART_CANCEL_N);

    std::vector<std::vector<ScalarField>> scalars;
    std::vector<size_t> starts;
    std::vector<uint32_t> dedup_infos;
    auto add = [&](size_t start, std::vector<ScalarField> s, uint32_t info) {
        starts.push_back(start);
        scalars.push_back(std::move(s));
        dedup_infos.push_back(info);
    };
    auto msm_safe = [&](size_t start, std::span<const ScalarField> s) {
        return scalar_multiplication::MSM<TypeParam>::msm(
            points, PolynomialSpan<const ScalarField>(start, s), /*handle_edge_cases=*/true);
    };

    add(0, random_scalars<ScalarField>(16000), 0); // 0: dense, full width
    add(0, random_scalars<ScalarField>(300), 0);   // 1: narrow
    {
        std::vector<ScalarField> s(12000, ScalarField::zero()); // 2: sparse
        for (size_t i = 0; i < s.size(); i += 7) {
            s[i] = ScalarField::random_element(&engine);
        }
        add(0, std::move(s), 0);
    }
    add(0, std::vector<ScalarField>(5000, ScalarField::zero()), 0); // 3: all zero
    // 4: dense in its first eight segments, and one non-zero scalar at the last segment's sample point; it splits
    // once after the head, so its tail part holds that single non-zero.
    constexpr size_t TAIL_NONZERO = (REGION_A * ((2 * COST_SEGMENTS) - 1)) / (2 * COST_SEGMENTS);
    {
        std::vector<ScalarField> s(REGION_A, ScalarField::zero());
        for (size_t i = 0; i < (8 * REGION_A) / COST_SEGMENTS; ++i) {
            s[i] = ScalarField::random_element(&engine);
        }
        s[TAIL_NONZERO] = ScalarField::random_element(&engine);
        add(0, std::move(s), 0);
    }
    {
        // 5: a single non-zero scalar off every sample point (estimated cost 0, one part).
        std::vector<ScalarField> s(10000, ScalarField::zero());
        s[7777] = ScalarField::random_element(&engine);
        add(0, std::move(s), 0);
    }
    {
        // 6: runs of 8 equal scalars with their measured duplicate count, shared out across its parts.
        const uint256_t high_bit(0, 0, 0, uint64_t{ 1 } << (200 - 192));
        std::vector<ScalarField> s(14000);
        uint32_t dup_count = 0;
        for (size_t i = 0; i < s.size(); ++i) {
            s[i] = ScalarField(high_bit + uint256_t((i / 8) + 1));
            dup_count += static_cast<uint32_t>(i % 8 != 0);
        }
        add(0, std::move(s), dup_count);
    }
    {
        // 7: dense, and its last point is minus the sum of the rest, so the member is infinity while no part is.
        std::vector<ScalarField> s = random_scalars<ScalarField>(CANCEL_N);
        s.back() = ScalarField::one();
        points[START_B + CANCEL_N - 1] = -msm_safe(START_B, std::span<const ScalarField>(s.data(), CANCEL_N - 1));
        add(START_B, std::move(s), 0);
    }
    add(START_C, random_scalars<ScalarField>(GLV_N), 1);     // 8: dense, shared doubled-point table, dedup hint
    // 9: same group, half width. A 4,096-point member has one legal two-way cut, at its middle, taken only when the
    // first half's sampled cost reaches half the total; equal-width scalars make the halves weigh the same.
    add(START_C, full_width_scalars<ScalarField>(GLV_N / 2), 0);
    add(START_D, random_scalars<ScalarField>(PART_CANCEL_N), 0);

    // 10: dense, cut in two at 32 and 64 workers. The last point of its first part is minus the sum of the rest of
    // that part, so that part alone is infinity. The cut lies on a segment boundary and the changed scalar sits just
    // before it, off every sample point, so the change leaves the plan as it is. WASM classifies small members by a
    // fixed bound and never reaches these plans; there any cut gives a valid member.
#ifdef __wasm__
    const size_t cut = PART_CANCEL_N / 2;
#else
    const std::vector<MsmPart> plan64 = batch_plan(scalars, 64);
    size_t cut = 0;
    for (const MsmPart& p : plan64) {
        if (p.member == 10 && p.offset == 0) {
            cut = p.length;
        }
    }
    ASSERT_EQ(part_count(plan64, 10), 2U);
#endif
    scalars[10][cut - 1] = ScalarField::one();
    points[START_D + cut - 1] = -msm_safe(START_D, std::span<const ScalarField>(scalars[10].data(), cut - 1));
    EXPECT_TRUE(msm_safe(START_D, std::span<const ScalarField>(scalars[10].data(), cut)).is_point_at_infinity());

    const size_t num_msms = scalars.size();
    std::vector<AffineElement> expected(num_msms);
    for (size_t k = 0; k < num_msms; ++k) {
        expected[k] = msm_safe(starts[k], scalars[k]);
    }
    EXPECT_TRUE(expected[7].is_point_at_infinity());

#ifndef __wasm__
    // The plans the batch makes at 64 and 32 workers.
    {
        const std::vector<MsmPart> plan = batch_plan(scalars, 64);
        ASSERT_EQ(plan.size(), plan64.size());
        std::vector<size_t> ids(num_msms);
        std::vector<size_t> lengths(num_msms);
        for (size_t k = 0; k < num_msms; ++k) {
            ids[k] = k;
            lengths[k] = scalars[k].size();
        }
        expect_tiling(plan, ids, lengths, "plan at 64");
        for (const size_t k : { size_t{ 0 }, size_t{ 6 }, size_t{ 7 }, size_t{ 8 }, size_t{ 10 } }) {
            EXPECT_GT(part_count(plan, k), 1U) << "member " << k << " is not split at 64 workers";
        }
        EXPECT_EQ(part_count(plan, 3), 1U);
        EXPECT_EQ(part_count(plan, 5), 1U);
        ASSERT_EQ(part_count(plan, 4), 2U);
        for (const MsmPart& p : plan) {
            if (p.member == 4 && p.offset != 0) {
                const auto& s = scalars[4];
                const auto nonzero = std::count_if(s.begin() + static_cast<std::ptrdiff_t>(p.offset),
                                                   s.end(),
                                                   [](const ScalarField& x) noexcept { return !x.is_zero(); });
                EXPECT_EQ(nonzero, 1) << "member 4's tail part does not hold exactly one non-zero";
            }
            if (p.member == 10 && p.offset == 0) {
                EXPECT_EQ(p.length, cut) << "member 10's plan moved";
            }
        }

        // At 32 workers only members under 8,192 points are small; members 9 and 10 split, and member 10 at the same
        // cut, since a two-part cut sits at the cost median whatever the worker count.
        const std::vector<MsmPart> plan32 = batch_plan(scalars, 32);
        EXPECT_EQ(part_count(plan32, 9), 2U);
        ASSERT_EQ(part_count(plan32, 10), 2U);
        for (const MsmPart& p : plan32) {
            if (p.member == 10 && p.offset == 0) {
                EXPECT_EQ(p.length, cut);
            }
        }
    }
#endif

#if defined(NO_MULTITHREADING) || defined(__wasm__)
    const std::vector<size_t> widths = { get_num_cpus() };
#else
    const std::vector<size_t> widths = { 1, 4, 32, 64 };
#endif
    for (const size_t pool_width : widths) {
        const auto original = scalars;
        const std::vector<AffineElement> split =
            run_batch<TypeParam>(points, scalars, starts, dedup_infos, SmallMsmSchedule::SPLIT_BY_COST, pool_width);
        EXPECT_EQ(scalars, original) << "pool " << pool_width << ": scalars changed";
        const std::vector<AffineElement> whole =
            run_batch<TypeParam>(points, scalars, starts, dedup_infos, SmallMsmSchedule::WHOLE, pool_width);
        ASSERT_EQ(split.size(), num_msms);
        ASSERT_EQ(whole.size(), num_msms);
        for (size_t k = 0; k < num_msms; ++k) {
            EXPECT_EQ(to_buffer(split[k]), to_buffer(expected[k])) << "pool " << pool_width << ", member " << k;
            EXPECT_EQ(to_buffer(split[k]), to_buffer(whole[k])) << "pool " << pool_width << ", member " << k;
        }
    }
}

// Without edge-case handling, a batch assumes no two partial sums it adds share an x coordinate, and aborts on the zero
// denominator otherwise. Here two equal (or opposite) points carry the scalar 2^100 and every other scalar is below
// 2^20, so in the window holding bit 100 those two points are their bucket's only entries and the affine tree adds them
// directly (with the dedup pre-pass, its cluster sum does). Scalars below 2^127 skip the GLV split (k1 = k, k2 = 0), so
// this holds with and without the doubled-point table. Both points sit inside one part: both schedules must abort.
// Death tests need a native process with threads.
#if !defined(__wasm__) && !defined(NO_MULTITHREADING)
TEST(BatchMsmSplitByCostDeathTest, CoincidingPointsInOneBucketAbort)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    using Curve = curve::BN254;
    using AffineElement = Curve::AffineElement;
    using ScalarField = Curve::ScalarField;

    constexpr size_t POOL = 32;
    constexpr size_t N = 6000;
    constexpr size_t FILLER = 300;
    constexpr size_t A = 100;
    const std::vector<AffineElement> base_points = random_points<AffineElement>(N + FILLER);
    auto small_scalars = [](size_t n) {
        std::vector<ScalarField> s(n);
        for (auto& x : s) {
            x = ScalarField(1 + (engine.get_random_uint32() % ((1U << 20) - 1)));
        }
        return s;
    };
    std::vector<std::vector<ScalarField>> scalars = { small_scalars(N), small_scalars(FILLER) };
    const ScalarField pair_scalar = ScalarField(uint256_t(1) << 100);
    scalars[0][A] = pair_scalar;
    scalars[0][A + 1] = pair_scalar;
    ASSERT_EQ(ScalarField::split_into_endomorphism_scalars(pair_scalar.from_montgomery_form()).second,
              (std::array<uint64_t, 2>{ 0, 0 }));
    const std::vector<size_t> starts = { 0, N };
    const std::vector<MsmPart> plan = batch_plan(scalars, POOL);
    ASSERT_EQ(part_count(plan, 0), 2U);
    for (const MsmPart& p : plan) {
        if (p.member == 0 && p.offset == 0) {
            ASSERT_GT(p.length, A + 1) << "the pair must share the first part";
        }
    }

    struct Case {
        bool opposite;
        uint32_t dedup;
    };
    for (const Case c : { Case{ false, 0 }, Case{ true, 1 } }) {
        std::vector<AffineElement> points = base_points;
        points[A + 1] = c.opposite ? -points[A] : points[A];
        const std::vector<uint32_t> dedup_infos = { c.dedup, 0 };
        auto run = [&](SmallMsmSchedule schedule) {
            static_cast<void>(run_batch<Curve>(points, scalars, starts, dedup_infos, schedule, POOL));
        };
        SCOPED_TRACE(std::string(c.opposite ? "opposite" : "equal") + (c.dedup != 0 ? ", dedup" : ""));
        EXPECT_DEATH(run(SmallMsmSchedule::WHOLE), "invert zero");
        EXPECT_DEATH(run(SmallMsmSchedule::SPLIT_BY_COST), "invert zero");
    }
}
#endif

// Not run by default (it prints, it does not check): the cost of SPLIT_BY_COST's planning on 4,000 small members,
// against the existing schedule, at 64 workers. With no member long enough for two parts the planner returns before
// sampling; with one such member (all zero, so nothing splits) every small member is sampled and planned.
#if !defined(__wasm__) && !defined(NO_MULTITHREADING)
TEST(BatchMsmSplitByCostPlanner, DISABLED_PlanningCost)
{
    using Curve = curve::BN254;
    using ScalarField = Curve::ScalarField;
    constexpr size_t TINY = 4000;
    const std::vector<Curve::AffineElement> points = random_points<Curve::AffineElement>(2 * MIN_SPLIT_PART_POINTS);
    for (const bool with_splittable : { false, true }) {
        std::vector<std::vector<ScalarField>> scalars;
        for (size_t k = 0; k < TINY; ++k) {
            scalars.push_back(random_scalars<ScalarField>(16 + (k % 49)));
        }
        if (with_splittable) {
            scalars.emplace_back(2 * MIN_SPLIT_PART_POINTS, ScalarField::zero());
        }
        const std::vector<size_t> starts(scalars.size(), 0);
        const std::vector<uint32_t> dedup_infos(scalars.size(), 0);
        auto best_us = [&](SmallMsmSchedule schedule) {
            double best = 1e30;
            for (size_t rep = 0; rep < 20; ++rep) {
                const auto t0 = std::chrono::steady_clock::now();
                static_cast<void>(run_batch<Curve>(points, scalars, starts, dedup_infos, schedule, 64));
                const auto t1 = std::chrono::steady_clock::now();
                best = std::min(best, std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
            return best;
        };
        const double whole = best_us(SmallMsmSchedule::WHOLE);
        const double split = best_us(SmallMsmSchedule::SPLIT_BY_COST);
        info("planning cost, ",
             scalars.size(),
             " small members",
             with_splittable ? " (one splittable)" : " (none splittable)",
             ": whole ",
             whole,
             " us, split ",
             split,
             " us, best of 20");
    }
}
#endif
