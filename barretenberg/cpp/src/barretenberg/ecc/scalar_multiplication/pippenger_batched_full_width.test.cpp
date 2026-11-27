#include "barretenberg/common/thread.hpp"
#include "barretenberg/ecc/curves/bn254/bn254.hpp"
#include "barretenberg/ecc/curves/grumpkin/grumpkin.hpp"
#include "barretenberg/ecc/scalar_multiplication/pippenger_arena_layout.hpp"
#include "barretenberg/ecc/scalar_multiplication/scalar_multiplication_fast.hpp"
#include "barretenberg/numeric/random/engine.hpp"
#include "barretenberg/polynomials/polynomial.hpp"
#include <gtest/gtest.h>
#include <type_traits>

using namespace bb;

namespace {
auto& engine = numeric::get_randomness();

class ConcurrencyScope {
    size_t prev_;

  public:
    explicit ConcurrencyScope(size_t n)
        : prev_(bb::get_num_cpus())
    {
        bb::set_parallel_for_concurrency(n);
    }
    ~ConcurrencyScope() { bb::set_parallel_for_concurrency(prev_); }
    ConcurrencyScope(const ConcurrencyScope&) = delete;
    ConcurrencyScope& operator=(const ConcurrencyScope&) = delete;
    ConcurrencyScope(ConcurrencyScope&&) = delete;
    ConcurrencyScope& operator=(ConcurrencyScope&&) = delete;
};
} // namespace

/**
 * @brief Batches whose small members have single-thread arenas large enough that the memory budget, not the
 *        member count, caps the concurrent pool. On wasm, at a pool width where that cap leaves at most half the
 *        pool busy, the driver runs those members one at a time at full width; at narrower widths it keeps the
 *        pool. Every width must return the same commitments.
 */
template <class Curve> class PippengerBatchedFullWidthTest : public ::testing::Test {
  public:
    using Element = typename Curve::Element;
    using AffineElement = typename Curve::AffineElement;
    using ScalarField = typename Curve::ScalarField;

    static constexpr size_t kSmallN = scalar_multiplication::SMALL_MSM_BATCH_THRESHOLD;
    // One member above the small-member bound keeps the sequential large-member path in the same batch.
    static constexpr size_t kNumPoints = kSmallN + 1024;
    static constexpr size_t kWidePool = 8;

    // points[i] = G * dlogs[i], so each expected MSM is one scalar multiplication of G.
    static inline std::vector<ScalarField> dlogs{};
    static inline std::vector<AffineElement> points{};

    static void SetUpTestSuite()
    {
        dlogs.resize(kNumPoints);
        for (auto& d : dlogs) {
            d = ScalarField::random_element(&engine);
        }
        points.resize(kNumPoints);
        parallel_for_range(kNumPoints, [&](size_t start, size_t end) {
            for (size_t i = start; i < end; ++i) {
                points[i] = AffineElement(Curve::Group::one * dlogs[i]);
            }
        });
    }

    struct Batch {
        std::vector<std::vector<ScalarField>> scalars;
        std::vector<uint32_t> dedup_infos;
        std::vector<AffineElement> expected;
    };

    // Dense members near the bound, a dedup-hinted member of repeated scalars, a member sparse enough to fall to
    // the compacting fallback at full width, an all-zero member, and one member above the bound.
    static Batch make_batch()
    {
        Batch b;
        auto add = [&](size_t n, uint32_t dedup_info, auto&& scalar_at) {
            std::vector<ScalarField> s(n);
            ScalarField acc = ScalarField::zero();
            for (size_t i = 0; i < n; ++i) {
                s[i] = scalar_at(i);
                acc += s[i] * dlogs[i];
            }
            b.scalars.push_back(std::move(s));
            b.dedup_infos.push_back(dedup_info);
            b.expected.push_back(AffineElement(Curve::Group::one * acc));
        };
        auto random = [](size_t) { return ScalarField::random_element(&engine); };
        for (size_t k = 0; k < 8; ++k) {
            add(kSmallN - (61 * k), 0, random);
        }
        const uint256_t high_bit(0, 0, 0, uint64_t{ 1 } << (200 - 192));
        add(kSmallN - 500, 1, [&](size_t i) { return ScalarField(high_bit + uint256_t((i / 8) + 1)); });
        add(kSmallN, 0, [&](size_t i) { return (i % 137 == 0) ? random(i) : ScalarField::zero(); });
        add(kSmallN / 2, 0, [](size_t) { return ScalarField::zero(); });
        add(kNumPoints, 0, random);
        return b;
    }

    static std::vector<AffineElement> run(Batch& b)
    {
        std::vector<PolynomialSpan<ScalarField>> spans;
        for (auto& s : b.scalars) {
            spans.emplace_back(0, std::span<ScalarField>(s));
        }
        return scalar_multiplication::MSM_fast<Curve>::batch_multi_scalar_mul(
            points, spans, /*handle_edge_cases=*/false, b.dedup_infos);
    }

    // Mirrors the driver's arena sizing for this batch's small members (one shared-SRS group, so GLV follows the
    // group's largest member), so a sizer change that stops the batch from reaching the full-width path fails here
    // instead of silently testing only the pool. The sizer is instantiated for BN254 only; Grumpkin's field and
    // point sizes are the same.
    static void expect_wide_pool_runs_full_width(const Batch& b)
    {
        ConcurrencyScope scope(kWidePool);
        const bool ext_glv = kNumPoints <= scalar_multiplication::round_parallel_detail::GLV_SMALL_N_THRESHOLD;
        size_t small_count = 0;
        size_t pool_arena = 0;
        size_t full_width_arena = 0;
        for (size_t m = 0; m < b.scalars.size(); ++m) {
            const size_t n = b.scalars[m].size();
            if (n > kSmallN) {
                continue;
            }
            ++small_count;
            const bool dd = b.dedup_infos[m] != 0;
            pool_arena =
                std::max(pool_arena, scalar_multiplication::compute_arena_bytes_for_msm<Curve>(n, ext_glv, dd, 1));
            full_width_arena =
                std::max(full_width_arena, scalar_multiplication::compute_arena_bytes_for_msm<Curve>(n, ext_glv, dd));
        }
        ASSERT_GT(pool_arena, size_t{ 0 });
        const size_t workers_by_budget =
            std::max<size_t>(1, scalar_multiplication::round_parallel_detail::BATCH_MEM_BUDGET / pool_arena);
        EXPECT_LT(workers_by_budget, std::min(kWidePool, small_count));
        EXPECT_LE(2 * workers_by_budget, kWidePool);
        EXPECT_LE(full_width_arena, workers_by_budget * pool_arena);
    }
};

using CurveTypes = ::testing::Types<bb::curve::BN254, bb::curve::Grumpkin>;
TYPED_TEST_SUITE(PippengerBatchedFullWidthTest, CurveTypes);

TYPED_TEST(PippengerBatchedFullWidthTest, SameCommitmentsAtEveryPoolWidth)
{
    auto batch = TestFixture::make_batch();
#ifdef __wasm__
    if constexpr (std::is_same_v<TypeParam, bb::curve::BN254>) {
        TestFixture::expect_wide_pool_runs_full_width(batch);
    }
#endif
    // On wasm, width 1 has no concurrent pool; at 8 (asserted above) and at any other width where the rule holds the
    // members run at full width, and elsewhere the pool runs.
    for (const size_t width : { size_t{ 1 }, size_t{ 2 }, size_t{ 4 }, size_t{ 8 }, size_t{ 16 } }) {
        ConcurrencyScope scope(width);
        const auto result = TestFixture::run(batch);
        ASSERT_EQ(result.size(), batch.expected.size());
        for (size_t m = 0; m < result.size(); ++m) {
            EXPECT_EQ(result[m], batch.expected[m])
                << "pool width " << width << ", member " << m << " (n=" << batch.scalars[m].size() << ")";
        }
    }
}
