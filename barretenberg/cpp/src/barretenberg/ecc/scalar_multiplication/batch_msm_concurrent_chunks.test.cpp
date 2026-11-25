#include "barretenberg/common/serialize.hpp"
#include "barretenberg/common/thread.hpp"
#include "barretenberg/ecc/curves/bn254/bn254.hpp"
#include "barretenberg/ecc/curves/grumpkin/grumpkin.hpp"
#include "barretenberg/numeric/random/engine.hpp"
#include "barretenberg/polynomials/polynomial.hpp"
#include "scalar_multiplication.hpp"
#include <algorithm>
#include <gtest/gtest.h>

using namespace bb;

namespace {
auto& engine = numeric::get_randomness();

class ConcurrencyScope {
    size_t prev_;

  public:
    explicit ConcurrencyScope(size_t n)
        : prev_(get_num_cpus())
    {
        set_parallel_for_concurrency(n);
    }
    ~ConcurrencyScope() { set_parallel_for_concurrency(prev_); }
    ConcurrencyScope(const ConcurrencyScope&) = delete;
    ConcurrencyScope& operator=(const ConcurrencyScope&) = delete;
    ConcurrencyScope(ConcurrencyScope&&) = delete;
    ConcurrencyScope& operator=(ConcurrencyScope&&) = delete;
};

// Independent random points and scalars: the affine batch addition assumes no two partial sums in a bucket coincide,
// which structured points under repeated scalars can violate. Every draw stays on this thread, since under
// BBERG_DEBUG_LOG the engine is one shared unsynchronised generator; only the point arithmetic runs in parallel.
template <class Curve>
void random_points_and_scalars(size_t n,
                               std::vector<typename Curve::AffineElement>& points,
                               std::vector<typename Curve::ScalarField>& scalars)
{
    using ScalarField = typename Curve::ScalarField;
    std::vector<ScalarField> discrete_logs(n);
    scalars.resize(n);
    for (size_t i = 0; i < n; ++i) {
        discrete_logs[i] = ScalarField::random_element(&engine);
        scalars[i] = ScalarField::random_element(&engine);
    }
    points.resize(n);
    parallel_for_range(n, [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            points[i] = Curve::Group::one * discrete_logs[i];
        }
    });
}
} // namespace

template <class Curve> class BatchMsmConcurrentChunksTest : public ::testing::Test {};

using CurveTypes = ::testing::Types<curve::BN254, curve::Grumpkin>;
TYPED_TEST_SUITE(BatchMsmConcurrentChunksTest, CurveTypes);

// The chunked path's arena block is capped at 2 GiB, so a pool wide enough that one slice per worker overflows the cap
// runs fewer workers than the pool. 128 members of 2^15 points at pool width 128 take the many-member admission as one
// chunk each; with every fourth scalar non-zero (a different residue per member) every member must still match its
// stand-alone MSM, checked exactly on the first, middle and last members and on all through one random linear
// combination, and get its scalars back. The test allocates an arena block of about 2 GiB.
TYPED_TEST(BatchMsmConcurrentChunksTest, BudgetCappedWorkersMatchStandaloneMsm)
{
#if defined(__wasm__) || defined(NO_MULTITHREADING)
    GTEST_SKIP() << "no concurrent large-member path without threads";
#else
    using Element = typename TypeParam::Element;
    using AffineElement = typename TypeParam::AffineElement;
    using ScalarField = typename TypeParam::ScalarField;

    constexpr size_t POOL = 128;
    constexpr size_t N = scalar_multiplication::MSM_MIN_PTS_PER_THREAD * POOL;
    // The dispatch's cap, checked against the smaller of the two GLV variants' slices, so it binds whichever the batch
    // uses.
    constexpr size_t CHUNK_ARENA_BUDGET = size_t{ 2 } << 30;
    const size_t with_glv = scalar_multiplication::compute_arena_bytes_for_msm<TypeParam>(N, true, false, 1);
    const size_t without_glv = scalar_multiplication::compute_arena_bytes_for_msm<TypeParam>(N, false, false, 1);
    const size_t slice = with_glv < without_glv ? with_glv : without_glv;
    ASSERT_GT(slice, 0U);
    ASSERT_LT(CHUNK_ARENA_BUDGET / slice, POOL);

    std::vector<AffineElement> points(N);
    for (auto& point : points) {
        point = AffineElement::random_element(&engine);
    }
    ConcurrencyScope concurrency(POOL);
    std::vector<std::vector<ScalarField>> scalars(POOL, std::vector<ScalarField>(N, ScalarField::zero()));
    std::vector<PolynomialSpan<ScalarField>> spans;
    for (size_t k = 0; k < POOL; ++k) {
        for (size_t i = k % 4; i < N; i += 4) {
            scalars[k][i] = ScalarField::random_element(&engine);
        }
        spans.emplace_back(0, std::span<ScalarField>(scalars[k]));
    }
    const auto original = scalars;
    const auto result =
        scalar_multiplication::MSM_fast<TypeParam>::batch_multi_scalar_mul(points, spans, /*handle_edge_cases=*/false);
    for (size_t k = 0; k < POOL; ++k) {
        EXPECT_EQ(scalars[k], original[k]) << "scalars of MSM " << k << " not restored";
    }
    for (const size_t k : { size_t{ 0 }, POOL / 2, POOL - 1 }) {
        const AffineElement expected = scalar_multiplication::MSM<TypeParam>::msm(
            points, PolynomialSpan<const ScalarField>(0, std::span<const ScalarField>(scalars[k])), true);
        EXPECT_EQ(to_buffer(result[k]), to_buffer(expected)) << "MSM " << k;
    }

    // Linearity over the shared points: sum_k r_k * MSM(points, s_k) == MSM(points, sum_k r_k * s_k) for non-zero r_k.
    std::vector<ScalarField> weights(POOL, ScalarField::zero());
    for (auto& weight : weights) {
        while (weight.is_zero()) {
            weight = ScalarField::random_element(&engine);
        }
    }
    std::vector<ScalarField> combined(N, ScalarField::zero());
    Element weighted_results = Element::infinity();
    for (size_t k = 0; k < POOL; ++k) {
        for (size_t i = k % 4; i < N; i += 4) {
            combined[i] += weights[k] * scalars[k][i];
        }
        weighted_results += Element(result[k]) * weights[k];
    }
    const AffineElement expected_combination = scalar_multiplication::MSM<TypeParam>::msm(
        points, PolynomialSpan<const ScalarField>(0, std::span<const ScalarField>(combined)), true);
    EXPECT_EQ(to_buffer(AffineElement(weighted_results)), to_buffer(expected_combination));
#endif
}

// Batches with more than 100 members, all at or above the large-member bound and none wider than Σn / pool_width,
// run their members concurrently; members wider than 2^16 points are split into chunks whose partial sums are added
// back. Each member must match a stand-alone MSM byte for byte, and its scalars must be restored.
TYPED_TEST(BatchMsmConcurrentChunksTest, MatchesStandaloneMsm)
{
#if defined(__wasm__) || defined(NO_MULTITHREADING)
    GTEST_SKIP() << "no concurrent large-member path without threads";
#else
    using AffineElement = typename TypeParam::AffineElement;
    using ScalarField = typename TypeParam::ScalarField;

    constexpr size_t CHUNK = size_t{ 1 } << 16;
    const std::vector<size_t> wide = { CHUNK, CHUNK + 1, (2 * CHUNK) + 12345 };
    const size_t max_wide = wide.back();

    std::vector<AffineElement> points;
    std::vector<ScalarField> pool;
    random_points_and_scalars<TypeParam>(max_wide, points, pool);

    const uint256_t high_bit(0, 0, 0, uint64_t{ 1 } << (200 - 192));
    for (const size_t pool_width : { size_t{ 4 }, size_t{ 8 } }) {
        ConcurrencyScope concurrency(pool_width);
        constexpr size_t NUM_FILLERS = 101;
        const size_t filler = std::max(scalar_multiplication::MSM_MIN_PTS_PER_THREAD * pool_width,
                                       ((pool_width * max_wide) / NUM_FILLERS) + 1);
        std::vector<size_t> sizes(NUM_FILLERS, filler);
        sizes.insert(sizes.end(), wide.begin(), wide.end());

        const size_t num_msms = sizes.size();
        std::vector<std::vector<ScalarField>> batch_scalars(num_msms);
        std::vector<PolynomialSpan<ScalarField>> spans;
        std::vector<uint32_t> dedup_infos(num_msms, 0);
        std::vector<AffineElement> expected(num_msms);
        for (size_t k = 0; k < num_msms; ++k) {
            // Fillers cycle through dense, sparse and duplicate-heavy scalars; the wide members are, in order,
            // sparse, duplicate-heavy with a measured count, and dense with a bare dedup hint.
            const size_t kind = k < NUM_FILLERS ? k % 4 : (k - NUM_FILLERS) + 1;
            batch_scalars[k].resize(sizes[k]);
            uint32_t dup_count = 0;
            for (size_t i = 0; i < sizes[k]; ++i) {
                const ScalarField& random = pool[((k * 31) + i) % max_wide];
                if (kind == 1) {
                    batch_scalars[k][i] = (i % 3 == 0) ? random : ScalarField::zero();
                } else if (kind == 2) {
                    batch_scalars[k][i] = ScalarField(high_bit + uint256_t((i / 8) + 1));
                    dup_count += static_cast<uint32_t>(i % 8 != 0);
                } else {
                    batch_scalars[k][i] = random;
                }
            }
            if (kind == 2) {
                dedup_infos[k] = (k >= NUM_FILLERS || k % 8 == 2) ? dup_count : uint32_t{ 1 };
            } else if (kind == 3 && k >= NUM_FILLERS) {
                dedup_infos[k] = 1;
            }
            spans.emplace_back(0, std::span<ScalarField>(batch_scalars[k]));
            expected[k] = scalar_multiplication::MSM<TypeParam>::msm(
                points, PolynomialSpan<const ScalarField>(0, std::span<const ScalarField>(batch_scalars[k])));
        }

        const auto original_scalars = batch_scalars;
        // MSM_fast directly: the MSM facade runs the legacy implementation when BB_MSM_LEGACY is set.
        const auto result = scalar_multiplication::MSM_fast<TypeParam>::batch_multi_scalar_mul(
            points, spans, /*handle_edge_cases=*/false, dedup_infos);
        for (size_t k = 0; k < num_msms; ++k) {
            // Serialized, as commitments reach the transcript and the proof.
            EXPECT_EQ(to_buffer(result[k]), to_buffer(expected[k]))
                << "pool " << pool_width << ", MSM " << k << " (n=" << sizes[k] << ")";
            EXPECT_EQ(batch_scalars[k], original_scalars[k]) << "scalars of MSM " << k << " not restored";
        }
    }
#endif
}

// A batch of three wide members takes the chunked path only where each member's share of the pool is at least 8/3
// threads and its 2^16-point chunks fill the pool: at pool width 8 here (10 minimum chunks, raised to 16 unequal
// parts), not at 4 or 12. Every width must give each member's stand-alone MSM byte for byte and restore its scalars.
TYPED_TEST(BatchMsmConcurrentChunksTest, FewWideMembersMatchStandaloneMsm)
{
#if defined(__wasm__) || defined(NO_MULTITHREADING)
    GTEST_SKIP() << "no concurrent large-member path without threads";
#else
    using AffineElement = typename TypeParam::AffineElement;
    using ScalarField = typename TypeParam::ScalarField;

    constexpr size_t CHUNK = size_t{ 1 } << 16;
    const std::vector<size_t> sizes = { (3 * CHUNK) + 5, (2 * CHUNK) + 777, (3 * CHUNK) - 3 };
    const size_t max_n = *std::max_element(sizes.begin(), sizes.end());
    std::vector<AffineElement> points;
    std::vector<ScalarField> pool;
    random_points_and_scalars<TypeParam>(max_n, points, pool);

    for (const size_t pool_width : { size_t{ 4 }, size_t{ 8 }, size_t{ 12 } }) {
        ConcurrencyScope concurrency(pool_width);
        std::vector<std::vector<ScalarField>> batch_scalars(sizes.size());
        std::vector<PolynomialSpan<ScalarField>> spans;
        std::vector<AffineElement> expected(sizes.size());
        for (size_t k = 0; k < sizes.size(); ++k) {
            // The second member is sparse, so the parts of one batch differ in cost as well as in length.
            batch_scalars[k].resize(sizes[k]);
            for (size_t i = 0; i < sizes[k]; ++i) {
                const ScalarField& random = pool[((k * 17) + i) % max_n];
                batch_scalars[k][i] = (k == 1 && i % 3 != 0) ? ScalarField::zero() : random;
            }
            spans.emplace_back(0, std::span<ScalarField>(batch_scalars[k]));
            expected[k] = scalar_multiplication::MSM<TypeParam>::msm(
                points, PolynomialSpan<const ScalarField>(0, std::span<const ScalarField>(batch_scalars[k])));
        }
        const auto original_scalars = batch_scalars;
        const auto result = scalar_multiplication::MSM_fast<TypeParam>::batch_multi_scalar_mul(
            points, spans, /*handle_edge_cases=*/false);
        for (size_t k = 0; k < sizes.size(); ++k) {
            EXPECT_EQ(to_buffer(result[k]), to_buffer(expected[k]))
                << "pool " << pool_width << ", MSM " << k << " (n=" << sizes[k] << ")";
            EXPECT_EQ(batch_scalars[k], original_scalars[k]) << "scalars of MSM " << k << " not restored";
        }
    }

    // Members of 3, 3 and 2 times 2^16 points at pool width 8 are admitted and cut into exactly eight parts, one per
    // 2^16 points. Member 0's chunks sum to equal points (each chunk doubles the points and halves the scalars),
    // member 1's middle chunk is all zero, and member 2's two chunks sum to opposite points, so its MSM is the point at
    // infinity. The same shape with member 1 reading part of member 0's scalars must not be admitted, and must still
    // give every stand-alone MSM and restore the shared scalars.
    using Element = typename TypeParam::Element;
    ConcurrencyScope concurrency(8);
    std::vector<AffineElement> chunk_points(3 * CHUNK);
    parallel_for_range(CHUNK, [&](size_t start, size_t end) {
        for (size_t j = start; j < end; ++j) {
            Element p(points[j]);
            for (size_t c = 0; c < 3; ++c) {
                chunk_points[(c * CHUNK) + j] = AffineElement(p);
                p = p.dbl();
            }
        }
    });
    const ScalarField half = ScalarField(2).invert();
    std::vector<std::vector<ScalarField>> edge(3);
    edge[0].resize(3 * CHUNK);
    edge[1].resize(3 * CHUNK);
    edge[2].resize(2 * CHUNK);
    for (size_t j = 0; j < CHUNK; ++j) {
        edge[0][j] = pool[j];
        edge[0][CHUNK + j] = pool[j] * half;
        edge[0][(2 * CHUNK) + j] = pool[j] * half * half;
        edge[1][j] = pool[(j + 7) % max_n];
        edge[1][CHUNK + j] = ScalarField::zero();
        edge[1][(2 * CHUNK) + j] = pool[(j + 11) % max_n];
        edge[2][j] = pool[(j + 13) % max_n];
        edge[2][CHUNK + j] = -(edge[2][j] * half);
    }
    auto standalone = [&](std::span<const ScalarField> scalars) {
        return scalar_multiplication::MSM<TypeParam>::msm(
            chunk_points, PolynomialSpan<const ScalarField>(0, scalars), /*handle_edge_cases=*/true);
    };
    {
        std::vector<PolynomialSpan<ScalarField>> spans;
        for (auto& scalars : edge) {
            spans.emplace_back(0, std::span<ScalarField>(scalars));
        }
        const auto original = edge;
        const auto result =
            scalar_multiplication::MSM_fast<TypeParam>::batch_multi_scalar_mul(chunk_points, spans, false);
        const AffineElement first_chunk = standalone(std::span<const ScalarField>(edge[0]).first(CHUNK));
        EXPECT_EQ(to_buffer(result[0]), to_buffer(AffineElement(Element(first_chunk) * ScalarField(3))));
        EXPECT_TRUE(result[2].is_point_at_infinity());
        for (size_t k = 0; k < edge.size(); ++k) {
            EXPECT_EQ(to_buffer(result[k]), to_buffer(standalone(edge[k]))) << "edge-case MSM " << k;
            EXPECT_EQ(edge[k], original[k]) << "scalars of edge-case MSM " << k << " not restored";
        }
    }
    {
        std::vector<PolynomialSpan<ScalarField>> spans;
        spans.emplace_back(0, std::span<ScalarField>(edge[0]));
        spans.emplace_back(0, std::span<ScalarField>(edge[0]).subspan(CHUNK));
        spans.emplace_back(0, std::span<ScalarField>(edge[1]));
        const auto original = edge;
        const auto result =
            scalar_multiplication::MSM_fast<TypeParam>::batch_multi_scalar_mul(chunk_points, spans, false);
        EXPECT_EQ(to_buffer(result[0]), to_buffer(standalone(edge[0])));
        EXPECT_EQ(to_buffer(result[1]), to_buffer(standalone(std::span<const ScalarField>(edge[0]).subspan(CHUNK))));
        EXPECT_EQ(to_buffer(result[2]), to_buffer(standalone(edge[1])));
        EXPECT_EQ(edge, original) << "aliased scalars not restored";
    }
#endif
}
