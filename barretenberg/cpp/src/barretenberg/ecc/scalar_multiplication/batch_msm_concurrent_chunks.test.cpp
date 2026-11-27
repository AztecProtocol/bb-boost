#include "barretenberg/common/serialize.hpp"
#include "barretenberg/common/thread.hpp"
#include "barretenberg/ecc/curves/bn254/bn254.hpp"
#include "barretenberg/ecc/curves/grumpkin/grumpkin.hpp"
#include "barretenberg/numeric/random/engine.hpp"
#include "barretenberg/polynomials/polynomial.hpp"
#include "scalar_multiplication.hpp"
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
} // namespace

template <class Curve> class BatchMsmConcurrentChunksTest : public ::testing::Test {};

using CurveTypes = ::testing::Types<curve::BN254, curve::Grumpkin>;
TYPED_TEST_SUITE(BatchMsmConcurrentChunksTest, CurveTypes);

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

    // Independent random points and scalars: the affine batch addition assumes no two partial sums in a bucket
    // coincide, which structured points under repeated scalars can violate.
    std::vector<AffineElement> points(max_wide);
    std::vector<ScalarField> pool(max_wide);
    parallel_for_range(max_wide, [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            points[i] = AffineElement::random_element(&engine);
            pool[i] = ScalarField::random_element(&engine);
        }
    });

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
        const auto result = scalar_multiplication::MSM<TypeParam>::batch_multi_scalar_mul(
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
