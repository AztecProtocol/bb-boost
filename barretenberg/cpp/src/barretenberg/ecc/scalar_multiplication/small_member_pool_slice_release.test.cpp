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

template <class Curve> class SmallMemberPoolSliceReleaseTest : public ::testing::Test {};

using CurveTypes = ::testing::Types<curve::BN254, curve::Grumpkin>;
TYPED_TEST_SUITE(SmallMemberPoolSliceReleaseTest, CurveTypes);

// Once the small-member pool's block of per-worker arena slices reaches 64 MiB, each worker releases its slice after
// its last member. Members just below the small-member bound, MSM_MIN_PTS_PER_THREAD * pool width, at a pool wide
// enough to reach that block size, must still match their stand-alone MSMs and get their scalars back. A second batch
// over the same scalars, whose block may land on the released pages, must return the same results. The test checks
// results with the release active on a qualifying block; it does not observe the advice itself, so it would also pass
// if no slice were released.
TYPED_TEST(SmallMemberPoolSliceReleaseTest, SmallMembersMatchStandaloneMsmWhenTheirSlicesAreReleased)
{
#if defined(__wasm__) || defined(NO_MULTITHREADING)
    GTEST_SKIP() << "no concurrent small-member pool without threads";
#else
    using AffineElement = typename TypeParam::AffineElement;
    using ScalarField = typename TypeParam::ScalarField;

    constexpr size_t RELEASE_FLOOR = size_t{ 64 } << 20;
    size_t pool_width = 0;
    size_t n = 0;
    for (const size_t width : { size_t{ 64 }, size_t{ 128 } }) {
        const size_t n_try = (scalar_multiplication::MSM_MIN_PTS_PER_THREAD * width) - 1;
        // The pool sizes its slices by the GLV variant the batch uses; the smaller one bounds both.
        const size_t with_glv = scalar_multiplication::compute_arena_bytes_for_msm<TypeParam>(n_try, true, false, 1);
        const size_t without_glv =
            scalar_multiplication::compute_arena_bytes_for_msm<TypeParam>(n_try, false, false, 1);
        if (std::min(with_glv, without_glv) * width >= RELEASE_FLOOR) {
            pool_width = width;
            n = n_try;
            break;
        }
    }
    ASSERT_GT(pool_width, 0U) << "no pool width reaches the release floor";

    std::vector<AffineElement> points;
    std::vector<ScalarField> pool;
    random_points_and_scalars<TypeParam>(n, points, pool);
    ConcurrencyScope concurrency(pool_width);
    const size_t num_members = pool_width;
    std::vector<std::vector<ScalarField>> scalars(num_members, std::vector<ScalarField>(n));
    std::vector<AffineElement> expected(num_members);
    for (size_t k = 0; k < num_members; ++k) {
        for (size_t i = 0; i < n; ++i) {
            scalars[k][i] = pool[((k * 29) + i) % n];
        }
        expected[k] = scalar_multiplication::MSM<TypeParam>::msm(
            points, PolynomialSpan<const ScalarField>(0, std::span<const ScalarField>(scalars[k])));
    }
    const auto original = scalars;
    for (size_t call = 0; call < 2; ++call) {
        std::vector<PolynomialSpan<ScalarField>> spans;
        for (auto& member : scalars) {
            spans.emplace_back(0, std::span<ScalarField>(member));
        }
        const auto result = scalar_multiplication::MSM_fast<TypeParam>::batch_multi_scalar_mul(
            points, spans, /*handle_edge_cases=*/false);
        for (size_t k = 0; k < num_members; ++k) {
            EXPECT_EQ(to_buffer(result[k]), to_buffer(expected[k])) << "call " << call << ", MSM " << k;
            EXPECT_EQ(scalars[k], original[k]) << "call " << call << ", scalars of MSM " << k << " not restored";
        }
    }
#endif
}
