#include "barretenberg/common/assert.hpp"
#include "barretenberg/common/ref_vector.hpp"
#include "barretenberg/srs/factories/mem_bn254_crs_factory.hpp"
#include "barretenberg/srs/factories/native_crs_factory.hpp"
#include "barretenberg/srs/global_crs.hpp"
#include "barretenberg/vm2/common/constants.hpp"
#include "barretenberg/vm2/constraining/flavor.hpp"
#include "barretenberg/vm2/constraining/polynomials.hpp"
#include "barretenberg/vm2/testing/fixtures.hpp"

#include <gtest/gtest.h>

#include <string>

namespace bb::avm2::constraining {
namespace {

using CommitmentKey = AvmFlavor::CommitmentKey;
using Polynomial = AvmFlavor::Polynomial;

// A commitment key holding exactly the points asked for. The global factory keeps the largest CRS any earlier test
// loaded, and since bb v6.1.0 every load from the file is rounded up to whole pinned chunks
// (srs::SRS_CHUNK_SIZE_POINTS), so either would hold more points than asked for: the key gets an in-memory copy of the
// first num_points points of a load of its own.
class IsolatedCommitmentKey : public CommitmentKey {
  public:
    explicit IsolatedCommitmentKey(size_t num_points)
    {
        bb::srs::factories::NativeBn254CrsFactory factory(bb::srs::bb_crs_path(), /*allow_download=*/false);
        const auto loaded = factory.get_crs(num_points);
        const auto points = loaded->get_monomial_points().first(num_points);
        bb::srs::factories::MemBn254CrsFactory exact(
            std::vector<curve::BN254::AffineElement>(points.begin(), points.end()), loaded->get_g2x());
        srs = exact.get_crs(num_points);
        srs_size = num_points;
    }
};

TEST(AvmCommitmentKeySizeTest, PowerOfTwoCoveringTheLongestColumn)
{
    EXPECT_EQ(commitment_key_size(0), size_t{ 1 });
    EXPECT_EQ(commitment_key_size(1), size_t{ 1 });
    EXPECT_EQ(commitment_key_size(size_t{ 1 } << 16), size_t{ 1 } << 16);
    EXPECT_EQ(commitment_key_size((size_t{ 1 } << 16) + 1), size_t{ 1 } << 17);
    EXPECT_EQ(commitment_key_size(MAX_AVM_TRACE_SIZE), MAX_AVM_TRACE_SIZE);
}

// The prover loads commitment_key_size(trace rows) points before the trace becomes polynomials. Exactly those points
// must cover every polynomial, so that the proving key keeps them instead of loading again, and a small trace must load
// far fewer points than the full circuit's.
TEST(AvmCommitmentKeySizeTest, PointsLoadedForTheLongestColumnCoverTheProvingKey)
{
    auto [trace, public_inputs] = testing::get_minimal_trace_with_pi();
    const size_t crs_points = commitment_key_size(trace.get_num_rows());
    const IsolatedCommitmentKey prefetched(crs_points);
    ASSERT_EQ(prefetched.get_monomial_size(), crs_points);

    auto polynomials = compute_polynomials(trace);
    const auto proving_key = proving_key_from_polynomials(polynomials, prefetched);

    EXPECT_EQ(proving_key->commitment_key.get_monomial_points().data(), prefetched.get_monomial_points().data());
    EXPECT_EQ(proving_key->commitment_key.get_monomial_size(), crs_points);
    EXPECT_LT(crs_points, MAX_AVM_TRACE_SIZE);
}

// The reload goes through the global factory, as in the prover; it raises that cache to at most this trace's points.
TEST(AvmCommitmentKeySizeTest, ProvingKeyLoadsThePointsAKeyTooSmallLacks)
{
    auto [trace, public_inputs] = testing::get_minimal_trace_with_pi();
    const size_t crs_points = commitment_key_size(trace.get_num_rows());
    const IsolatedCommitmentKey too_small(crs_points / 2);
    ASSERT_LT(too_small.get_monomial_size(), crs_points);

    auto polynomials = compute_polynomials(trace);
    srs::init_file_crs_factory(srs::bb_crs_path());
    const auto proving_key = proving_key_from_polynomials(polynomials, too_small);

    EXPECT_NE(proving_key->commitment_key.get_monomial_points().data(), too_small.get_monomial_points().data());
    EXPECT_GE(proving_key->commitment_key.get_monomial_size(), crs_points);
}

// Commitments check the points actually loaded: a single coefficient at offset `loaded` needs one point more.
TEST(AvmCommitmentKeySizeTest, CommitsBeyondTheLoadedPointsAbort)
{
    const IsolatedCommitmentKey commitment_key(16);
    const size_t loaded = commitment_key.get_monomial_size();
    ASSERT_EQ(loaded, size_t{ 16 });
    Polynomial fits = Polynomial::random(loaded);
    Polynomial past_the_end(/*size=*/1, /*virtual_size=*/loaded + 1, /*start_index=*/loaded);
    past_the_end.at(loaded) = AvmFlavor::FF(1);
    const std::string message = "needs " + std::to_string(loaded + 1) + " points";

    EXPECT_NO_THROW(static_cast<void>(commitment_key.commit(fits)));
    EXPECT_THROW_OR_ABORT(static_cast<void>(commitment_key.commit(past_the_end)), message);
    EXPECT_THROW_OR_ABORT(static_cast<void>(commitment_key.batch_commit(RefVector<Polynomial>{ fits, past_the_end })),
                          message);
}

} // namespace
} // namespace bb::avm2::constraining
