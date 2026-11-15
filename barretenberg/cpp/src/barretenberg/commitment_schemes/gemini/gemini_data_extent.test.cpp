#include "barretenberg/commitment_schemes/gemini/gemini.hpp"
#include "barretenberg/commitment_schemes/kzg/kzg.hpp"
#include "barretenberg/commitment_schemes/pcs_test_utils.hpp"
#include "barretenberg/commitment_schemes/shplonk/shplemini.hpp"
#include "barretenberg/commitment_schemes/small_subgroup_ipa/small_subgroup_ipa_utils.hpp"
#include "barretenberg/ecc/curves/bn254/bn254.hpp"
#include "barretenberg/transcript/transcript.hpp"

#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <vector>

// A₀ backed on the data extent against the same claims batched without one (A₀ zero-filled to the dyadic size): the
// serialised proof (fold commitments and evaluations, Shplonk and KZG commitments) must be identical, for short, odd
// and full extents, a shifted source, a source only on the masking rows 1-3, with and without virtual rounds and ZK
// claims.
TEST(GeminiDataExtent, BatchMatchesZeroPaddedProof)
{
    using Curve = bb::curve::BN254;
    using FF = bb::fr;
    using Polynomial = bb::Polynomial<FF>;
    using PolynomialBatcher = bb::GeminiProver_<Curve>::PolynomialBatcher;
    constexpr size_t log_n = 5;
    constexpr size_t n = 1UL << log_n;
    const auto ck = bb::create_commitment_key<bb::CommitmentKey<Curve>>(2 * n);

    // The grand-sum slot is opened at 1 against 0, so it is (X - 1) q(X).
    std::array<Polynomial, bb::NUM_SMALL_IPA_COMMITMENTS> libra = { Polynomial::random(8),
                                                                    Polynomial(8),
                                                                    Polynomial::random(8) };
    const Polynomial q = Polynomial::random(7);
    libra[1].at(0) = -q[0];
    for (size_t i = 1; i < 7; ++i) {
        libra[1].at(i) = q[i - 1] - q[i];
    }
    libra[1].at(7) = q[6];

    for (const size_t extent : { size_t{ 3 }, size_t{ 4 }, size_t{ 17 }, size_t{ 31 }, n }) {
        std::vector<Polynomial> polys;
        polys.push_back(Polynomial::random(extent, n, 0));
        polys.push_back(Polynomial::random(std::min<size_t>(extent, 4) - 1, n, 1));
        polys.push_back(Polynomial::random(extent - 1, n, 1));
        for (const size_t virtual_log_n : { log_n, log_n + 2 }) {
            for (const bool zk : { false, true }) {
                std::vector<FF> challenge(virtual_log_n);
                for (auto& u : challenge) {
                    u = FF::random_element();
                }
                auto prove = [&](size_t data_extent) {
                    PolynomialBatcher batcher(n, data_extent);
                    batcher.set_unshifted(bb::RefVector<Polynomial>(polys));
                    batcher.set_to_be_shifted_by_one(bb::RefVector<Polynomial>(polys[2]));
                    auto transcript = bb::NativeTranscript::test_prover_init_empty();
                    const auto claim = bb::ShpleminiProver_<Curve>::prove(
                        n,
                        batcher,
                        challenge,
                        ck,
                        transcript,
                        zk ? libra : std::array<Polynomial, bb::NUM_SMALL_IPA_COMMITMENTS>{});
                    bb::KZG<Curve>::compute_opening_proof(ck, claim, transcript);
                    return transcript->export_proof();
                };
                EXPECT_EQ(prove(extent), prove(n))
                    << "extent " << extent << ", virtual_log_n " << virtual_log_n << ", zk " << zk;
            }
        }
    }
}
