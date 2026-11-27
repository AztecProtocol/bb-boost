#include "barretenberg/multilinear_batching/multilinear_batching_sumcheck.hpp"
#include "barretenberg/common/assert.hpp"
#include "barretenberg/common/serialize.hpp"
#include "barretenberg/common/test.hpp"
#include "barretenberg/common/thread.hpp"
#include "barretenberg/multilinear_batching/multilinear_batching_prover.hpp"
#include "barretenberg/numeric/random/engine.hpp"
#include "barretenberg/sumcheck/sumcheck.hpp"
#include "barretenberg/transcript/transcript.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

using namespace bb;

namespace {

using FF = MultilinearBatchingProverClaim::FF;
using Polynomial = bb::Polynomial<FF>;

// One claim's polynomial shapes: dyadic size 2^log_size, the non-shifted backed range [start, end) (empty when
// start == end and default_empty) and the pre-shift backed range [shift_start, shift_end), shift_start >= 1.
struct ClaimShape {
    size_t log_size;
    size_t start;
    size_t end;
    size_t shift_start;
    size_t shift_end;
    bool default_empty = false;
};

// A coordinate of a claim challenge forced to a fixed value.
struct ForcedChallenge {
    size_t claim;
    size_t coordinate;
    FF value;
};

template <size_t K> using BatchingFlavor = MultilinearBatchingFlavor_<K>;

Polynomial random_backed(size_t start, size_t end, size_t dyadic_size, numeric::RNG& engine)
{
    Polynomial result(end - start, dyadic_size, start);
    for (size_t i = start; i < end; ++i) {
        result.at(i) = FF::random_element(&engine);
    }
    return result;
}

std::vector<MultilinearBatchingProverClaim> make_claims(const std::vector<ClaimShape>& shapes,
                                                        const std::vector<ForcedChallenge>& forced,
                                                        uint64_t seed)
{
    auto& engine = numeric::get_debug_randomness(/*reset=*/true, seed);
    std::vector<MultilinearBatchingProverClaim> claims;
    for (const ClaimShape& shape : shapes) {
        const size_t dyadic_size = size_t{ 1 } << shape.log_size;
        MultilinearBatchingProverClaim claim;
        claim.challenge.resize(CONST_FOLDING_LOG_N);
        for (FF& r : claim.challenge) {
            r = FF::random_element(&engine);
        }
        claim.non_shifted_polynomial =
            shape.default_empty ? Polynomial() : random_backed(shape.start, shape.end, dyadic_size, engine);
        claim.shifted_polynomial = random_backed(shape.shift_start, shape.shift_end, dyadic_size, engine);
        claim.non_shifted_evaluation = FF(0);
        claim.shifted_evaluation = FF(0);
        claim.non_shifted_commitment = MultilinearBatchingProverClaim::Commitment::one();
        claim.shifted_commitment = MultilinearBatchingProverClaim::Commitment::one();
        claim.dyadic_size = dyadic_size;
        claims.push_back(std::move(claim));
    }
    for (const ForcedChallenge& f : forced) {
        claims[f.claim].challenge[f.coordinate] = f.value;
    }
    return claims;
}

// Sets the slot count get_num_cpus() reports, which decides how the factored prover splits a round. The pool's worker
// count is fixed at its first use, so this varies the partition, not the number of running threads.
void set_slots(size_t num_slots)
{
#ifndef NO_MULTITHREADING
    set_parallel_for_concurrency(num_slots);
#else
    static_cast<void>(num_slots);
#endif
}

struct SumcheckRun {
    std::vector<uint8_t> proof_bytes;
    std::vector<uint8_t> output_bytes;
};

// Builds the proving key as the production constructor does, in the factored form (one suffix table per claim) or with
// the full eq tables the generic sumcheck takes. The constructor itself is defined and instantiated only inside the
// prover's translation unit, so the test reaches it through MultilinearBatchingProver alone.
template <size_t K>
typename BatchingFlavor<K>::ProvingKey build_key(std::vector<MultilinearBatchingProverClaim>&& claims, bool factored)
{
    using F = BatchingFlavor<K>;
    typename F::ProvingKey key;
    for (const auto& claim : claims) {
        key.circuit_size = std::max(key.circuit_size, claim.dyadic_size);
    }
    const size_t log_n = numeric::get_msb(key.circuit_size);
    key.eq_factored = factored;
    for (size_t idx = 0; idx < K; ++idx) {
        auto& claim = claims[idx];
        key.polynomials.non_shifted(idx) = std::move(claim.non_shifted_polynomial);
        key.preshifted_polynomials[idx] = std::move(claim.shifted_polynomial);
        key.polynomials.shifted(idx) = key.preshifted_polynomials[idx].shifted();
        if (factored) {
            key.eq_suffix_tables[idx] =
                ProverEqPolynomial<FF>::construct(std::span<const FF>(claim.challenge).subspan(1), log_n - 1);
        } else {
            key.polynomials.eq(idx) = ProverEqPolynomial<FF>::construct(claim.challenge, log_n);
        }
        key.polynomials.claim_challenges[idx] = std::move(claim.challenge);
    }
    key.polynomials.increase_polynomials_virtual_size(size_t{ 1 } << F::VIRTUAL_LOG_N);
    return key;
}

std::shared_ptr<NativeTranscript> seeded_transcript()
{
    auto transcript = std::make_shared<NativeTranscript>();
    transcript->send_to_verifier("seed", FF(7));
    return transcript;
}

// Runs the batching sumcheck on claims regenerated from the seed, with the given slot count, on the factored prover or
// on the generic SumcheckProver, drawing the challenges MultilinearBatchingProver draws before it.
template <size_t K>
SumcheckRun run_sumcheck(const std::vector<ClaimShape>& shapes,
                         const std::vector<ForcedChallenge>& forced,
                         uint64_t seed,
                         bool factored,
                         size_t num_slots)
{
    using F = BatchingFlavor<K>;
    set_slots(num_slots);
    typename F::ProvingKey key = build_key<K>(make_claims(shapes, forced, seed), factored);

    auto transcript = seeded_transcript();
    const FF gamma = transcript->template get_challenge<FF>("claim_batching_challenge");
    const FF alpha = transcript->template get_challenge<FF>("Sumcheck:alpha");
    RelationParameters<FF> relation_parameters;
    relation_parameters.compute_multilinear_batching_challenges(gamma, K);

    std::optional<SumcheckOutput<F>> output;
    if (factored) {
        MultilinearBatchingSumcheckProver<F> sumcheck(key, transcript, alpha, relation_parameters);
        output = sumcheck.prove();
    } else {
        SumcheckProver<F> sumcheck(
            key.circuit_size, key.polynomials, transcript, alpha, {}, relation_parameters, F::VIRTUAL_LOG_N);
        output = sumcheck.prove();
    }

    SumcheckRun run;
    run.proof_bytes = to_buffer(transcript->export_proof());
    std::vector<FF> output_fields = output->challenge;
    for (const FF& eval : output->claimed_evaluations.get_all()) {
        output_fields.push_back(eval);
    }
    run.output_bytes = to_buffer(output_fields);
    return run;
}

// The production prover end to end: its key constructor (and guard), whichever sumcheck the guard selects, then the
// claim merge, which needs the largest claim's backed range to contain every other claim's.
std::vector<uint8_t> prover_proof(const std::vector<ClaimShape>& shapes,
                                  const std::vector<ForcedChallenge>& forced,
                                  uint64_t seed,
                                  size_t num_slots)
{
    set_slots(num_slots);
    MultilinearBatchingProver prover(make_claims(shapes, forced, seed), seeded_transcript());
    return to_buffer(prover.construct_proof());
}

// The factored prover's proof and output must equal the generic prover's byte for byte at every slot count, and so must
// the production prover's proof when the shapes satisfy its merge.
template <size_t K>
void expect_identical(const std::vector<ClaimShape>& shapes,
                      const std::vector<ForcedChallenge>& forced = {},
                      bool check_prover = true)
{
    ASSERT_EQ(shapes.size(), K);
    const size_t default_slots = get_num_cpus();
    const uint64_t seed = 0xb1a5ed + K;
    const SumcheckRun reference = run_sumcheck<K>(shapes, forced, seed, /*factored=*/false, default_slots);
    for (const size_t num_slots : { size_t{ 1 }, size_t{ 3 }, size_t{ 4 }, size_t{ 8 } }) {
        const SumcheckRun generic = run_sumcheck<K>(shapes, forced, seed, /*factored=*/false, num_slots);
        const SumcheckRun factored = run_sumcheck<K>(shapes, forced, seed, /*factored=*/true, num_slots);
        EXPECT_EQ(generic.proof_bytes, reference.proof_bytes) << "generic, slots " << num_slots;
        EXPECT_EQ(factored.proof_bytes, reference.proof_bytes) << "factored, slots " << num_slots;
        EXPECT_EQ(factored.output_bytes, reference.output_bytes) << "factored, slots " << num_slots;
        if (check_prover) {
            EXPECT_EQ(prover_proof(shapes, forced, seed, num_slots), reference.proof_bytes)
                << "prover, slots " << num_slots;
        }
    }
    set_slots(default_slots);
}

} // namespace

// Mixed dyadic sizes; odd starts and ends; support up to n - 1; a pre-shift range ending at n.
TEST(MultilinearBatchingSumcheck, MixedShapesMatchGeneric)
{
    expect_identical<3>({ { .log_size = 9, .start = 0, .end = 511, .shift_start = 1, .shift_end = 512 },
                          { .log_size = 5, .start = 3, .end = 29, .shift_start = 5, .shift_end = 31 },
                          { .log_size = 7, .start = 1, .end = 100, .shift_start = 2, .shift_end = 101 } });
}

// The smallest factored size, L = 2.
TEST(MultilinearBatchingSumcheck, TwoRealRoundsMatchGeneric)
{
    expect_identical<2>({ { .log_size = 2, .start = 0, .end = 4, .shift_start = 1, .shift_end = 4 },
                          { .log_size = 2, .start = 1, .end = 3, .shift_start = 1, .shift_end = 2 } });
}

// Seven claims, one with a default-empty non-shifted polynomial (virtual size 0, never read) and one whose pre-shift
// polynomial is backed on nothing but has a positive virtual size. No claim's range contains all the others, so the
// production prover's merge rejects these shapes and only the sumcheck provers are compared.
TEST(MultilinearBatchingSumcheck, EmptyPolynomialsMatchGeneric)
{
    expect_identical<7>(
        { { .log_size = 8, .start = 0, .end = 200, .shift_start = 1, .shift_end = 201 },
          { .log_size = 6, .start = 0, .end = 0, .shift_start = 1, .shift_end = 40, .default_empty = true },
          { .log_size = 8, .start = 7, .end = 256, .shift_start = 1, .shift_end = 1 },
          { .log_size = 3, .start = 0, .end = 8, .shift_start = 1, .shift_end = 8 },
          { .log_size = 8, .start = 0, .end = 1, .shift_start = 255, .shift_end = 256 },
          { .log_size = 5, .start = 31, .end = 32, .shift_start = 1, .shift_end = 2 },
          { .log_size = 7, .start = 64, .end = 65, .shift_start = 63, .shift_end = 66 } },
        {},
        /*check_prover=*/false);
}

// r_j = 0 in a real coordinate (j < L) and in a trailing one (j >= L) keeps the factored path exact.
TEST(MultilinearBatchingSumcheck, ZeroChallengeCoordinatesMatchGeneric)
{
    expect_identical<2>({ { .log_size = 6, .start = 0, .end = 64, .shift_start = 1, .shift_end = 64 },
                          { .log_size = 6, .start = 2, .end = 61, .shift_start = 3, .shift_end = 63 } },
                        { { .claim = 0, .coordinate = 0, .value = FF(0) },
                          { .claim = 0, .coordinate = 3, .value = FF(0) },
                          { .claim = 1, .coordinate = 10, .value = FF(0) } });
}

// For r_j = 1 the key's guard selects the generic eq tables, whose construction rejects a challenge longer than L (as
// it does without the factored prover). The guard's L < 2 branch is not run: at n = 2 the generic sumcheck itself
// rejects the input in its first virtual round.
TEST(MultilinearBatchingSumcheck, GuardSelectsGenericPath)
{
    // Coordinate 0 is not a factor of the suffix table, so only the key's guard routes it to the generic tables.
    for (const size_t coordinate : { size_t{ 0 }, size_t{ 2 }, size_t{ 12 } }) {
        auto claims = make_claims({ { .log_size = 5, .start = 0, .end = 32, .shift_start = 1, .shift_end = 32 },
                                    { .log_size = 5, .start = 0, .end = 20, .shift_start = 1, .shift_end = 20 } },
                                  { { .claim = 1, .coordinate = coordinate, .value = FF(1) } },
                                  /*seed=*/2);
        EXPECT_THROW_OR_ABORT(MultilinearBatchingProver(std::move(claims), seeded_transcript()).construct_proof(),
                              ".*");
    }
}
