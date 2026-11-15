#include "barretenberg/numeric/bitop/get_msb.hpp"
#include "barretenberg/polynomials/gate_separator.hpp"
#include "barretenberg/relations/relation_parameters.hpp"
#include "barretenberg/relations/utils.hpp"
#include "barretenberg/sumcheck/sumcheck_round.hpp"
#include "barretenberg/vm2/constraining/flavor.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

namespace bb::avm2::constraining {
namespace {

using FF = AvmFlavor::FF;
using Polynomial = AvmFlavor::Polynomial;
using Round = SumcheckProverRound<AvmFlavor>;
using Utils = RelationUtils<AvmFlavor>;

constexpr size_t ROUND_SIZE = 1024;

// The chunks AVM's round scheduler hands out, in pop order, for a round whose witnesses end at `witness_end`.
std::vector<std::pair<size_t, size_t>> scheduled_chunks(size_t witness_end)
{
    AvmFlavor::ProverPolynomials polynomials;
    polynomials.get_witness()[0] = Polynomial(witness_end, ROUND_SIZE);
    const auto round = std::make_unique<Round>(ROUND_SIZE);
    auto chunks = round->make_edge_chunks(polynomials);
    std::vector<std::pair<size_t, size_t>> popped;
    while (auto chunk = chunks.pop()) {
        popped.emplace_back(chunk->begin, chunk->end);
    }
    return popped;
}

// The first 512 rows go out as one-edge chunks, then 16-row chunks to the round's end; every edge exactly once.
TEST(AvmSumcheckDenseHeadTest, SchedulesTheHeadAsOneEdgeChunksThenSixteenRowChunks)
{
    for (const size_t witness_end : { size_t{ 16 }, size_t{ 999 } }) {
        const size_t end = witness_end + (witness_end % 2);
        std::vector<std::pair<size_t, size_t>> expected;
        size_t row = 0;
        for (; row < std::min<size_t>(512, end); row += 2) {
            expected.emplace_back(row, row + 2);
        }
        for (; row < end; row += 16) {
            expected.emplace_back(row, std::min(row + 16, end));
        }
        EXPECT_EQ(scheduled_chunks(witness_end), expected) << "witness end " << witness_end;
    }
    EXPECT_TRUE(scheduled_chunks(0).empty());
}

std::vector<FF> flatten(Round::SumcheckTupleOfTuplesOfUnivariates& accumulators)
{
    std::vector<FF> evaluations;
    Utils::apply_to_tuple_of_tuples(accumulators, [&](const auto& univariate) {
        evaluations.insert(evaluations.end(), univariate.evaluations.begin(), univariate.evaluations.end());
    });
    return evaluations;
}

// Equivalence check against plain 16-row chunks, so it also passes when the production scheduler has no one-edge head:
// the head changes no accumulator as a field element. Columns are random prefixes of varying length, so both the head
// and the tail meet live and zero edges.
TEST(AvmSumcheckDenseHeadTest, HeadChunkingLeavesTheRoundAccumulatorsUnchanged)
{
    constexpr size_t END = 1000;
    AvmFlavor::ProverPolynomials polynomials;
    size_t column = 0;
    for (auto& polynomial : polynomials.get_all()) {
        const size_t length = 1 + ((column++ * 7919) % END);
        polynomial = Polynomial::random(length, ROUND_SIZE, 0);
    }
    polynomials.get_witness()[0] = Polynomial::random(END, ROUND_SIZE, 0);

    std::vector<FF> betas(numeric::get_msb(ROUND_SIZE));
    for (auto& beta : betas) {
        beta = FF::random_element();
    }
    const GateSeparatorPolynomial<FF> gate_separators(betas, betas.size());
    Round::SubrelationSeparators alphas;
    for (auto& alpha : alphas) {
        alpha = FF::random_element();
    }
    const auto relation_parameters = RelationParameters<FF>::get_random();
    const auto batching = Round::compute_subrelation_batching(alphas);

    const auto with_head = std::make_unique<Round>(ROUND_SIZE);
    auto head_chunks = with_head->make_edge_chunks(polynomials);
    with_head->accumulate_edge_chunks<FF>(head_chunks, polynomials, relation_parameters, gate_separators, &batching);

    const auto without_head = std::make_unique<Round>(ROUND_SIZE);
    Round::ContiguousEdgeChunks plain_chunks{ 0, END, Round::ROWS_PER_CHUNK };
    without_head->accumulate_edge_chunks<FF>(
        plain_chunks, polynomials, relation_parameters, gate_separators, &batching);

    const auto expected = flatten(without_head->univariate_accumulators);
    const auto actual = flatten(with_head->univariate_accumulators);
    ASSERT_EQ(actual.size(), expected.size());
    size_t mismatches = 0;
    size_t nonzero = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        mismatches += actual[i] == expected[i] ? 0 : 1;
        nonzero += expected[i].is_zero() ? 0 : 1;
    }
    EXPECT_EQ(mismatches, 0U);
    EXPECT_GT(nonzero, 0U);
}

} // namespace
} // namespace bb::avm2::constraining
