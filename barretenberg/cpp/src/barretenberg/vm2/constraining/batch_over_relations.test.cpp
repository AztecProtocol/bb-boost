#include "barretenberg/relations/utils.hpp"
#include "barretenberg/sumcheck/sumcheck_round.hpp"
#include "barretenberg/vm2/constraining/flavor.hpp"
#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <utility>

namespace bb::avm2 {
namespace {

using FF = AvmFlavor::FF;
using Round = SumcheckProverRound<AvmFlavor>;
using Utils = RelationUtils<AvmFlavor>;
using Accumulators = Round::SumcheckTupleOfTuplesOfUnivariates;
using RoundUnivariate = Round::SumcheckRoundUnivariate;

bool same_limbs(const FF& a, const FF& b)
{
    return std::memcmp(a.data, b.data, sizeof(a.data)) == 0;
}

// The batched round univariate, limbs included, and the zeroed accumulators must be those of scaling the whole tuple,
// batching it untouched, then zeroing it.
TEST(AvmBatchOverRelations, MatchesScalingAndZeroingAroundTheBatch)
{
    auto accumulators = std::make_unique<Accumulators>();
    Utils::apply_to_tuple_of_tuples(
        *accumulators, [](auto& univariate) { univariate = std::decay_t<decltype(univariate)>::get_random(); });
    auto reference = std::make_unique<Accumulators>(*accumulators);

    Round::SubrelationSeparators alphas;
    for (auto& alpha : alphas) {
        alpha = FF::random_element();
    }
    const GateSeparatorPolynomial<FF> gate_separators(std::vector<FF>{ FF::random_element() }, 1);

    Utils::scale_univariates(*reference, alphas);
    RoundUnivariate expected(0);
    Round::extend_and_batch_univariates(std::as_const(*reference), expected, gate_separators);

    const auto result = Round::batch_over_relations<RoundUnivariate>(*accumulators, alphas, gate_separators);

    for (size_t k = 0; k < RoundUnivariate::LENGTH; ++k) {
        EXPECT_TRUE(same_limbs(result.evaluations[k], expected.evaluations[k])) << "evaluation " << k;
    }
    size_t nonzero = 0;
    Utils::apply_to_tuple_of_tuples(*accumulators, [&](const auto& univariate) {
        for (const auto& evaluation : univariate.evaluations) {
            nonzero += same_limbs(evaluation, FF(0)) ? 0 : 1;
        }
    });
    EXPECT_EQ(nonzero, 0U);
}

} // namespace
} // namespace bb::avm2
