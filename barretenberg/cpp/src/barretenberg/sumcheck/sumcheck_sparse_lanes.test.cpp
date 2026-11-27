#include "barretenberg/flavor/mega_flavor.hpp"
#include "barretenberg/sumcheck/sumcheck_round.hpp"

#include <gtest/gtest.h>

#include <map>

using namespace bb;

namespace {

using Flavor = MegaFlavor;
using FF = Flavor::FF;
using EntityId = Flavor::EntityId;
using SelectorRows = std::map<EntityId, std::vector<size_t>>;

static_assert(HasSparseSelector<Poseidon2ExternalRelation<FF>>);
static_assert(HasSparseSelector<Poseidon2InitialExternalRelation<FF>>);
static_assert(HasSparseSelector<Poseidon2QuadInternalRelation<FF>>);
static_assert(HasSparseSelector<Poseidon2QuadInternalTerminalRelation<FF>>);
static_assert(HasSparseSelector<Poseidon2TransitionEntryRelation<FF>>);

// Selector rows of one Poseidon2 permutation in Mega's `poseidon2` block, starting at `base`: initial linear layer,
// four external rounds, transition entry, thirteen quad-internal rows, terminal, four external rounds, output row.
void add_permutation_rows(SelectorRows& rows, const size_t base)
{
    rows[EntityId::q_poseidon2_external_initial].push_back(base);
    for (size_t i = 0; i < 4; ++i) {
        rows[EntityId::q_poseidon2_external].push_back(base + 1 + i);
        rows[EntityId::q_poseidon2_external].push_back(base + 20 + i);
    }
    rows[EntityId::q_poseidon2_transition_entry].push_back(base + 5);
    for (size_t i = 0; i < 13; ++i) {
        rows[EntityId::q_poseidon2_quad_internal].push_back(base + 6 + i);
    }
    rows[EntityId::q_poseidon2_quad_internal_terminal].push_back(base + 19);
}

/**
 * @brief The SIMD-element round univariate must equal the scalar one when the Poseidon2 selectors are sparse.
 * @details Every entity is random except the Poseidon2 selectors, which are random on `selector_rows` and zero
 * elsewhere, so the 10-row groups of the `VectorField` path see one live edge (scalar evaluation of that lane), several
 * (packed evaluation) or none (skip). On native `VectorField` is an emulation, so this checks the lane orchestration,
 * not the SIMD kernels.
 */
void check_sparse_selectors_match_scalar(const SelectorRows& selector_rows, const size_t log_n)
{
    const size_t round_size = size_t{ 1 } << log_n;
    typename Flavor::ProverPolynomials prover_polynomials;
    for (size_t entity = 0; entity < Flavor::NUM_ALL_ENTITIES; ++entity) {
        Polynomial<FF> poly(round_size);
        const auto sparse = selector_rows.find(static_cast<EntityId>(entity));
        if (sparse != selector_rows.end()) {
            for (const size_t row : sparse->second) {
                poly.at(row) = FF::random_element();
            }
        } else {
            for (size_t row = 0; row < round_size; ++row) {
                poly.at(row) = FF::random_element();
            }
        }
        prover_polynomials.get_all()[entity] = poly.share();
    }

    const auto relation_parameters = RelationParameters<FF>::get_random();
    std::vector<FF> betas(log_n);
    for (auto& beta : betas) {
        beta = FF::random_element();
    }
    GateSeparatorPolynomial<FF> gate_separators(betas, log_n);
    std::array<FF, Flavor::NUM_SUBRELATIONS - 1> alphas;
    for (auto& alpha : alphas) {
        alpha = FF::random_element();
    }

    SumcheckProverRound<Flavor> round_scalar(round_size);
    const auto result_scalar =
        round_scalar.template compute_univariate<FF>(prover_polynomials, relation_parameters, gate_separators, alphas);
    SumcheckProverRound<Flavor> round_vec(round_size);
    const auto result_vec = round_vec.template compute_univariate<VectorField<FF::Params>>(
        prover_polynomials, relation_parameters, gate_separators, alphas);

    EXPECT_EQ(result_scalar, result_vec);
}

} // namespace

// Hand-placed rows in a 32-row round (10-row groups at rows 0, 10, 20; scalar tail at rows 30-31): every Poseidon2
// relation is live on a single edge of some group, External and QuadInternal also on several edges of others, and
// External and ExtInitial in the tail.
TEST(SumcheckSparseLanes, SingleAndMultiLaneGroupsMatchScalarMega)
{
    const SelectorRows rows{
        { EntityId::q_poseidon2_quad_internal_terminal, { 3 } },
        { EntityId::q_poseidon2_transition_entry, { 12, 13, 25 } },
        { EntityId::q_poseidon2_external_initial, { 17, 31 } },
        { EntityId::q_poseidon2_external, { 1, 2, 30 } },
        { EntityId::q_poseidon2_quad_internal, { 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 28 } },
    };
    check_sparse_selectors_match_scalar(rows, 5);
}

// Back-to-back permutations in Mega's row layout across a 128-row round (several work-stealing chunks).
TEST(SumcheckSparseLanes, PermutationLayoutMatchesScalarMega)
{
    SelectorRows rows;
    for (size_t base = 3; base + 25 <= 128; base += 25) {
        add_permutation_rows(rows, base);
    }
    check_sparse_selectors_match_scalar(rows, 7);
}

// No Poseidon2 row at all: every group skips the five relations.
TEST(SumcheckSparseLanes, NoPoseidon2RowsMatchesScalarMega)
{
    SelectorRows rows;
    for (const auto id : { EntityId::q_poseidon2_external,
                           EntityId::q_poseidon2_external_initial,
                           EntityId::q_poseidon2_quad_internal,
                           EntityId::q_poseidon2_quad_internal_terminal,
                           EntityId::q_poseidon2_transition_entry }) {
        rows[id] = {};
    }
    check_sparse_selectors_match_scalar(rows, 5);
}
