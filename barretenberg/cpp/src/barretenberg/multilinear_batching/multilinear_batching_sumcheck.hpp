// === AUDIT STATUS ===
// internal:    { status: not started, auditors: [], commit: }
// external_1:  { status: not started, auditors: [], commit: }
// external_2:  { status: not started, auditors: [], commit: }
// =====================
#pragma once

#include "barretenberg/common/assert.hpp"
#include "barretenberg/common/log.hpp"
#include "barretenberg/common/thread.hpp"
#include "barretenberg/numeric/bitop/get_msb.hpp"
#include "barretenberg/polynomials/eq_polynomial.hpp"
#include "barretenberg/polynomials/fold_stride2.hpp"
#include "barretenberg/polynomials/univariate.hpp"
#include "barretenberg/relations/relation_parameters.hpp"
#include "barretenberg/sumcheck/sumcheck_output.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace bb {

/**
 * @brief Sumcheck prover for the MultilinearBatching relation with each claim's eq polynomial kept in factored form.
 *
 * @details Write eq1(r, x) = (1 - r)(1 - x) + r·x and L = log2(n). After the challenges u_0, .., u_{k-1}, the eq table
 * of claim i folded as SumcheckProver folds it is c_{i,k} · eq1(r_{i,k}, X) · E_{i,k}(y), where
 *   c_{i,k} = ∏_{j<k} eq1(r_{i,j}, u_j),   E_{i,k}(y) = ∏_{k<j<L} eq1(r_{i,j}, y_{j-k-1}) · ∏_{j>=L} (1 - r_{i,j}).
 * The round univariate is therefore
 *   s_k(X) = Σ_i γ^i · c_{i,k} · eq1(r_{i,k}, X) · Σ_y E_{i,k}(y) · (P_i(X, y) + α · Q_i(X, y)),
 * and P_i, Q_i are linear in X, so a round needs two sums per polynomial over its own backed rows instead of extending
 * and multiplying eq, P and Q on every edge. Only E_{i,0} (n/2 entries, built by the proving key) is stored, since
 * E_{i,k}(y) = E_{i,0}(y << k) · ∏_{1<=j<=k} (1 - r_{i,j})^{-1}.
 *
 * Every value is an exact field identity, so the sent univariates, the challenges and the claimed evaluations equal
 * those of SumcheckProver<Flavor> on the same claims, at any thread count. The proving key selects this prover only
 * when every 1 - r_{i,j} is invertible and L >= 2 (see ProvingKey::eq_factored).
 */
template <typename Flavor> class MultilinearBatchingSumcheckProver {
  public:
    using FF = typename Flavor::FF;
    using Polynomial = typename Flavor::Polynomial;
    using ProvingKey = typename Flavor::ProvingKey;
    using Transcript = typename Flavor::Transcript;
    using ClaimedEvaluations = typename Flavor::AllValues;
    using RoundUnivariate = Univariate<FF, Flavor::BATCHED_RELATION_PARTIAL_LENGTH>;

    static constexpr size_t NUM_CLAIMS = Flavor::NUM_CLAIMS;
    static constexpr size_t NUM_TABLES = 2 * NUM_CLAIMS;
    static constexpr size_t VIRTUAL_LOG_N = Flavor::VIRTUAL_LOG_N;
    // The edge granularity of the generic FF sumcheck round (64 rows per chunk).
    static constexpr size_t EDGES_PER_CHUNK = 32;
    static_assert(Flavor::BATCHED_RELATION_PARTIAL_LENGTH == 3, "the round univariate is sent at X = 0, 1, 2");

    // Per table t (P_i at t = 2i, Q_i at t = 2i + 1): the sums over y of E·f(2y) and of E·f(2y + 1).
    using RoundSums = std::array<FF, 2 * NUM_TABLES>;

    MultilinearBatchingSumcheckProver(ProvingKey& key,
                                      std::shared_ptr<Transcript> transcript,
                                      const FF& alpha,
                                      const RelationParameters<FF>& relation_parameters)
        : key(key)
        , transcript(std::move(transcript))
        , alpha(alpha)
    {
        BB_ASSERT_EQ(relation_parameters.num_multilinear_batching_challenges,
                     NUM_CLAIMS,
                     "Incorrect number of computed multilinear batching challenges.");
        for (size_t idx = 0; idx < NUM_CLAIMS; ++idx) {
            gamma_powers[idx] = relation_parameters.multilinear_batching_challenges[idx];
        }
    }

    SumcheckOutput<Flavor> prove()
    {
        vinfo("starting sumcheck rounds...");
        const size_t circuit_size = key.circuit_size;
        BB_ASSERT(circuit_size != 0 && (circuit_size & (circuit_size - 1)) == 0,
                  "MultilinearBatchingSumcheckProver: circuit size must be a power of two");
        const size_t log_n = numeric::get_msb(circuit_size);
        BB_ASSERT(key.eq_factored && log_n >= 2,
                  "MultilinearBatchingSumcheckProver: the key holds no eq-suffix tables");

        const auto& claim_challenges = key.polynomials.claim_challenges;
        std::vector<FF> challenge;
        challenge.reserve(VIRTUAL_LOG_N);

        // inverses[idx * (log_n - 1) + j - 1] = (1 - r_{idx,j})^{-1} for 1 <= j < log_n.
        std::vector<FF> inverses(NUM_CLAIMS * (log_n - 1));
        for (size_t idx = 0; idx < NUM_CLAIMS; ++idx) {
            for (size_t j = 1; j < log_n; ++j) {
                inverses[(idx * (log_n - 1)) + j - 1] = FF(1) - claim_challenges[idx][j];
            }
        }
        FF::batch_invert(inverses);

        // eq_prefix[idx] = c_{idx,k}; suffix_scale[idx] = ∏_{1<=j<=k} (1 - r_{idx,j})^{-1}.
        std::array<FF, NUM_CLAIMS> eq_prefix;
        std::array<FF, NUM_CLAIMS> suffix_scale;
        eq_prefix.fill(FF(1));
        suffix_scale.fill(FF(1));

        std::array<const Polynomial*, NUM_TABLES> tables;
        for (size_t idx = 0; idx < NUM_CLAIMS; ++idx) {
            tables[2 * idx] = &key.polynomials.non_shifted(idx);
            tables[(2 * idx) + 1] = &key.polynomials.shifted(idx);
        }
        // Book-keeping tables of P and Q after the first fold, allocated as PartiallyEvaluatedMultivariatesBase does.
        std::array<Polynomial, NUM_TABLES> folded;

        for (size_t round = 0; round < log_n; ++round) {
            if (round > 0) {
                for (size_t idx = 0; idx < NUM_CLAIMS; ++idx) {
                    suffix_scale[idx] *= inverses[(idx * (log_n - 1)) + round - 1];
                }
            }
            const RoundSums sums = compute_round_sums(tables, round);

            std::array<FF, 3> evaluations{ FF(0), FF(0), FF(0) };
            for (size_t idx = 0; idx < NUM_CLAIMS; ++idx) {
                const FF scale = gamma_powers[idx] * eq_prefix[idx] * suffix_scale[idx];
                const FF& r = claim_challenges[idx][round];
                const FF at_0 = sums[4 * idx] + (alpha * sums[(4 * idx) + 2]);
                const FF at_1 = sums[(4 * idx) + 1] + (alpha * sums[(4 * idx) + 3]);
                const FF at_2 = at_1 + at_1 - at_0;
                evaluations[0] += scale * (FF(1) - r) * at_0;
                evaluations[1] += scale * r * at_1;
                evaluations[2] += scale * (r + r + r - FF(1)) * at_2;
            }
            const RoundUnivariate round_univariate(evaluations);
            transcript->send_to_verifier("Sumcheck:univariate_" + std::to_string(round), round_univariate);
            const FF round_challenge = transcript->template get_challenge<FF>("Sumcheck:u_" + std::to_string(round));
            challenge.emplace_back(round_challenge);

            if (round == 0) {
                for (size_t t = 0; t < NUM_TABLES; ++t) {
                    const auto [start, end] = folded_backed_range(*tables[t]);
                    folded[t] = Polynomial(end - start, circuit_size / 2, start, Polynomial::DontZeroMemory::FLAG);
                }
                parallel_for(NUM_TABLES,
                             [&](size_t t) { fold_stride2_backed(*tables[t], folded[t], round_challenge); });
                for (size_t t = 0; t < NUM_TABLES; ++t) {
                    tables[t] = &folded[t];
                }
            } else {
                parallel_for(NUM_TABLES, [&](size_t t) { fold_stride2_backed(folded[t], folded[t], round_challenge); });
            }

            for (size_t idx = 0; idx < NUM_CLAIMS; ++idx) {
                const FF& r = claim_challenges[idx][round];
                eq_prefix[idx] *= ((FF(1) - r) * (FF(1) - round_challenge)) + (r * round_challenge);
            }
        }
        vinfo("completed ", log_n, " rounds of sumcheck");

        // After the real rounds each table has at most one backed entry, at index 0, and eq_i equals c_{i,L} times the
        // trailing factor ∏_{j>=L} (1 - r_{i,j}), which is the generic eq table folded L times.
        std::array<FF, NUM_CLAIMS> non_shifted_value;
        std::array<FF, NUM_CLAIMS> shifted_value;
        std::array<FF, NUM_CLAIMS> eq_value;
        for (size_t idx = 0; idx < NUM_CLAIMS; ++idx) {
            key.eq_suffix_tables[idx] = Polynomial();
            non_shifted_value[idx] = value_at_zero(folded[2 * idx]);
            shifted_value[idx] = value_at_zero(folded[(2 * idx) + 1]);
            FF trailing(1);
            for (size_t j = log_n; j < claim_challenges[idx].size(); ++j) {
                trailing *= FF(1) - claim_challenges[idx][j];
            }
            eq_value[idx] = eq_prefix[idx] * trailing;
        }

        // Virtual rounds: P and Q are padded by zero, so their edge is (p, 0); eq's edge is (e_0, e_1) with e_1 the
        // claim's eq at (u_0, .., u_{k-1}, 1, 0, .., 0), as in Flavor::extend_eq_polynomials_for_virtual_round.
        for (size_t round = log_n; round < VIRTUAL_LOG_N; ++round) {
            std::vector<FF> round_point(VIRTUAL_LOG_N);
            for (size_t j = 0; j < round; ++j) {
                round_point[j] = challenge[j];
            }
            round_point[round] = FF(1);

            std::array<FF, NUM_CLAIMS> eq_at_one;
            std::array<FF, 3> evaluations{ FF(0), FF(0), FF(0) };
            for (size_t idx = 0; idx < NUM_CLAIMS; ++idx) {
                eq_at_one[idx] = VerifierEqPolynomial<FF>::eval(claim_challenges[idx], round_point);
                const FF eq_slope = eq_at_one[idx] - eq_value[idx];
                const FF batched = non_shifted_value[idx] + (alpha * shifted_value[idx]);
                // At X = 0, 1, 2: eq = e_0, e_1, e_1 + slope; the padded P + α·Q = batched, 0, -batched.
                evaluations[0] += gamma_powers[idx] * eq_value[idx] * batched;
                evaluations[2] -= gamma_powers[idx] * (eq_at_one[idx] + eq_slope) * batched;
            }
            const RoundUnivariate round_univariate(evaluations);
            transcript->send_to_verifier("Sumcheck:univariate_" + std::to_string(round), round_univariate);
            const FF round_challenge = transcript->template get_challenge<FF>("Sumcheck:u_" + std::to_string(round));
            challenge.emplace_back(round_challenge);

            for (size_t idx = 0; idx < NUM_CLAIMS; ++idx) {
                non_shifted_value[idx] *= FF(1) - round_challenge;
                shifted_value[idx] *= FF(1) - round_challenge;
                eq_value[idx] = (eq_value[idx] * (FF(1) - round_challenge)) + (eq_at_one[idx] * round_challenge);
            }
        }

        ClaimedEvaluations claimed_evaluations;
        for (size_t idx = 0; idx < NUM_CLAIMS; ++idx) {
            claimed_evaluations.non_shifted(idx) = non_shifted_value[idx];
            claimed_evaluations.shifted(idx) = shifted_value[idx];
            claimed_evaluations.eq(idx) = eq_value[idx];
        }
        transcript->send_to_verifier("Sumcheck:evaluations", claimed_evaluations.get_all());

        vinfo("finished sumcheck");
        return SumcheckOutput<Flavor>{ .challenge = challenge, .claimed_evaluations = claimed_evaluations };
    }

  private:
    ProvingKey& key;
    std::shared_ptr<Transcript> transcript;
    FF alpha;
    std::array<FF, NUM_CLAIMS> gamma_powers;

    static FF value_at_zero(const Polynomial& table) { return table.is_empty() ? FF(0) : table[0]; }

    /**
     * @brief Adds Σ_{y in [first, last)} E(y << shift) · f(2y) to even_sum and the same with f(2y + 1) to odd_sum.
     * @details f is zero outside its backed range, which is read directly; only the edges at an odd start or end have a
     * single backed row.
     */
    static void add_edge_sums(const Polynomial& f,
                              const Polynomial& eq_suffix,
                              const size_t shift,
                              const size_t first,
                              const size_t last,
                              FF& even_sum,
                              FF& odd_sum)
    {
        const size_t start = f.start_index();
        const size_t end = f.end_index();
        const FF* values = f.data();
        const FF* suffix = eq_suffix.data();

        const size_t full_begin = std::max(first, (start + 1) / 2);
        const size_t full_end = std::min(last, end / 2);
        for (size_t y = full_begin; y < full_end; ++y) {
            const FF& weight = suffix[y << shift];
            even_sum += weight * values[(2 * y) - start];
            odd_sum += weight * values[(2 * y) + 1 - start];
        }
        if ((start & 1) != 0 && first <= start / 2 && start / 2 < last) {
            odd_sum += suffix[(start / 2) << shift] * values[0];
        }
        if ((end & 1) != 0 && first <= end / 2 && end / 2 < last) {
            even_sum += suffix[(end / 2) << shift] * values[end - 1 - start];
        }
    }

    /**
     * @brief The round's per-table sums, split into equal shares of work across up to get_num_cpus() slots.
     * @details One unit of work is one edge of one table's backed range; empty tables are never read. Field addition is
     * exact, so the result does not depend on the split.
     */
    RoundSums compute_round_sums(const std::array<const Polynomial*, NUM_TABLES>& tables, const size_t round) const
    {
        std::array<size_t, NUM_TABLES + 1> offsets{};
        for (size_t t = 0; t < NUM_TABLES; ++t) {
            const Polynomial& f = *tables[t];
            const size_t num_edges = f.is_empty() ? 0 : ((f.end_index() + 1) / 2) - (f.start_index() / 2);
            offsets[t + 1] = offsets[t] + num_edges;
        }
        const size_t total = offsets[NUM_TABLES];
        const size_t num_chunks = (total + EDGES_PER_CHUNK - 1) / EDGES_PER_CHUNK;
        const size_t num_slots = std::min(get_num_cpus(), std::max<size_t>(num_chunks, 1));

        RoundSums zero;
        zero.fill(FF(0));
        std::vector<RoundSums> slot_sums(num_slots, zero);
        parallel_for(num_slots, [&](size_t slot) {
            const size_t begin = (total * slot) / num_slots;
            const size_t end = (total * (slot + 1)) / num_slots;
            RoundSums& acc = slot_sums[slot];
            for (size_t t = 0; t < NUM_TABLES; ++t) {
                const size_t lo = std::max(begin, offsets[t]);
                const size_t hi = std::min(end, offsets[t + 1]);
                if (lo >= hi) {
                    continue;
                }
                const Polynomial& f = *tables[t];
                const size_t first_edge = f.start_index() / 2;
                add_edge_sums(f,
                              key.eq_suffix_tables[t / 2],
                              round,
                              first_edge + (lo - offsets[t]),
                              first_edge + (hi - offsets[t]),
                              acc[2 * t],
                              acc[(2 * t) + 1]);
            }
        });

        RoundSums sums = zero;
        for (const RoundSums& acc : slot_sums) {
            for (size_t j = 0; j < sums.size(); ++j) {
                sums[j] += acc[j];
            }
        }
        return sums;
    }
};

} // namespace bb
