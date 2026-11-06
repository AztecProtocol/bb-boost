// === AUDIT STATUS ===
// internal:    { status: Completed, auditors: [Federico], commit: 0e37cb8}
// external_1:  { status: not started, auditors: [], commit: }
// external_2:  { status: not started, auditors: [], commit: }
// =====================
#include "flavor.hpp"
#include "barretenberg/common/assert.hpp"
#include "barretenberg/flavor/flavor_concepts.hpp"

namespace bb::avm2 {

AvmFlavor::ProverPolynomials::ProverPolynomials(ProvingKey& proving_key)
{
    for (auto [prover_poly, key_poly] : zip_view(this->get_unshifted(), proving_key.get_all())) {
        BB_ASSERT_DEBUG(flavor_get_label(*this, prover_poly) == flavor_get_label(proving_key, key_poly));
        prover_poly = key_poly.share();
    }
    for (auto [prover_poly, key_poly] : zip_view(this->get_shifted(), proving_key.get_to_be_shifted())) {
        BB_ASSERT_DEBUG(flavor_get_label(*this, prover_poly) == flavor_get_label(proving_key, key_poly) + "_shift");
        prover_poly = key_poly.shifted();
    }
}

// Deserializes the proof from proof_data. Length must match COMPUTED_AVM_PROOF_LENGTH_IN_FIELDS in flavor.hpp.
void AvmFlavor::Transcript::deserialize_full_transcript()
{
    size_t num_frs_read = 0;

    for (auto& commitment : commitments) {
        commitment = deserialize_from_buffer<Commitment>(proof_data, num_frs_read);
    }

    for (size_t i = 0; i < log_circuit_size; ++i) {
        sumcheck_univariates.emplace_back(deserialize_from_buffer<bb::Univariate<FF, BATCHED_RELATION_PARTIAL_LENGTH>>(
            Transcript::proof_data, num_frs_read));
    }

    sumcheck_evaluations =
        deserialize_from_buffer<std::array<FF, NUM_ALL_ENTITIES>>(Transcript::proof_data, num_frs_read);

    for (size_t i = 0; i < log_circuit_size - 1; ++i) {
        gemini_fold_comms.push_back(deserialize_from_buffer<Commitment>(proof_data, num_frs_read));
    }

    for (size_t i = 0; i < log_circuit_size; ++i) {
        gemini_fold_evals.push_back(deserialize_from_buffer<FF>(proof_data, num_frs_read));
    }

    shplonk_q_comm = deserialize_from_buffer<Commitment>(proof_data, num_frs_read);

    kzg_w_comm = deserialize_from_buffer<Commitment>(proof_data, num_frs_read);
}

// Serializes the proof to proof_data. Length must match COMPUTED_AVM_PROOF_LENGTH_IN_FIELDS in flavor.hpp.
void AvmFlavor::Transcript::serialize_full_transcript()
{
    size_t old_proof_length = proof_data.size();
    Transcript::proof_data.clear();

    for (const auto& commitment : commitments) {
        serialize_to_buffer(commitment, Transcript::proof_data);
    }

    for (size_t i = 0; i < log_circuit_size; ++i) {
        serialize_to_buffer(sumcheck_univariates[i], Transcript::proof_data);
    }

    serialize_to_buffer(sumcheck_evaluations, Transcript::proof_data);

    for (size_t i = 0; i < log_circuit_size - 1; ++i) {
        serialize_to_buffer(gemini_fold_comms[i], proof_data);
    }

    for (size_t i = 0; i < log_circuit_size; ++i) {
        serialize_to_buffer(gemini_fold_evals[i], proof_data);
    }

    serialize_to_buffer(shplonk_q_comm, proof_data);
    serialize_to_buffer(kzg_w_comm, proof_data);

    // sanity check to make sure we generate the same length of proof as before.
    BB_ASSERT_EQ(proof_data.size(), old_proof_length);
}

AvmFlavor::ProverPolynomials::ProverPolynomials(const ProverPolynomials& full_polynomials, size_t circuit_size)
{
    for (auto [poly, full_poly] : zip_view(get_all(), full_polynomials.get_all())) {
        // After the initial sumcheck round, the new size is CEIL(size/2).
        size_t desired_size = (full_poly.end_index() / 2) + (full_poly.end_index() % 2);
        // The first fold overwrites [floor(start/2), ceil(end/2)). Avoid zeroing
        // when that covers this entire allocation. Keep zero-filled leading holes
        // (and the original empty-source behavior) for other backing ranges.
        if (!full_poly.is_empty() && full_poly.start_index() < 2) {
            poly = Polynomial(desired_size, circuit_size / 2, Polynomial::DontZeroMemory::FLAG);
        } else {
            poly = Polynomial(desired_size, circuit_size / 2);
        }
    }
}

AvmFlavor::ProvingKey::ProvingKey()
    : commitment_key(circuit_size) {
        // The proving key's polynomials are not allocated here because they are later overwritten
        // AvmComposer::compute_witness(). We should probably refactor this flow.
    };

void AvmFlavor::LazilyExtendedProverUnivariates::set_current_edge(size_t edge_idx)
{
    current_edge = edge_idx;
    valid.reset();
    used = 0;
}

const bb::Univariate<AvmFlavor::FF, AvmFlavor::MAX_PARTIAL_RELATION_LENGTH>& AvmFlavor::
    LazilyExtendedProverUnivariates::get(ColumnAndShifts c) const
{
    const auto column = static_cast<size_t>(c);
    if (valid[column]) {
        return *cached[column];
    }
    const auto& multivariate = multivariates.get(c);
    if (multivariate.is_empty() || multivariate.end_index() < current_edge) {
        static const auto zero_univariate = bb::Univariate<FF, MAX_PARTIAL_RELATION_LENGTH>::zero();
        cached[column] = &zero_univariate;
    } else {
        const auto& val0 = multivariate[current_edge];
        const auto& val1 = multivariate[current_edge + 1];
        if (val0.is_zero() && val1.is_zero()) {
            static const auto zero_univariate = bb::Univariate<FF, MAX_PARTIAL_RELATION_LENGTH>::zero();
            cached[column] = &zero_univariate;
        } else {
            // Constant nonzero edges need only copies, not a subtraction and
            // MAX_PARTIAL_RELATION_LENGTH - 2 field additions. Raw equality is cheap;
            // different coarse representatives keep the original extension path.
            const bool constant = val0.data[0] == val1.data[0] && val0.data[1] == val1.data[1] &&
                                  val0.data[2] == val1.data[2] && val0.data[3] == val1.data[3];
            const auto extended = constant
                                      ? ExtendedUnivariate(val0)
                                      : bb::Univariate<FF, 2>({ val0, val1 })
                                            .template extend_to<MAX_PARTIAL_RELATION_LENGTH>();
            if (used == pool.size()) {
                pool.push_back(std::make_unique<ExtendedUnivariate>(extended));
            } else {
                *pool[used] = extended;
            }
            cached[column] = pool[used++].get();
        }
    }
    valid[column] = true;
    return *cached[column];
}

} // namespace bb::avm2
