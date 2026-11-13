#pragma once

#include "barretenberg/vm2/constraining/flavor.hpp"
#include "barretenberg/vm2/constraining/prover.hpp"
#include "barretenberg/vm2/tracegen/trace_container.hpp"

namespace bb::avm2::constraining {

// Computes the polynomials from the trace, and destroys it in the process.
AvmProver::ProverPolynomials compute_polynomials(tracegen::TraceContainer& trace);

// In our lookups and permutations, the inverses are computed whenever the src or destination selector is non-zero.
// This means that the inverse polynomial needs to be resized to the maximum of the src and dst selector sizes.
void resize_inverses(AvmFlavor::ProverPolynomials& prover_polynomials,
                     Column inverses_col,
                     Column src_selector_col,
                     Column dst_selector_col);

// The number of CRS points the proof needs when its longest column has `num_rows` rows: every commitment, including
// the PCS's, is to a polynomial that ends within round_up_power_2(num_rows).
size_t commitment_key_size(size_t num_rows);

// Computes the proving key from the polynomials. Its commitment key is `commitment_key` when that already holds the
// points the polynomials need (commitment_key_size of their largest end index), and a newly loaded one otherwise.
// Remark: This doesn't need to be a shared_ptr, but BB requires it.
std::shared_ptr<AvmProver::ProvingKey> proving_key_from_polynomials(AvmProver::ProverPolynomials& polynomials,
                                                                    AvmFlavor::CommitmentKey commitment_key = {});

} // namespace bb::avm2::constraining
