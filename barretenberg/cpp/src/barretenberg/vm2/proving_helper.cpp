#include "barretenberg/vm2/proving_helper.hpp"

#include <cstdint>
#include <cstdlib>
#include <future>
#include <memory>
#include <stdexcept>

#include "barretenberg/common/assert.hpp"
#include "barretenberg/common/serialize.hpp"
#include "barretenberg/common/thread.hpp"
#include "barretenberg/numeric/bitop/get_msb.hpp"
#include "barretenberg/polynomials/backing_memory.hpp"
#include "barretenberg/vm2/common/constants.hpp"
#include "barretenberg/vm2/constraining/avm_fixed_vk.hpp"
#include "barretenberg/vm2/constraining/check_circuit.hpp"
#include "barretenberg/vm2/constraining/polynomials.hpp"
#include "barretenberg/vm2/constraining/prover.hpp"
#include "barretenberg/vm2/constraining/verifier.hpp"
#include "barretenberg/vm2/tooling/stats.hpp"

namespace bb::avm2 {

namespace {
// Without multithreading the CRS factory has no mutex, so the CRS is loaded where it is first needed.
#ifdef NO_MULTITHREADING
constexpr auto CRS_LOAD_POLICY = std::launch::deferred;
#else
constexpr auto CRS_LOAD_POLICY = std::launch::async;
#endif
} // namespace

AvmProvingHelper::Proof AvmProvingHelper::prove(tracegen::TraceContainer&& trace)
{
    // Declared first so that it ends after every polynomial below is destroyed.
    PolynomialMemoryRelease release_polynomials_on_return;

    // The longest column bounds every commitment of the proof, so the CRS points it needs are known before the trace
    // becomes polynomials (which frees its columns), and they load while the polynomials are built. The parallel_for
    // thread count is per thread, so the loading thread takes the caller's.
    const size_t num_rows = trace.get_num_rows();
    const size_t crs_points = constraining::commitment_key_size(num_rows);
    auto commitment_key_loaded = std::async(CRS_LOAD_POLICY, [crs_points, num_cpus = get_num_cpus()] {
        if (get_num_cpus() != num_cpus) {
            set_parallel_for_concurrency(num_cpus);
        }
        return AvmFlavor::CommitmentKey(crs_points);
    });

    auto polynomials = AVM_TRACK_TIME_V("proving/prove:compute_polynomials", constraining::compute_polynomials(trace));
    // A loading error is rethrown here, before any proving work starts.
    auto commitment_key = AVM_TRACK_TIME_V("proving/prove:crs_wait", commitment_key_loaded.get());
    auto proving_key =
        AVM_TRACK_TIME_V("proving/prove:proving_key",
                         constraining::proving_key_from_polynomials(polynomials, std::move(commitment_key)));
    vinfo("CRS: largest column ",
          num_rows,
          " rows, proof needs ",
          crs_points,
          " points, ",
          proving_key->commitment_key.get_monomial_size(),
          " loaded");

    auto verification_key = std::make_shared<AvmVerifier::VerificationKey>();

    auto prover = AVM_TRACK_TIME_V("proving/prove:construct_prover",
                                   AvmProver(proving_key, verification_key, proving_key->commitment_key));

    auto proof = AVM_TRACK_TIME_V("proving/construct_proof", prover.construct_proof());

    // The proof holds no polynomial memory: blocks freed from here on, as this function returns, are released together.
    release_polynomials_on_return.activate();
    return proof;
}

bool AvmProvingHelper::check_circuit(tracegen::TraceContainer&& trace)
{
    // The proof is done over the whole circuit (2^21 rows).
    // However, for check-circuit purposes we run only over the witness rows
    // PLUS one extra row to catch any possible errors in the empty remainder
    // of the circuit.
    const bool skippable_enabled = (getenv("AVM_DISABLE_SKIPPABLE") == nullptr);
    const size_t num_rows = skippable_enabled ? trace.get_num_witness_rows() + 1 : trace.get_num_rows();
    vinfo("Running check ",
          skippable_enabled ? "(with skippable)" : "(without skippable)",
          " circuit over ",
          num_rows,
          " rows.");

    // Warning: this destroys the trace.
    auto polynomials = AVM_TRACK_TIME_V("proving/prove:compute_polynomials", constraining::compute_polynomials(trace));
    try {
        AVM_TRACK_TIME("proving/check_circuit",
                       constraining::run_check_circuit(polynomials, num_rows, skippable_enabled));
    } catch (std::runtime_error& e) {
        // FIXME: This exception is never caught because it's thrown in a different thread.
        // Execution never gets here!
        vinfo("Circuit check failed: ", e.what());
    }

    return true;
}

bool AvmProvingHelper::verify(const AvmProvingHelper::Proof& proof, const PublicInputs& pi)
{
    auto vk =
        AVM_TRACK_TIME_V("proving/verify:create_verification_key", std::make_shared<AvmFlavor::VerificationKey>());
    auto verifier = AVM_TRACK_TIME_V("proving/verify:construct_verifier", AvmVerifier());
    return AVM_TRACK_TIME_V("proving/verify_proof", verifier.verify_proof(proof, pi.to_columns()));
}

} // namespace bb::avm2
