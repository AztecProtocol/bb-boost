#include "barretenberg/vm2/avm_api.hpp"

#include <future>

#include "barretenberg/common/thread.hpp"
#include "barretenberg/vm2/constraining/flavor.hpp"
#include "barretenberg/vm2/proving_helper.hpp"
#include "barretenberg/vm2/simulation_helper.hpp"
#include "barretenberg/vm2/tooling/debugger.hpp"
#include "barretenberg/vm2/tooling/stats.hpp"
#include "barretenberg/vm2/tracegen_helper.hpp"

namespace bb::avm2 {

using namespace bb::avm2::simulation;

namespace {
// Without multithreading the CRS factory has no mutex, so the CRS is loaded where it is first needed.
#ifdef NO_MULTITHREADING
constexpr auto CRS_LOAD_POLICY = std::launch::deferred;
#else
constexpr auto CRS_LOAD_POLICY = std::launch::async;
#endif
} // namespace

AvmAPI::AvmProof AvmAPI::prove(const AvmAPI::ProvingInputs& inputs)
{
    // The proving key's commitment key needs the CRS for the full (fixed) circuit size, which does not depend on the
    // inputs. Build the same commitment key on a background thread so that reading and decoding the CRS overlaps
    // simulation and trace generation; the CRS factory keeps the points, so the proving key later finds them loaded.
    // The parallel_for thread count is per thread, so the loading thread takes the caller's.
    auto crs_loaded = std::async(CRS_LOAD_POLICY, [num_cpus = get_num_cpus()] {
        if (get_num_cpus() != num_cpus) {
            set_parallel_for_concurrency(num_cpus);
        }
        [[maybe_unused]] AvmFlavor::CommitmentKey commitment_key(AvmFlavor::ProvingKey::circuit_size);
    });

    // Simulate.
    vinfo("Simulating...");
    AvmSimulationHelper simulation_helper;

    auto events = AVM_TRACK_TIME_V("simulation/all", simulation_helper.simulate_for_witgen(inputs.hints));

    // Generate trace.
    vinfo("Generating trace...");
    AvmTraceGenHelper tracegen_helper;
    auto trace =
        AVM_TRACK_TIME_V("tracegen/all", tracegen_helper.generate_trace(std::move(events), inputs.public_inputs));

    // Waits for the CRS; a loading error is rethrown here, before any proving work starts.
    crs_loaded.get();

    // Prove.
    vinfo("Proving...");
    AvmProvingHelper proving_helper;
    auto proof = AVM_TRACK_TIME_V("proving/all", proving_helper.prove(std::move(trace)));

    vinfo("Done!");
    return proof;
}

bool AvmAPI::check_circuit(const AvmAPI::ProvingInputs& inputs)
{
    // Simulate.
    vinfo("Simulating...");
    AvmSimulationHelper simulation_helper;

    auto events = AVM_TRACK_TIME_V("simulation/all", simulation_helper.simulate_for_witgen(inputs.hints));

    // Generate trace.
    // In contrast to proving, we do this step by step since it's usually more useful to debug
    // before trying to run the interaction builders.
    vinfo("Generating trace...");
    AvmTraceGenHelper tracegen_helper;
    tracegen::TraceContainer trace;
    AVM_TRACK_TIME("tracegen/all", tracegen_helper.fill_trace_columns(trace, std::move(events), inputs.public_inputs));

    // Go into interactive debug mode if requested.
    if (getenv("AVM_DEBUG") != nullptr) {
        InteractiveDebugger debugger(trace);
        debugger.run();
    }

    AVM_TRACK_TIME("tracegen/all", tracegen_helper.fill_trace_interactions(trace));

    // Check circuit.
    vinfo("Checking circuit...");
    AvmProvingHelper proving_helper;
    return proving_helper.check_circuit(std::move(trace));
}

bool AvmAPI::verify(const AvmProof& proof, const PublicInputs& pi)
{
    vinfo("Verifying...");
    AvmProvingHelper proving_helper;
    return AVM_TRACK_TIME_V("verifing/all", proving_helper.verify(proof, pi));
}

AvmAPI::AvmVerificationKey AvmAPI::get_verification_key()
{
    return AvmAPI::AvmVerificationKey();
}

} // namespace bb::avm2
