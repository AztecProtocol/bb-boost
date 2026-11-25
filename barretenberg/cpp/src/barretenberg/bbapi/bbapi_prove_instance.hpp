#pragma once
#include "barretenberg/trace_to_polynomials/copied_circuit_data.hpp"
#include "barretenberg/ultra_honk/prover_instance.hpp"
#include <memory>

namespace bb::bbapi {

/**
 * @brief Builds the prover instance a CircuitProve proof is made from. The request reads nothing from the circuit
 * afterwards, so each trace block and lookup table is freed as soon as it has been copied into the polynomials.
 */
template <typename Flavor>
std::shared_ptr<ProverInstance_<Flavor>> circuit_prove_instance(typename Flavor::CircuitBuilder& circuit)
{
    return std::make_shared<ProverInstance_<Flavor>>(circuit, CopiedCircuitData::RELEASE);
}

} // namespace bb::bbapi
