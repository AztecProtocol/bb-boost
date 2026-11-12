#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "barretenberg/vm2/simulation/events/bytecode_events.hpp"
#include "barretenberg/vm2/simulation/events/event_emitter.hpp"
#include "barretenberg/vm2/tracegen/lib/interaction_def.hpp"
#include "barretenberg/vm2/tracegen/trace_container.hpp"

namespace bb::avm2::tracegen {

class BytecodeTraceBuilder final {
  public:
    void process_hashing(const simulation::EventEmitterInterface<simulation::BytecodeHashingEvent>::Container& events,
                         TraceContainer& trace);

    void process_retrieval(
        const simulation::EventEmitterInterface<simulation::BytecodeRetrievalEvent>::Container& events,
        TraceContainer& trace);

    void process_decomposition(
        const simulation::EventEmitterInterface<simulation::BytecodeDecompositionEvent>::Container& events,
        TraceContainer& trace);

    // A decomposition of up to this many rows runs as one job; a longer one runs as up to MAX_DECOMPOSITION_PARTS.
    static constexpr uint64_t DECOMPOSITION_ROWS_PER_PART = 1 << 13;
    static constexpr size_t MAX_DECOMPOSITION_PARTS = 8;

    // The number of parts for a decomposition of num_rows rows: it depends on the row count alone.
    static size_t decomposition_parts(uint64_t num_rows);

    /**
     * @brief Jobs that together fill exactly the cells process_decomposition fills, with the same values.
     * @details The jobs may run in any order and concurrently. Each writes the cells of its own whole trace shards;
     * whichever finishes last then runs the columns' batch inversion, once, and after it `after_last`. `parts` = 0
     * takes decomposition_parts(rows); one part is process_decomposition itself.
     */
    static std::vector<std::function<void()>> decomposition_jobs(
        const simulation::EventEmitterInterface<simulation::BytecodeDecompositionEvent>::Container& events,
        TraceContainer& trace,
        std::function<void()> after_last,
        size_t parts = 0);

    void process_instruction_fetching(
        const simulation::EventEmitterInterface<simulation::InstructionFetchingEvent>::Container& events,
        TraceContainer& trace);

    static const InteractionDefinition interactions;
};

} // namespace bb::avm2::tracegen
