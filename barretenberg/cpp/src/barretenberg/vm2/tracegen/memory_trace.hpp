#pragma once

#include <array>
#include <cstdint>

#include "barretenberg/vm2/generated/columns.hpp"
#include "barretenberg/vm2/simulation/events/event_emitter.hpp"
#include "barretenberg/vm2/simulation/events/memory_event.hpp"
#include "barretenberg/vm2/tracegen/lib/interaction_def.hpp"
#include "barretenberg/vm2/tracegen/trace_container.hpp"

namespace bb::avm2::tracegen {

class MemoryTraceBuilder final {
  public:
    // Exact lexicographic comparator key, including the signed mode ordering.
    static std::array<uint64_t, 2> get_event_sort_key(const simulation::MemoryEvent& event);

    void process(const simulation::EventEmitterInterface<simulation::MemoryEvent>::Container& events,
                 TraceContainer& trace);

    static const InteractionDefinition interactions;
};

} // namespace bb::avm2::tracegen
