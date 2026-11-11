#pragma once

#include <array>
#include <cstdint>
#include <vector>

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

    // process() in two steps: the sort can run beside other builders, and write_rows() then fills the rows on all
    // threads. The events must outlive write_rows().
    void sort_events(const simulation::EventEmitterInterface<simulation::MemoryEvent>::Container& events);
    void write_rows(TraceContainer& trace) const;

    static const InteractionDefinition interactions;

  private:
    // Caches the immutable comparison key beside a pointer, avoiding scattered event reads on every comparison.
    struct SortEntry {
        std::array<uint64_t, 2> key;
        const simulation::MemoryEvent* event;
    };
    std::vector<SortEntry> sorted_events;
};

} // namespace bb::avm2::tracegen
