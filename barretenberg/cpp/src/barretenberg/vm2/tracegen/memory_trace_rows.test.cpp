#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

#include "barretenberg/common/assert.hpp"
#include "barretenberg/vm2/common/memory_types.hpp"
#include "barretenberg/vm2/common/tagged_value.hpp"
#include "barretenberg/vm2/generated/columns.hpp"
#include "barretenberg/vm2/simulation/events/memory_event.hpp"
#include "barretenberg/vm2/tracegen/memory_trace.hpp"
#include "barretenberg/vm2/tracegen/test_trace_container.hpp"

namespace bb::avm2::tracegen {
namespace {

using C = Column;
using simulation::MemoryEvent;
using simulation::MemoryMode;

// Enough events for write_rows() to fill the rows in chunks on several threads: every row, including those at chunk
// boundaries, must come from its own sorted event and the next one.
TEST(MemoryTraceRowsTest, RowsWrittenInChunks)
{
    const uint32_t n = (5 * TraceContainer::INTERVAL_SIZE) + 3;
    // Event k (from 1 to n) sorts to row k: three consecutive clocks per address, emitted in reverse order.
    std::vector<MemoryEvent> events;
    for (uint32_t k = n; k >= 1; --k) {
        events.push_back({ .execution_clk = k,
                           .mode = MemoryMode::READ,
                           .addr = k / 3,
                           .value = MemoryValue::from_tag(MemoryTag::U32, k),
                           .space_id = 0 });
    }

    TestTraceContainer trace;
    MemoryTraceBuilder builder;
    builder.process(events, trace);

    EXPECT_EQ(trace.get_column_rows(C::memory_sel), n + 1);
    for (uint32_t row = 1; row <= n; ++row) {
        const bool is_last = row == n;
        const bool last_access = is_last || row / 3 != (row + 1) / 3;
        const uint64_t diff = is_last ? 0 : (last_access ? 1 : 2);
        ASSERT_EQ(trace.get(C::memory_sel, row), 1) << row;
        ASSERT_EQ(trace.get(C::memory_value, row), row) << row;
        ASSERT_EQ(trace.get(C::memory_clk, row), row) << row;
        ASSERT_EQ(trace.get(C::memory_address, row), row / 3) << row;
        ASSERT_EQ(trace.get(C::memory_last_access, row), last_access ? 1 : 0) << row;
        ASSERT_EQ(trace.get(C::memory_sel_rng_chk, row), is_last ? 0 : 1) << row;
        ASSERT_EQ(trace.get(C::memory_diff, row), diff) << row;
        ASSERT_EQ(trace.get(C::memory_limb_0_, row), diff) << row;
        // The address difference is 1 at a last access before the last row, and 1 is its own inverse.
        ASSERT_EQ(trace.get(C::memory_glob_addr_diff_inv, row), (last_access && !is_last) ? 1 : 0) << row;
    }
}

// A diff too large for its limbs, found while the rows are written in chunks, fails on the calling thread with the
// serial loop's assertion.
TEST(MemoryTraceRowsTest, DiffTooLargeFailsOnCallingThread)
{
    const uint32_t n = 5 * TraceContainer::INTERVAL_SIZE;
    std::vector<MemoryEvent> events;
    for (uint32_t k = 1; k <= n; ++k) {
        events.push_back({ .execution_clk = k,
                           .mode = MemoryMode::READ,
                           .addr = k,
                           .value = MemoryValue::from_tag(MemoryTag::U32, k) });
    }
    // Two writes to one address at one clock, in a late chunk.
    for (int i = 0; i < 2; ++i) {
        events.push_back({ .execution_clk = 7,
                           .mode = MemoryMode::WRITE,
                           .addr = n - 10,
                           .value = MemoryValue::from_tag(MemoryTag::U32, 1) });
    }

    TestTraceContainer trace;
    MemoryTraceBuilder builder;
    EXPECT_THROW_WITH_MESSAGE(builder.process(events, trace), "diff < \\(1ULL << 48\\)");
}

} // namespace
} // namespace bb::avm2::tracegen
