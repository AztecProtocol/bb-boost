#include "barretenberg/vm2/tracegen_helper.hpp"

#include <gtest/gtest.h>

#include "barretenberg/vm2/generated/columns.hpp"
#include "barretenberg/vm2/simulation_helper.hpp"
#include "barretenberg/vm2/testing/fixtures.hpp"
#include "barretenberg/vm2/tracegen/execution_trace.hpp"
#include "barretenberg/vm2/tracegen/trace_container.hpp"

namespace bb::avm2 {
namespace {

// The execution events are freed while the trace is filled; the execution rows must still be the ones the events
// describe, so the events are freed only after the execution builder has read them.
TEST(AvmTraceGenHelperTest, ExecutionEventsFreedAfterTheirRows)
{
    const AvmProvingInputs inputs = testing::get_minimal_proving_inputs();
    auto events = AvmSimulationHelper().simulate_for_witgen(inputs.hints);
    ASSERT_FALSE(events.execution.empty());

    tracegen::TraceContainer expected;
    tracegen::ExecutionTraceBuilder().process(events.execution, expected);

    tracegen::TraceContainer trace;
    AvmTraceGenHelper().fill_trace_columns(trace, std::move(events), inputs.public_inputs);
    EXPECT_TRUE(events.execution.empty()); // NOLINT(bugprone-use-after-move): taken by reference and cleared in place

    // Columns are chosen by what the execution builder wrote, not by name: column names are only compiled in with
    // AVM_INCLUDE_COLUMN_INFORMATION.
    size_t execution_columns = 0;
    for (size_t c = 0; c < tracegen::TraceContainer::num_columns(); ++c) {
        const auto col = static_cast<Column>(c);
        if (expected.get_column_rows(col) == 0) {
            continue;
        }
        ++execution_columns;
        ASSERT_EQ(trace.get_column_rows(col), expected.get_column_rows(col)) << "column " << c;
        for (uint32_t row = 0; row < expected.get_column_rows(col); ++row) {
            ASSERT_EQ(trace.get(col, row), expected.get(col, row)) << "column " << c << " row " << row;
        }
    }
    EXPECT_GT(execution_columns, 0U);
}

} // namespace
} // namespace bb::avm2
