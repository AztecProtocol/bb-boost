#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "barretenberg/common/thread.hpp"
#include "barretenberg/vm2/generated/columns.hpp"
#include "barretenberg/vm2/simulation/events/bytecode_events.hpp"
#include "barretenberg/vm2/tracegen/bytecode_trace.hpp"
#include "barretenberg/vm2/tracegen/trace_container.hpp"

namespace bb::avm2::tracegen {
namespace {

// Bytecodes whose windows and 31-byte groups straddle the parts' boundaries, with an empty one among them.
std::vector<simulation::BytecodeDecompositionEvent> straddling_decomposition_events()
{
    std::vector<simulation::BytecodeDecompositionEvent> events;
    uint8_t next_byte = 1;
    for (const size_t length : std::array<size_t, 9>{ 1, 30, 31, 0, 32, 62, (1 << 13) + 17, 4000, 37 }) {
        auto bytecode = std::make_shared<std::vector<uint8_t>>(length);
        for (auto& byte : *bytecode) {
            byte = next_byte;
            next_byte = static_cast<uint8_t>((next_byte * 31) + 7);
        }
        events.push_back(
            { .bytecode_id = static_cast<uint64_t>(events.size() + 100), .bytecode = std::move(bytecode) });
    }
    return events;
}

// Both traces hold only the decomposition, so every cell of every column must have the same raw limbs. Columns are
// not chosen by name: column names are only compiled in with AVM_INCLUDE_COLUMN_INFORMATION.
void expect_same_decomposition_limbs(const TraceContainer& actual, const TraceContainer& expected)
{
    size_t written_columns = 0;
    for (size_t c = 0; c < TraceContainer::num_columns(); ++c) {
        const auto col = static_cast<Column>(c);
        ASSERT_EQ(actual.get_column_rows(col), expected.get_column_rows(col)) << "column " << c;
        written_columns += expected.get_column_rows(col) > 0 ? 1 : 0;
        for (uint32_t row = 0; row < expected.get_column_rows(col); ++row) {
            ASSERT_EQ(std::memcmp(&actual.get(col, row), &expected.get(col, row), sizeof(FF)), 0)
                << "column " << c << " row " << row;
        }
    }
    EXPECT_GT(written_columns, 0U);
}

TEST(BytecodeDecompositionPartsTest, MatchOneJobInEverySequentialOrder)
{
    const auto events = straddling_decomposition_events();
    TraceContainer expected;
    BytecodeTraceBuilder().process_decomposition(events, expected);

    // The parts run one after another, in each of the six orders, so each part is once the last and runs the batch
    // inversion. A missing or repeated inversion, or a row no part writes, would leave the trace different from the
    // one-job trace. Concurrent publication is covered by the next test.
    std::vector<size_t> order = { 0, 1, 2 };
    do {
        TraceContainer trace;
        size_t after_last_runs = 0;
        auto jobs = BytecodeTraceBuilder::decomposition_jobs(events, trace, [&]() { ++after_last_runs; }, 3);
        ASSERT_EQ(jobs.size(), 3U);
        for (const size_t part : order) {
            jobs[part]();
        }
        EXPECT_EQ(after_last_runs, 1U);
        expect_same_decomposition_limbs(trace, expected);
    } while (std::next_permutation(order.begin(), order.end()));
}

TEST(BytecodeDecompositionPartsTest, MatchOneJobWhenThePartsRunConcurrently)
{
    const auto events = straddling_decomposition_events();
    TraceContainer expected;
    BytecodeTraceBuilder().process_decomposition(events, expected);

    // Seven parts of 2^11 rows run on the pool, so the part that inverts must see every other part's writes through
    // the countdown. Repeated, since which part finishes last changes from run to run.
    for (size_t run = 0; run < 8; ++run) {
        TraceContainer trace;
        std::atomic<size_t> after_last_runs = 0;
        auto jobs = BytecodeTraceBuilder::decomposition_jobs(events, trace, [&]() { ++after_last_runs; }, 8);
        ASSERT_EQ(jobs.size(), 7U);
        bb::parallel_for(jobs.size(), [&](size_t i) { jobs[i](); });
        EXPECT_EQ(after_last_runs.load(), 1U);
        expect_same_decomposition_limbs(trace, expected);
    }
}

TEST(BytecodeDecompositionPartsTest, DependOnTheRowCountAlone)
{
    constexpr uint64_t PER_PART = BytecodeTraceBuilder::DECOMPOSITION_ROWS_PER_PART;
    EXPECT_EQ(BytecodeTraceBuilder::decomposition_parts(0), 1U);
    EXPECT_EQ(BytecodeTraceBuilder::decomposition_parts(PER_PART), 1U);
    EXPECT_EQ(BytecodeTraceBuilder::decomposition_parts(PER_PART + 1), 2U);
    EXPECT_EQ(BytecodeTraceBuilder::decomposition_parts(uint64_t{ 1 } << 30),
              BytecodeTraceBuilder::MAX_DECOMPOSITION_PARTS);

    // By default the parts follow the row count: these 12,402 rows make two.
    const auto events = straddling_decomposition_events();
    TraceContainer expected;
    BytecodeTraceBuilder().process_decomposition(events, expected);
    TraceContainer trace;
    size_t after_last_runs = 0;
    auto jobs = BytecodeTraceBuilder::decomposition_jobs(events, trace, [&]() { ++after_last_runs; });
    EXPECT_EQ(jobs.size(), 2U);
    for (auto& job : jobs) {
        job();
    }
    EXPECT_EQ(after_last_runs, 1U);
    expect_same_decomposition_limbs(trace, expected);
}

} // namespace
} // namespace bb::avm2::tracegen
