#include "barretenberg/vm2/tracegen/lib/multi_permutation_builder.hpp"

#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <ranges>
#include <stdexcept>
#include <string>

#include "barretenberg/common/thread.hpp"
#include "barretenberg/vm2/generated/columns.hpp"
#include "barretenberg/vm2/generated/relations/perms_registers.hpp"
#include "barretenberg/vm2/tracegen/test_trace_container.hpp"

namespace bb::avm2::tracegen {
namespace {

using C = Column;
using Builder = MultiPermutationBuilder<perm_registers_mem_op_0_settings, perm_registers_mem_op_1_settings>;

constexpr uint32_t NUM_PAIRS = 3000;

// Memory rows 2k and 2k + 1 hold the same tuple, keyed by k. Register 0 reads every k, register 1 every even k, so
// the two permutations share the rows of even k and the result depends on the order the builder hands rows out in.
TestTraceContainer make_trace()
{
    TestTraceContainer trace;
    for (uint32_t row = 2; row < (2 * NUM_PAIRS) + 2; ++row) {
        const uint32_t k = row / 2;
        trace.set(row,
                  { { { C::memory_sel, 1 },
                      { C::memory_clk, k },
                      { C::memory_address, k % 7 },
                      { C::memory_value, k * 3 },
                      { C::memory_tag, 2 } } });
    }
    for (uint32_t k = 1; k <= NUM_PAIRS; ++k) {
        trace.set(k,
                  { { { C::execution_sel_op_reg_effective_0_, 1 },
                      { C::execution_clk, k },
                      { C::execution_rop_0_, k % 7 },
                      { C::execution_register_0_, k * 3 },
                      { C::execution_mem_tag_reg_0_, 2 } } });
    }
    // Register 1 takes the clock of register 0's row.
    for (uint32_t k = 2; k <= NUM_PAIRS; k += 2) {
        trace.set(k,
                  { { { C::execution_sel_op_reg_effective_1_, 1 },
                      { C::execution_rop_1_, k % 7 },
                      { C::execution_register_1_, k * 3 },
                      { C::execution_mem_tag_reg_1_, 2 } } });
    }
    return trace;
}

// Runs the parts as interaction jobs run: process() on every part, concurrently or one after another in reverse
// order, then finish() on this thread. Returns the message of the exception finish() throws.
std::string run_parts(TestTraceContainer& trace, size_t parts, bool concurrent)
{
    auto jobs = Builder(C::memory_sel).split_into(parts);
    EXPECT_EQ(jobs.size(), parts);
    if (concurrent) {
        parallel_for(jobs.size(), [&](size_t i) { jobs[i]->process(trace); });
    } else {
        for (auto& job : std::views::reverse(jobs)) {
            job->process(trace);
        }
    }
    try {
        for (auto& job : jobs) {
            job->finish();
        }
    } catch (const std::runtime_error& e) {
        return e.what();
    }
    return "";
}

std::string run_whole(TestTraceContainer& trace)
{
    try {
        Builder(C::memory_sel).process(trace);
    } catch (const std::runtime_error& e) {
        return e.what();
    }
    return "";
}

TEST(MultiPermutationBuilderTest, PartsSetTheSameSelectors)
{
    TestTraceContainer expected = make_trace();
    ASSERT_EQ(run_whole(expected), "");
    // Register 0 takes the larger row of each pair, register 1 the remaining row of even pairs.
    EXPECT_EQ(expected.get(C::memory_sel_register_op_0_, 5), 1);
    EXPECT_EQ(expected.get(C::memory_sel_register_op_1_, 4), 1);

    for (const size_t parts : std::array<size_t, 3>{ 2, 3, 8 }) {
        TestTraceContainer trace = make_trace();
        ASSERT_EQ(run_parts(trace, parts, /*concurrent=*/true), "");
        for (const C col : { C::memory_sel_register_op_0_, C::memory_sel_register_op_1_ }) {
            for (uint32_t row = 0; row < (2 * NUM_PAIRS) + 3; ++row) {
                ASSERT_EQ(trace.get(col, row), expected.get(col, row)) << parts << " " << row;
            }
        }
    }
}

TEST(MultiPermutationBuilderTest, PartsThrowTheFirstFailure)
{
    // Tuples that are not in memory: register 1 at three rows, register 0 late in its rows.
    auto add_failures = [](TestTraceContainer& trace) {
        for (const uint32_t k : { 2901U, 1501U, 777U }) {
            trace.set(k,
                      { { { C::execution_sel_op_reg_effective_1_, 1 },
                          { C::execution_rop_1_, k % 7 },
                          { C::execution_register_1_, (k * 3) + 1 },
                          { C::execution_mem_tag_reg_1_, 2 } } });
        }
        trace.set(C::execution_register_0_, 2999, 1);
    };
    TestTraceContainer whole = make_trace();
    add_failures(whole);
    const std::string expected = run_whole(whole);
    ASSERT_NE(expected.find("PERM_REGISTERS_MEM_OP_0"), std::string::npos) << expected;

    for (const size_t parts : std::array<size_t, 3>{ 2, 3, 8 }) {
        for (const bool concurrent : { false, true }) {
            TestTraceContainer trace = make_trace();
            add_failures(trace);
            EXPECT_EQ(run_parts(trace, parts, concurrent), expected) << parts << " " << concurrent;
        }
    }

    // Without the register 0 failure, the first failure is register 1's earliest row.
    TestTraceContainer whole_1 = make_trace();
    add_failures(whole_1);
    whole_1.set(C::execution_register_0_, 2999, 2999 * 3);
    const std::string expected_1 = run_whole(whole_1);
    ASSERT_NE(expected_1.find("PERM_REGISTERS_MEM_OP_1. Could not find tuple in destination.\nSRC tuple (row 777)"),
              std::string::npos)
        << expected_1;
    for (const size_t parts : std::array<size_t, 3>{ 2, 3, 8 }) {
        for (const bool concurrent : { false, true }) {
            TestTraceContainer trace = make_trace();
            add_failures(trace);
            trace.set(C::execution_register_0_, 2999, 2999 * 3);
            EXPECT_EQ(run_parts(trace, parts, concurrent), expected_1) << parts << " " << concurrent;
        }
    }
}

} // namespace
} // namespace bb::avm2::tracegen
