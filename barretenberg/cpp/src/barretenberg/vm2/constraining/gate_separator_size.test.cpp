#include "barretenberg/common/log.hpp"
#include "barretenberg/relations/relation_parameters.hpp"
#include "barretenberg/sumcheck/sumcheck.hpp"
#include "barretenberg/vm2/common/constants.hpp"
#include "barretenberg/vm2/constraining/flavor.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace bb::avm2::constraining {
namespace {

using FF = AvmFlavor::FF;
using Polynomial = AvmFlavor::Polynomial;

// Collects the verbose log while alive and restores the previous level and sink on exit, including when a test aborts
// out of the prover.
class VerboseLogCapture {
  public:
    VerboseLogCapture()
        : previous_level(bb_log_level)
        , previous_function(log_function)
    {
        bb_log_level = LogLevel::VERBOSE;
        set_log_function([this](LogLevel /*level*/, const std::string& message) {
            const std::lock_guard<std::mutex> lock(mutex);
            lines.push_back(message);
        });
    }
    VerboseLogCapture(const VerboseLogCapture&) = delete;
    VerboseLogCapture& operator=(const VerboseLogCapture&) = delete;
    VerboseLogCapture(VerboseLogCapture&&) = delete;
    VerboseLogCapture& operator=(VerboseLogCapture&&) = delete;
    ~VerboseLogCapture()
    {
        set_log_function(previous_function);
        bb_log_level = previous_level;
    }

    std::vector<std::string> lines_starting_with(const std::string& prefix)
    {
        const std::lock_guard<std::mutex> lock(mutex);
        std::vector<std::string> matches;
        for (const auto& line : lines) {
            if (line.starts_with(prefix)) {
                matches.push_back(line);
            }
        }
        return matches;
    }

  private:
    LogLevel previous_level;
    LogFunction previous_function;
    std::mutex mutex;
    std::vector<std::string> lines;
};

// Runs the AVM sumcheck prover over polynomials whose only non-empty witness ends at `witness_end`, and returns the
// gate separator table lines it logs. The table's size and its per-round guard live inside SumcheckProver::prove, so
// this goes through the production selection; a guard failure aborts the run.
std::vector<std::string> gate_separator_table_lines(size_t witness_end)
{
    AvmFlavor::ProverPolynomials polynomials;
    if (witness_end > 0) {
        auto& witness = polynomials.get_witness()[0];
        witness = Polynomial(witness_end, MAX_AVM_TRACE_SIZE);
        for (size_t row = 0; row < witness_end; ++row) {
            witness.at(row) = FF(row + 1);
        }
    }
    std::vector<FF> gate_challenges(MAX_AVM_TRACE_LOG_SIZE);
    for (auto& challenge : gate_challenges) {
        challenge = FF::random_element();
    }

    VerboseLogCapture log;
    SumcheckProver<AvmFlavor> sumcheck(MAX_AVM_TRACE_SIZE,
                                       polynomials,
                                       std::make_shared<AvmFlavor::Transcript>(),
                                       FF::random_element(),
                                       gate_challenges,
                                       RelationParameters<FF>{},
                                       MAX_AVM_TRACE_LOG_SIZE);
    static_cast<void>(sumcheck.prove());
    return log.lines_starting_with("sumcheck: gate separator table");
}

// The table has 2^ceil(log2 E) entries, E being round 0's effective round size (the largest witness end rounded up to
// even), and a single entry when no witness has support.
TEST(AvmGateSeparatorSizeTest, SumcheckSizesTheTableToTheFirstRoundsActiveRows)
{
    EXPECT_EQ(gate_separator_table_lines(0),
              std::vector<std::string>{ "sumcheck: gate separator table of 2^0 entries" });
    EXPECT_EQ(gate_separator_table_lines(5),
              std::vector<std::string>{ "sumcheck: gate separator table of 2^3 entries" });
    EXPECT_EQ(gate_separator_table_lines(1000),
              std::vector<std::string>{ "sumcheck: gate separator table of 2^10 entries" });
}

} // namespace
} // namespace bb::avm2::constraining
