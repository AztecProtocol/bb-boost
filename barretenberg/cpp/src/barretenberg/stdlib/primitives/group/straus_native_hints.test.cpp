#include "barretenberg/stdlib/primitives/group/straus_native_hints.hpp"
#include "barretenberg/circuit_checker/circuit_checker.hpp"
#include "barretenberg/common/thread.hpp"
#include "barretenberg/numeric/general/general.hpp"
#include "barretenberg/numeric/random/engine.hpp"
#include "barretenberg/stdlib/primitives/group/cycle_group.hpp"
#include "barretenberg/stdlib_circuit_builders/ultra_circuit_builder_test_utils.hpp"

#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace bb::stdlib {
namespace {

using Builder = UltraCircuitBuilder;
using cycle_group_ct = cycle_group<Builder>;
using cycle_scalar_ct = cycle_group_ct::cycle_scalar;
using Group = bb::grumpkin::g1;
using Element = Group::element;
using AffineElement = Group::affine_element;

constexpr size_t NUM_ROUNDS = 32;
constexpr size_t TABLE_BITS = 2;
constexpr size_t TABLE_SIZE = size_t{ 1 } << TABLE_BITS;
// The parallel code cuts each round into chunks of 1024 points; these cases end in a partial chunk.
constexpr size_t SMALL_NUM_POINTS = 5;
constexpr size_t LARGE_NUM_POINTS = 2053;
static_assert(SMALL_NUM_POINTS * NUM_ROUNDS < STRAUS_HINTS_MIN_PARALLEL_READS);
static_assert(LARGE_NUM_POINTS * NUM_ROUNDS >= STRAUS_HINTS_MIN_PARALLEL_READS);

// Points share a few tables; the computation does not care which table a pointer names.
struct Reads {
    std::vector<AffineElement> offset_generators;
    std::vector<std::array<AffineElement, TABLE_SIZE>> distinct_tables;
    std::vector<std::vector<uint64_t>> scalar_slices;
    std::vector<const AffineElement*> tables;
    std::vector<const uint64_t*> slices;

    void point_to_tables(const std::vector<size_t>& table_of_point)
    {
        for (size_t j = 0; j < table_of_point.size(); ++j) {
            tables.push_back(distinct_tables[table_of_point[j]].data());
            slices.push_back(scalar_slices[j].data());
        }
    }
    StrausTableReads reads() const
    {
        return { .offset_generators = offset_generators,
                 .tables = tables,
                 .slices = slices,
                 .num_rounds = NUM_ROUNDS,
                 .table_bits = TABLE_BITS };
    }
};

std::vector<std::vector<uint64_t>> random_slices(const size_t num_points)
{
    auto& engine = numeric::get_debug_randomness();
    std::vector<std::vector<uint64_t>> slices(num_points, std::vector<uint64_t>(NUM_ROUNDS));
    for (auto& point_slices : slices) {
        for (auto& slice : point_slices) {
            slice = engine.get_random_uint64() % TABLE_SIZE;
        }
    }
    return slices;
}

// Tables as the plookup batch multiplication builds them: offset generator + k * base point.
Reads random_reads(const size_t num_points)
{
    constexpr size_t num_tables = 4;
    const auto offset_generators = cycle_group_ct::OffsetGenerators::default_generators().get(num_points + 1);
    const auto bases = Group::derive_generators("straus_native_hints_test", num_tables);
    Reads reads{ .offset_generators = { offset_generators.begin(), offset_generators.end() },
                 .distinct_tables = std::vector<std::array<AffineElement, TABLE_SIZE>>(num_tables),
                 .scalar_slices = random_slices(num_points) };
    for (size_t t = 0; t < num_tables; ++t) {
        Element entry = offset_generators[t + 1];
        for (auto& table_entry : reads.distinct_tables[t]) {
            table_entry = AffineElement(entry);
            entry += bases[t];
        }
    }
    std::vector<size_t> table_of_point(num_points);
    for (size_t j = 0; j < num_points; ++j) {
        table_of_point[j] = j % num_tables;
    }
    reads.point_to_tables(table_of_point);
    return reads;
}

/**
 * The serial code's edge cases, from an accumulator starting at P: adding P doubles it and adding a table entry at
 * infinity keeps it; with `cancel`, adding -P twice takes it to infinity, a transcript entry at infinity, from which
 * adding P restarts it. In the last round, at the start of the second chunk P then -P cancels inside a chunk sum and
 * Q + Q doubles there, and the third chunk, which ends the round, sums to infinity through a table entry at infinity.
 */
Reads edge_reads(const size_t num_points, const bool cancel)
{
    const auto points = Group::derive_generators("straus_native_hints_edge_test", 2);
    const AffineElement& p = points[0];
    const AffineElement& q = points[1];
    constexpr uint64_t P = 0;
    constexpr uint64_t INFINITY_ENTRY = 1;
    constexpr uint64_t MINUS_P = 2;
    constexpr uint64_t Q = 3;
    Reads reads{ .offset_generators = std::vector<AffineElement>(num_points + 1, q),
                 .distinct_tables = { { p, Group::affine_point_at_infinity, -p, q } },
                 .scalar_slices = random_slices(num_points) };
    reads.offset_generators[0] = p;
    // Round 0 reads each point's most significant slice.
    const std::vector<uint64_t> first_round = { P, INFINITY_ENTRY, MINUS_P, cancel ? MINUS_P : Q, P };
    for (size_t j = 0; j < first_round.size(); ++j) {
        reads.scalar_slices[j][NUM_ROUNDS - 1] = first_round[j];
    }
    if (num_points == LARGE_NUM_POINTS) {
        const std::array<uint64_t, 4> chunk_start = { P, MINUS_P, Q, Q };
        for (size_t j = 0; j < chunk_start.size(); ++j) {
            reads.scalar_slices[1024 + j][0] = chunk_start[j];
        }
        const std::array<uint64_t, LARGE_NUM_POINTS - 2048> last_chunk = { P, MINUS_P, INFINITY_ENTRY, P, MINUS_P };
        for (size_t j = 0; j < last_chunk.size(); ++j) {
            reads.scalar_slices[2048 + j][0] = last_chunk[j];
        }
    }
    reads.point_to_tables(std::vector<size_t>(num_points, 0));
    return reads;
}

/**
 * Round 0 reaches infinity in its third chunk, which the threaded code replays from an accumulator rebuilt from the
 * first two chunks' sums: every point reads a table entry at infinity, which keeps the accumulator at P, except the
 * one before last, which reads -P. The entry at infinity carries a nonzero raw y, so the serial accumulator ends the
 * round with that y, which round 1's doublings keep, while the threaded code rebuilds the round's end from its last
 * chunk's sum, -P, and gets infinity in its own encoding. Only the serial fallback makes those doubling hints agree.
 */
Reads later_chunk_infinity_reads()
{
    const auto points = Group::derive_generators("straus_native_hints_edge_test", 2);
    const AffineElement& p = points[0];
    const AffineElement& q = points[1];
    AffineElement infinity = Group::affine_point_at_infinity;
    infinity.y = bb::grumpkin::fq(7);
    constexpr uint64_t INFINITY_ENTRY = 1;
    constexpr uint64_t MINUS_P = 2;
    Reads reads{ .offset_generators = std::vector<AffineElement>(LARGE_NUM_POINTS + 1, q),
                 .distinct_tables = { { p, infinity, -p, q } },
                 .scalar_slices = random_slices(LARGE_NUM_POINTS) };
    reads.offset_generators[0] = p;
    for (size_t j = 0; j < LARGE_NUM_POINTS; ++j) {
        reads.scalar_slices[j][NUM_ROUNDS - 1] = j == LARGE_NUM_POINTS - 2 ? MINUS_P : INFINITY_ENTRY;
    }
    reads.point_to_tables(std::vector<size_t>(LARGE_NUM_POINTS, 0));
    return reads;
}

// The affine hints the batch multiplication makes of a transcript, including the raw coordinates of infinity.
std::vector<AffineElement> affine_hints(const StrausNativeHints& hints)
{
    std::vector<Element> transcript(hints.operation_transcript.get(),
                                    hints.operation_transcript.get() + hints.transcript_size);
    Element::batch_normalize(transcript.data(), transcript.size());
    std::vector<AffineElement> affine(transcript.size());
    for (size_t k = 0; k < transcript.size(); ++k) {
        affine[k] = AffineElement(transcript[k].x, transcript[k].y);
    }
    return affine;
}

bool same_hint(const AffineElement& actual, const AffineElement& expected)
{
    if (actual.is_point_at_infinity() || expected.is_point_at_infinity()) {
        return std::ranges::equal(actual.x.data, expected.x.data) && std::ranges::equal(actual.y.data, expected.y.data);
    }
    return actual == expected;
}

void expect_same_hints(const StrausNativeHints& actual, const StrausNativeHints& expected, const std::string& label)
{
    const auto actual_hints = affine_hints(actual);
    const auto expected_hints = affine_hints(expected);
    ASSERT_EQ(actual_hints.size(), expected_hints.size()) << label;
    size_t num_mismatches = 0;
    for (size_t k = 0; k < actual_hints.size(); ++k) {
        num_mismatches += static_cast<size_t>(!same_hint(actual_hints[k], expected_hints[k]));
    }
    EXPECT_EQ(num_mismatches, 0U) << label;
    EXPECT_EQ(AffineElement(actual.offset_generator_accumulator), AffineElement(expected.offset_generator_accumulator))
        << label;
}

Builder build_batch_mul(const std::vector<AffineElement>& points, const std::vector<bb::grumpkin::fr>& scalars)
{
    Builder builder;
    std::vector<cycle_group_ct> constant_points;
    std::vector<cycle_scalar_ct> witness_scalars;
    for (size_t i = 0; i < points.size(); ++i) {
        constant_points.emplace_back(points[i]);
        witness_scalars.push_back(cycle_scalar_ct::from_witness(&builder, scalars[i]));
    }
    cycle_group_ct::fixed_batch_mul(constant_points, witness_scalars, /*table_bits=*/4);
    return builder;
}

#if !defined(NO_MULTITHREADING) && !defined(OMP_MULTITHREADING)
/**
 * @brief Gives the calling thread's parallel_for a fresh pool of `width` threads, itself included, and on destruction
 * restores the width it found, again with a fresh pool.
 * @details The pool is made by the first parallel_for after a reset, at the width of that moment; a width change alone
 * leaves an existing pool as it is.
 */
class FreshParallelForPool {
  public:
    explicit FreshParallelForPool(const size_t width)
        : original_width_(get_num_cpus())
    {
        reset(width);
    }
    FreshParallelForPool(const FreshParallelForPool&) = delete;
    FreshParallelForPool(FreshParallelForPool&&) = delete;
    FreshParallelForPool& operator=(const FreshParallelForPool&) = delete;
    FreshParallelForPool& operator=(FreshParallelForPool&&) = delete;
    ~FreshParallelForPool() { reset(original_width_); }

  private:
    static void reset(const size_t width)
    {
        set_parallel_for_concurrency(width);
        parallel_for_stop_workers();
    }

    size_t original_width_;
};
#endif

} // namespace

TEST(StrausNativeHints, ThreadedHintsEqualTheSerialOnes)
{
    struct Case {
        std::string label;
        Reads reads;
        bool has_infinity_entry;
    };
    std::vector<Case> cases;
    cases.reserve(7);
    cases.push_back({ "random, small", random_reads(SMALL_NUM_POINTS), false });
    cases.push_back({ "random, large", random_reads(LARGE_NUM_POINTS), false });
    for (const size_t num_points : { SMALL_NUM_POINTS, LARGE_NUM_POINTS }) {
        for (const bool cancel : { false, true }) {
            cases.push_back({ (cancel ? "edges with a transcript entry at infinity, " : "edges, ") +
                                  std::string(num_points == SMALL_NUM_POINTS ? "small" : "large"),
                              edge_reads(num_points, cancel),
                              cancel });
        }
    }
    cases.push_back({ "a round ending at infinity in a later chunk", later_chunk_infinity_reads(), true });
    for (const Case& test_case : cases) {
        const StrausTableReads reads = test_case.reads.reads();
        const StrausNativeHints serial = compute_straus_hints_serial(reads);
        const auto serial_hints = affine_hints(serial);
        EXPECT_EQ(
            std::ranges::any_of(serial_hints, [](const AffineElement& hint) { return hint.is_point_at_infinity(); }),
            test_case.has_infinity_entry)
            << test_case.label;
        expect_same_hints(compute_straus_hints_parallel(reads), serial, test_case.label + ", threaded");
        expect_same_hints(compute_straus_hints(reads), serial, test_case.label + ", dispatched");
    }
}

// A pool of one thread computes the hints serially and larger pools in parallel; the circuits must be the same.
TEST(StrausNativeHints, LargeBatchMulCircuitDoesNotDependOnThreadCount)
{
    constexpr size_t table_bits = 4;
    constexpr size_t num_points = 1025;
    constexpr size_t num_reads = numeric::ceil_div(cycle_scalar_ct::NUM_BITS, table_bits) * num_points;
    static_assert(num_reads >= STRAUS_HINTS_MIN_PARALLEL_READS);
    const auto points = Group::derive_generators("straus_native_hints_circuit_test", num_points);
    auto& engine = numeric::get_debug_randomness();
    std::vector<bb::grumpkin::fr> scalars(num_points);
    for (auto& scalar : scalars) {
        scalar = bb::grumpkin::fr::random_element(&engine);
    }
    // A zero scalar reads every table's first entry, its offset generator.
    scalars[5] = 0;

#if defined(NO_MULTITHREADING) || defined(OMP_MULTITHREADING)
    const Builder reference = build_batch_mul(points, scalars);
    EXPECT_TRUE(CircuitChecker::check(reference));
    GTEST_SKIP() << "builds without threads or with OpenMP have no parallel_for pool whose width the test can set";
#else
    // After each build the pool it ran on is still the calling thread's, so its worker count shows the build's width.
    const size_t original_width = get_num_cpus();
    const Builder reference = [&] {
        const FreshParallelForPool pool(1);
        EXPECT_FALSE(straus_hints_use_threads(num_reads));
        Builder built = build_batch_mul(points, scalars);
        EXPECT_EQ(parallel_for_num_workers(), 0U);
        return built;
    }();
    EXPECT_TRUE(CircuitChecker::check(reference));
    for (const size_t width : { size_t{ 4 }, std::max<size_t>(original_width, 16) }) {
        SCOPED_TRACE("width " + std::to_string(width));
        const FreshParallelForPool pool(width);
        EXPECT_TRUE(straus_hints_use_threads(num_reads));
        const Builder builder = build_batch_mul(points, scalars);
        EXPECT_EQ(parallel_for_num_workers(), width - 1);
        test_utils::expect_same_builder(builder, reference);
    }
#endif
}

} // namespace bb::stdlib
