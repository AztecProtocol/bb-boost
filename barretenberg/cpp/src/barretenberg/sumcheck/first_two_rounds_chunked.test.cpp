#include "barretenberg/flavor/ultra_flavor.hpp"
#include "barretenberg/sumcheck/sumcheck.hpp"

#include <array>
#include <cstddef>
#include <gtest/gtest.h>
#include <vector>

using namespace bb;

// The chunked two-round fold against the per-polynomial fold on a complete table (every polynomial backed, shifted
// views included, extents that do and do not divide the chunk) at 1, 4 and 16 threads, each on a fresh pool. The
// reference shares the arithmetic kernel, so this is a scheduling reference: it checks chunk coverage, boundaries and
// end indices. Destinations are poisoned first, so an output no chunk writes cannot match by leftover memory.
TEST(SumcheckFirstTwoRounds, ChunkedMatchesPerPolynomial)
{
    using Flavor = UltraFlavor;
    using FF = typename Flavor::FF;
    using Polynomial = typename Flavor::Polynomial;
    using PartiallyEvaluated = typename Flavor::PartiallyEvaluatedMultivariates;
    constexpr size_t multivariate_d = 8;
    constexpr size_t multivariate_n = size_t{ 1 } << multivariate_d;
    constexpr std::array<size_t, 4> starts{ 1, 2, 5, 9 };
    constexpr std::array<size_t, 5> ends{ 256, 243, 130, 13, 66 };

    typename Flavor::ProverPolynomials full;
    size_t i = 0;
    for (auto& poly : full.get_unshifted()) {
        const size_t start = starts[i % starts.size()];
        const size_t end = ends[i % ends.size()];
        poly = Polynomial(end - start, multivariate_n, start);
        for (size_t row = start; row < end; ++row) {
            poly.at(row) = FF::random_element();
        }
        ++i;
    }
    full.set_shifted();

    const FF u0 = FF::random_element();
    const FF u1 = FF::random_element();
    const FF poison = FF::random_element();
    PartiallyEvaluated expected(full, multivariate_n, /*num_folds=*/2);
    for (auto [src, dst] : zip_view(full.get_all(), expected.get_all())) {
        fold_stride4_backed(src, dst, u0, u1);
    }
    auto count_mismatches = [&](PartiallyEvaluated& folded) {
        size_t mismatches = 0;
        for (auto [got, want] : zip_view(folded.get_all(), expected.get_all())) {
            EXPECT_EQ(got.start_index(), want.start_index());
            EXPECT_EQ(got.end_index(), want.end_index());
            for (size_t k = want.start_index(); k < want.end_index(); ++k) {
                mismatches += got[k] == want[k] ? 0 : 1;
            }
        }
        return mismatches;
    };

#ifdef NO_MULTITHREADING
    const std::vector<size_t> thread_counts{ 1 };
#else
    const std::vector<size_t> thread_counts{ 1, 4, 16 };
    const size_t saved_concurrency = get_num_cpus();
#endif
    for (const size_t outputs_per_chunk : { size_t{ 16 }, FOLD_STRIDE4_OUTPUTS_PER_CHUNK }) {
        for (const size_t threads : thread_counts) {
#ifndef NO_MULTITHREADING
            set_parallel_for_concurrency(threads);
            parallel_for_stop_workers();
#endif
            PartiallyEvaluated folded(full, multivariate_n, /*num_folds=*/2);
            for (auto& poly : folded.get_all()) {
                for (size_t k = poly.start_index(); k < poly.end_index(); ++k) {
                    poly.at(k) = poison;
                }
            }
            fold_stride4_backed_all(full.get_all(), folded.get_all(), u0, u1, outputs_per_chunk);
            EXPECT_EQ(count_mismatches(folded), 0UL)
                << threads << " threads, " << outputs_per_chunk << " outputs per chunk";
        }
    }
#ifndef NO_MULTITHREADING
    set_parallel_for_concurrency(saved_concurrency);
    parallel_for_stop_workers();
#endif

    auto transcript = Flavor::Transcript::test_prover_init_empty();
    SumcheckProver<Flavor> sumcheck(
        multivariate_n, full, transcript, FF(1), std::vector<FF>(multivariate_d, FF(1)), {}, multivariate_d);
    auto folded = sumcheck.partially_evaluate_first_two_rounds(full, u0, u1);
    EXPECT_EQ(count_mismatches(folded), 0UL);
}
