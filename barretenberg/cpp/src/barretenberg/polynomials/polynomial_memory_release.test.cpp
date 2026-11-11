#include "barretenberg/common/thread.hpp"
#include "barretenberg/ecc/curves/bn254/fr.hpp"
#include "barretenberg/polynomials/backing_memory.hpp"
#include "barretenberg/polynomials/polynomial.hpp"
#include <gtest/gtest.h>
#include <optional>
#include <string>

// The condition under which backing_memory.cpp builds the release.
#if !defined(__wasm__) && !defined(_WIN32) && !defined(NO_MULTITHREADING) && !defined(BB_NO_EXCEPTIONS)
#define RELEASE_IS_BUILT 1
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <set>
#include <thread>
#include <unistd.h>
#endif

using namespace bb;

namespace {
using Fr = bb::fr;
constexpr size_t SIZE = size_t{ 1 } << 21; // 64 MiB per polynomial, enough for the release to run in parallel

Polynomial<Fr> make_polynomial(uint64_t seed)
{
    Polynomial<Fr> poly(SIZE);
    parallel_for_range(SIZE, [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            poly.at(i) = Fr(seed + i);
        }
    });
    return poly;
}

// Polynomials dropped inside an active scope are freed when it ends; a block that still has an owner outside the scope
// is never released, so its contents survive.
void expect_shared_block_survives(const std::string& label)
{
    std::optional<Polynomial<Fr>> kept;
    {
        PolynomialMemoryRelease release;
        auto dropped = make_polynomial(1);
        auto shared = make_polynomial(2);
        kept = shared.share();
        auto dropped_after = make_polynomial(3);
        release.activate();
    }
    ASSERT_TRUE(kept.has_value());
    size_t mismatches = 0;
    for (size_t i = 0; i < SIZE; ++i) {
        mismatches += static_cast<size_t>(kept->get(i) != Fr(2 + i));
    }
    EXPECT_EQ(mismatches, 0U) << label;
}

#ifdef RELEASE_IS_BUILT
constexpr size_t BLOCK_BYTES = size_t{ 64 } << 20;

struct FreedBlock {
    size_t frees = 0;
    size_t nonzero_interior_bytes = 0;
};
// The scope frees its blocks on its own thread.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local FreedBlock freed_block;

void inspect_and_free(void* ptr)
{
    auto* bytes = static_cast<uint8_t*>(ptr);
    const auto page = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
    const auto start = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t begin = ((start + page - 1) & ~(page - 1)) - start;
    const uintptr_t end = ((start + BLOCK_BYTES) & ~(page - 1)) - start;
    freed_block.nonzero_interior_bytes +=
        static_cast<size_t>(std::count_if(bytes + begin, bytes + end, [](uint8_t b) { return b != 0; }));
    ++freed_block.frees;
    delete[] bytes;
}

// A collected block is freed exactly once, when the scope ends, after every whole page inside it has been released.
void expect_collected_block_released(const std::string& label)
{
    auto* block = new uint8_t[BLOCK_BYTES];
    std::memset(block, 0xAB, BLOCK_BYTES);
    freed_block = {};
    {
        PolynomialMemoryRelease release;
        release.activate();
        if (!PolynomialMemoryRelease::defer_free(block, BLOCK_BYTES, inspect_and_free)) {
            delete[] block;
            FAIL() << label << ": the active scope did not collect the block";
        }
        EXPECT_EQ(freed_block.frees, 0U) << label;
    }
    EXPECT_EQ(freed_block.frees, 1U) << label;
#ifdef __linux__
    // Linux discards private anonymous pages on MADV_DONTNEED, so released pages read back as zero; Darwin keeps them.
    EXPECT_EQ(freed_block.nonzero_interior_bytes, 0U) << label;
#endif
}

// Number of distinct threads that run iterations of a parallel_for on the calling thread's pool.
size_t threads_in_pool()
{
    constexpr size_t ITERATIONS = 48;
    std::mutex mutex;
    std::set<std::thread::id> ids;
    parallel_for(ITERATIONS, [&](size_t) {
        const std::lock_guard<std::mutex> lock(mutex);
        ids.insert(std::this_thread::get_id());
    });
    return ids.size();
}
#endif
} // namespace

TEST(PolynomialMemoryRelease, ReleasesCollectedBlocksAndKeepsSharedOnes)
{
#ifndef RELEASE_IS_BUILT
    expect_shared_block_survives("release not built");
#else
    // The concurrency setting and the parallel_for pool are both per thread, and the pool is sized by the thread's
    // first parallel_for, so each setting is applied on a fresh thread before anything else runs there. The outcomes
    // prove the release: one free per collected block, its whole interior pages zero on Linux, a shared block intact.
    // How many pool threads join in depends on scheduling, so only the upper bound is checked.
    for (const size_t concurrency : { size_t{ 16 }, size_t{ 3 }, size_t{ 1 } }) {
        std::thread([concurrency] {
            set_parallel_for_concurrency(concurrency);
            const std::string label = "concurrency " + std::to_string(concurrency);
            EXPECT_EQ(get_num_cpus(), concurrency) << label;
            EXPECT_LE(threads_in_pool(), concurrency) << label;
            expect_shared_block_survives(label);
            expect_collected_block_released(label);
        }).join();
    }
#endif
}
