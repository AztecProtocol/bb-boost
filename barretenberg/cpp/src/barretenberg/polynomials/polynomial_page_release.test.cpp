#include <algorithm>
#include <array>
#include <cstddef>
#include <gtest/gtest.h>

#include "barretenberg/polynomials/polynomial.hpp"

// A polynomial listed twice still counts as one owner of its memory, so memory also held outside the list keeps its
// contents while the memory only the list holds is released.
TEST(Polynomial, ReleasePagesLeavesMemoryHeldOutsideTheList)
{
// Released anonymous pages read back as zeros only on Linux.
#if !defined(__linux__) || defined(NO_MULTITHREADING)
    GTEST_SKIP() << "needs Linux and native threads";
#else
    if (slow_low_memory) {
        GTEST_SKIP() << "file-backed memory is never released";
    }
    using FF = bb::fr;
    using Polynomial = bb::Polynomial<FF>;
    constexpr size_t MIB_ELEMENTS = (size_t{ 1 } << 20) / sizeof(FF);
    const FF fill(7);
    // `sole` alone reaches the 64 MiB minimum of a release; `held` is at the 32 MiB floor of a block, so it would be
    // released too if its owner outside the list were missed.
    Polynomial sole(64 * MIB_ELEMENTS, Polynomial::DontZeroMemory::FLAG);
    Polynomial held(32 * MIB_ELEMENTS, Polynomial::DontZeroMemory::FLAG);
    std::fill(sole.data(), sole.data() + sole.size(), fill);
    std::fill(held.data(), held.data() + held.size(), fill);
    const Polynomial outside = held.share();

    Polynomial::release_pages_before_destruction(std::array<const Polynomial*, 3>{ &sole, &held, &held });

    EXPECT_EQ(sole.get(sole.size() / 2), FF::zero());
    EXPECT_EQ(std::count(outside.data(), outside.data() + outside.size(), fill),
              static_cast<std::ptrdiff_t>(outside.size()));
#endif
}
