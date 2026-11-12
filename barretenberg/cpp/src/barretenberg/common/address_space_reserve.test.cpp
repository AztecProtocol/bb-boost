#include "barretenberg/common/address_space_reserve.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>

using namespace bb;

#ifdef BB_ADDRESS_SPACE_RESERVE
namespace {

// Throws std::bad_alloc for any allocation above BUDGET_BYTES.
template <typename T> struct BudgetAllocator {
    static constexpr size_t BUDGET_BYTES = 2048;
    using value_type = T;
    BudgetAllocator() = default;
    template <typename U> BudgetAllocator(const BudgetAllocator<U>& /*other*/) noexcept {}
    T* allocate(size_t n)
    {
        if (n > BUDGET_BYTES / sizeof(T)) {
            throw std::bad_alloc();
        }
        return std::allocator<T>{}.allocate(n);
    }
    void deallocate(T* p, size_t n) noexcept { std::allocator<T>{}.deallocate(p, n); }
    bool operator==(const BudgetAllocator& /*other*/) const noexcept { return true; }
};

} // namespace
#endif

TEST(AddressSpaceReserve, PolicyNeedsEveryFact)
{
    const std::array<std::pair<AddressSpaceFacts, bool>, 6> rows{ {
        { { 0, true, false }, true },
        { { 1, true, false }, true },
        { { 2, true, false }, false },
        { { std::nullopt, true, false }, false },
        { { 0, false, false }, false },
        { { 0, true, true }, false },
    } };
    for (const auto& [facts, allowed] : rows) {
        EXPECT_EQ(address_space_reservation_policy(facts), allowed);
    }
}

TEST(AddressSpaceReserve, RefusalReleasesEveryVector)
{
#ifdef BB_ADDRESS_SPACE_RESERVE
    using Narrow = std::vector<uint32_t, BudgetAllocator<uint32_t>>;
    Narrow a;
    Narrow b;
    Narrow c;
    Narrow d;
    std::vector<uint64_t, BudgetAllocator<uint64_t>> wide;
    // 512 entries fit the budget as uint32_t, not as uint64_t.
    EXPECT_FALSE(try_reserve_address_space(512, a, b, c, d, wide));
    EXPECT_EQ((std::array<size_t, 5>{ a.capacity(), b.capacity(), c.capacity(), d.capacity(), wide.capacity() }),
              (std::array<size_t, 5>{}));
    a.reserve(64);
    b.reserve(64);
    c.reserve(64);
    d.reserve(64);
    wide.reserve(64);
    EXPECT_GE(std::min({ a.capacity(), b.capacity(), c.capacity(), d.capacity(), wide.capacity() }), 64U);

    Narrow e;
    std::vector<uint64_t, BudgetAllocator<uint64_t>> f;
    EXPECT_TRUE(try_reserve_address_space(256, e, f));
    EXPECT_TRUE(e.capacity() >= 256 && f.capacity() >= 256);
#else
    GTEST_SKIP() << "address-space reservation is compiled out";
#endif
}

TEST(AddressSpaceReserve, RefusesVectorsThatHaveCapacity)
{
    std::vector<uint32_t> empty;
    std::vector<uint32_t> filled{ 1, 2, 3 };
    const size_t filled_capacity = filled.capacity();
    EXPECT_FALSE(try_reserve_address_space(512, empty, filled));
    EXPECT_EQ(empty.capacity(), 0U);
    EXPECT_EQ(filled.capacity(), filled_capacity);
    EXPECT_EQ(filled, (std::vector<uint32_t>{ 1, 2, 3 }));
}

TEST(AddressSpaceReserve, AllowedReadsThisProcess)
{
#ifdef BB_ADDRESS_SPACE_RESERVE
    AddressSpaceFacts facts;
    std::ifstream overcommit("/proc/sys/vm/overcommit_memory");
    int mode = 0;
    if (overcommit >> mode) {
        facts.overcommit_mode = mode;
    }
    rlimit limit{};
    facts.address_space_unlimited = ::getrlimit(RLIMIT_AS, &limit) == 0 && limit.rlim_cur == RLIM_INFINITY;
    facts.allocator_env_set = std::getenv("MALLOC_PERTURB_") != nullptr || std::getenv("GLIBC_TUNABLES") != nullptr;
    EXPECT_EQ(address_space_reservation_allowed(), address_space_reservation_policy(facts));
#else
    EXPECT_FALSE(address_space_reservation_allowed());
#endif
}
