#include "barretenberg/common/release_pages.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <span>
#include <vector>

// Released anonymous pages read back as zeros only on Linux.
#if defined(__linux__) && !defined(NO_MULTITHREADING)
#define BB_RELEASE_PAGES_TESTABLE 1
#include <sys/mman.h>
#include <unistd.h>
#endif

using namespace bb;

namespace {

#ifdef BB_RELEASE_PAGES_TESTABLE
constexpr unsigned char FILL = 0xAB;
constexpr size_t MIB = size_t{ 1 } << 20;

class Mapping {
  public:
    explicit Mapping(size_t bytes)
        : bytes_(bytes)
        , base_(static_cast<unsigned char*>(
              mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)))
    {
        if (base_ == MAP_FAILED) {
            base_ = nullptr;
            ADD_FAILURE() << "mmap failed";
            return;
        }
        std::memset(base_, FILL, bytes_);
    }
    Mapping(const Mapping&) = delete;
    Mapping(Mapping&&) = delete;
    Mapping& operator=(const Mapping&) = delete;
    Mapping& operator=(Mapping&&) = delete;
    ~Mapping()
    {
        if (base_ != nullptr) {
            munmap(base_, bytes_);
        }
    }

    unsigned char* at(size_t offset) const { return base_ + offset; }
    size_t size() const { return bytes_; }

  private:
    size_t bytes_;
    unsigned char* base_;
};

size_t page_size()
{
    return static_cast<size_t>(sysconf(_SC_PAGESIZE));
}

// Checks page by page that exactly the whole pages strictly inside `released` read as zero and every other byte of the
// mapping still holds FILL.
void expect_only_whole_pages_released(const Mapping& mapping, const std::vector<PageBlock>& released)
{
    ASSERT_NE(mapping.at(0), nullptr);
    const size_t page = page_size();
    const std::vector<unsigned char> filled(page, FILL);
    const std::vector<unsigned char> zeros(page, 0);
    const auto base = reinterpret_cast<uintptr_t>(mapping.at(0));
    for (size_t offset = 0; offset < mapping.size(); offset += page) {
        const uintptr_t lo = base + offset;
        bool is_released = false;
        for (const PageBlock& block : released) {
            const auto begin = reinterpret_cast<uintptr_t>(block.ptr);
            is_released = is_released || (lo >= begin && lo + page <= begin + block.bytes);
        }
        const auto& expected = is_released ? zeros : filled;
        ASSERT_EQ(std::memcmp(mapping.at(offset), expected.data(), page), 0)
            << "page at offset " << offset << " expected " << (is_released ? "released" : "untouched");
    }
}
#endif

} // namespace

TEST(ReleasePages, ListReleasesWholePagesOfLargeBlocksOnly)
{
#ifndef BB_RELEASE_PAGES_TESTABLE
    GTEST_SKIP() << "needs Linux and native threads";
#else
    const size_t page = page_size();
    const size_t large = (32 * MIB) + (4 * page);
    Mapping mapping((2 * large) + MIB + (2 * page));

    // Both large blocks start and end mid-page; the small one is page aligned but under the per-block floor.
    const PageBlock first{ mapping.at(100), (32 * MIB) + 1234 };
    const PageBlock second{ mapping.at(large + 7), (32 * MIB) + (2 * page) };
    const PageBlock small{ mapping.at(2 * large), MIB };
    const PageBlock empty{ mapping.at(0), 0 };
    const PageBlock null_block{ nullptr, 64 * MIB };

    PageBlockList<3> list;
    EXPECT_FALSE(list.add(empty.ptr, empty.bytes));
    EXPECT_TRUE(list.add(first.ptr, first.bytes));
    EXPECT_FALSE(list.add(small.ptr, small.bytes));
    EXPECT_FALSE(list.add(null_block.ptr, null_block.bytes));
    EXPECT_TRUE(list.add(second.ptr, second.bytes));
    EXPECT_TRUE(list.release());
    expect_only_whole_pages_released(mapping, { first, second });

    // The list form applies the same rules to blocks it is handed directly.
    Mapping again((2 * large) + MIB + (2 * page));
    const std::vector<PageBlock> blocks{ { again.at(2 * large), MIB },
                                         { again.at(100), (32 * MIB) + 1234 },
                                         { nullptr, 64 * MIB },
                                         { again.at(0), 0 },
                                         { again.at(large + 7), (32 * MIB) + (2 * page) } };
    EXPECT_TRUE(release_pages_in_parallel(std::span<const PageBlock>(blocks)));
    expect_only_whole_pages_released(again, { blocks[1], blocks[4] });
#endif
}

TEST(ReleasePages, SmallTotalsAndFullListsAreLeftAlone)
{
#ifndef BB_RELEASE_PAGES_TESTABLE
    GTEST_SKIP() << "needs Linux and native threads";
#else
    Mapping mapping(64 * MIB);

    // One eligible block of 40 MiB plus a block under the floor total less than the 64 MiB minimum.
    const std::vector<PageBlock> blocks{ { mapping.at(0), 40 * MIB }, { mapping.at(40 * MIB), 24 * MIB } };
    EXPECT_FALSE(release_pages_in_parallel(std::span<const PageBlock>(blocks)));
    expect_only_whole_pages_released(mapping, {});

    // A full list drops the extra block rather than growing; one 32 MiB block is again under the minimum.
    PageBlockList<1> list;
    EXPECT_TRUE(list.add(mapping.at(0), 32 * MIB));
    EXPECT_FALSE(list.add(mapping.at(32 * MIB), 32 * MIB));
    EXPECT_FALSE(list.release());
    expect_only_whole_pages_released(mapping, {});

    // The single-block form keeps its own 64 MiB minimum.
    release_pages_in_parallel(mapping.at(1), (64 * MIB) - 1);
    expect_only_whole_pages_released(mapping, {});
#endif
}
