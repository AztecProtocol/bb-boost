#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#if !defined(__wasm__) && !defined(_WIN32) && !defined(NO_MULTITHREADING)
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace bb {

/**
 * @brief A block of memory whose pages may be handed back to the OS just before it is freed.
 */
struct PageBlock {
    void* ptr = nullptr;
    // The allocation's size, which decides whether the block is large enough to release.
    size_t bytes = 0;
    // The bytes in use from ptr, at most `bytes`; only pages inside them are released and counted.
    size_t used_bytes = 0;

    constexpr PageBlock() noexcept = default;
    constexpr PageBlock(void* block, size_t block_bytes) noexcept
        : PageBlock(block, block_bytes, block_bytes)
    {}
    constexpr PageBlock(void* block, size_t block_bytes, size_t block_used_bytes) noexcept
        : ptr(block)
        , bytes(block_bytes)
        , used_bytes(std::min(block_used_bytes, block_bytes))
    {}
};

/**
 * @brief The smallest block the list form of release_pages_in_parallel releases.
 * @details glibc gives a block a mapping of its own only at or above its dynamic mmap threshold, which rises to at most
 * 32 MiB on 64-bit. A smaller block may sit inside a malloc arena, where free() would not unmap it, and releasing its
 * pages only makes the next allocation that reuses that memory fault them in again.
 */
inline constexpr size_t MIN_PAGE_RELEASE_BLOCK_BYTES = size_t{ 32 } << 20;

namespace release_pages_detail {

inline constexpr size_t MIN_PARALLEL_RELEASE_BYTES = size_t{ 64 } << 20;
inline constexpr size_t BYTES_PER_THREAD = size_t{ 16 } << 20;
inline constexpr size_t MAX_THREADS = 16;

#if !defined(__wasm__) && !defined(_WIN32) && !defined(NO_MULTITHREADING) && defined(MADV_DONTNEED)
struct PageRange {
    uintptr_t begin = 0;
    uintptr_t end = 0;
};

// The whole pages strictly inside an eligible block's used bytes, never allocator metadata next to the block.
inline PageRange whole_pages(const PageBlock& block, uintptr_t page)
{
    if (block.ptr == nullptr || block.bytes < MIN_PAGE_RELEASE_BLOCK_BYTES) {
        return {};
    }
    const auto lo = reinterpret_cast<uintptr_t>(block.ptr);
    const uintptr_t begin = (lo + page - 1) & ~(page - 1);
    const uintptr_t end = (lo + std::min(block.used_bytes, block.bytes)) & ~(page - 1);
    return end > begin ? PageRange{ begin, end } : PageRange{};
}

// Pages [first_page, end_page) of the eligible blocks' whole pages, numbered consecutively across blocks.
struct Slice {
    const PageBlock* blocks = nullptr;
    size_t num_blocks = 0;
    uintptr_t page = 0;
    size_t first_page = 0;
    size_t end_page = 0;
};

inline void* release_slice(void* arg)
{
    const auto* slice = static_cast<const Slice*>(arg);
    size_t block_first_page = 0;
    for (size_t i = 0; i < slice->num_blocks && block_first_page < slice->end_page; ++i) {
        const PageRange range = whole_pages(slice->blocks[i], slice->page);
        const size_t num_pages = (range.end - range.begin) / slice->page;
        const size_t lo = std::max(block_first_page, slice->first_page);
        const size_t hi = std::min(block_first_page + num_pages, slice->end_page);
        if (lo < hi) {
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)
            auto* start = reinterpret_cast<void*>(range.begin + ((lo - block_first_page) * slice->page));
            static_cast<void>(madvise(start, (hi - lo) * slice->page, MADV_DONTNEED));
        }
        block_first_page += num_pages;
    }
    return nullptr;
}
#endif

} // namespace release_pages_detail

/**
 * @brief Before large blocks are freed, hand their pages back to the OS from one team of threads.
 *
 * @details Freeing a multi-megabyte block unmaps it on the calling thread, and the kernel then frees its pages one at a
 * time (on the order of 0.1 ms per MiB), so gigabytes of memory take a noticeable serial pause to tear down.
 * madvise(MADV_DONTNEED) on disjoint ranges runs concurrently, after which each free only unmaps an empty range. Only
 * whole pages strictly inside each block's used bytes are released, split evenly over up to 16 threads, one per 16 MiB,
 * whatever the block boundaries. Blocks whose allocation is under MIN_PAGE_RELEASE_BLOCK_BYTES (32 MiB) are skipped,
 * since the allocator may keep them in a malloc arena rather than unmap them on free, and nothing is released when the
 * used bytes of the remaining blocks total under 64 MiB, so a large reserve that was never written starts no threads.
 * The released blocks' contents are lost: call this only immediately before freeing them. Never throws, and uses
 * short-lived threads rather than the parallel_for pool, so it is safe from destructors running on any thread,
 * including inside a parallel_for. A no-op without native threads.
 *
 * @return Whether the release was attempted: true once the qualifying pages have been handed to madvise, whether or not
 * every call succeeded, so the caller need not release them again; false when nothing qualified or without native
 * threads.
 */
inline bool release_pages_in_parallel([[maybe_unused]] std::span<const PageBlock> blocks) noexcept
{
#if !defined(__wasm__) && !defined(_WIN32) && !defined(NO_MULTITHREADING) && defined(MADV_DONTNEED)
    using namespace release_pages_detail;
    const auto page = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
    size_t eligible_bytes = 0;
    size_t num_pages = 0;
    for (const PageBlock& block : blocks) {
        const PageRange range = whole_pages(block, page);
        if (range.end > range.begin) {
            eligible_bytes += std::min(block.used_bytes, block.bytes);
            num_pages += (range.end - range.begin) / page;
        }
    }
    if (eligible_bytes < MIN_PARALLEL_RELEASE_BYTES || num_pages == 0) {
        return false;
    }
    const size_t num_threads = std::clamp<size_t>((num_pages * page) / BYTES_PER_THREAD, 1, MAX_THREADS);
    std::array<Slice, MAX_THREADS> slices{};
    for (size_t t = 0; t < num_threads; ++t) {
        slices[t] = {
            blocks.data(), blocks.size(), page, (num_pages * t) / num_threads, (num_pages * (t + 1)) / num_threads
        };
    }
    // pthread_create reports failure by its return value and the handles live in a fixed array, so nothing here can
    // throw, including a failed allocation: this runs inside deleters and noexcept releases.
    std::array<pthread_t, MAX_THREADS> threads{};
    size_t spawned = 1;
    while (spawned < num_threads && pthread_create(&threads[spawned], nullptr, release_slice, &slices[spawned]) == 0) {
        ++spawned;
    }
    // Out of threads: this thread releases the slices nobody took.
    for (size_t t = spawned; t < num_threads; ++t) {
        release_slice(&slices[t]);
    }
    release_slice(slices.data());
    for (size_t t = 1; t < spawned; ++t) {
        pthread_join(threads[t], nullptr);
    }
    return true;
#else
    return false;
#endif
}

/**
 * @brief Before a large block is freed, hand its pages back to the OS from several threads at once.
 *
 * @details The single-block form of release_pages_in_parallel above, for a block of at least 64 MiB; smaller blocks are
 * left to the allocator.
 */
inline void release_pages_in_parallel(void* ptr, size_t bytes)
{
    if (ptr == nullptr || bytes < release_pages_detail::MIN_PARALLEL_RELEASE_BYTES) {
        return;
    }
    const PageBlock block{ ptr, bytes };
    static_cast<void>(release_pages_in_parallel(std::span<const PageBlock>(&block, 1)));
}

/**
 * @brief A fixed-capacity list of blocks for the list form of release_pages_in_parallel; adding never allocates or
 * throws, so it can be filled inside destructors.
 */
template <size_t N> class PageBlockList {
  public:
    /**
     * @brief Adds a block of `bytes` whose first `used_bytes` are in use, if the block is at least
     * MIN_PAGE_RELEASE_BLOCK_BYTES and the list has room; only pages inside the used bytes are released.
     * @return Whether the block was added; a block that was not is left to the allocator.
     */
    bool add(void* ptr, size_t bytes, size_t used_bytes) noexcept
    {
        if (ptr == nullptr || bytes < MIN_PAGE_RELEASE_BLOCK_BYTES || size_ == N) {
            return false;
        }
        blocks_[size_++] = PageBlock{ ptr, bytes, used_bytes };
        return true;
    }

    /**
     * @brief Adds a block that is in use in full; see the three-argument add.
     */
    bool add(void* ptr, size_t bytes) noexcept { return add(ptr, bytes, bytes); }

    /**
     * @brief Releases the pages of every added block (see release_pages_in_parallel); call it only immediately before
     * freeing them.
     * @return Whether the release was attempted (see release_pages_in_parallel).
     */
    bool release() const noexcept
    {
        return release_pages_in_parallel(std::span<const PageBlock>(blocks_.data(), size_));
    }

  private:
    std::array<PageBlock, N> blocks_{};
    size_t size_ = 0;
};

} // namespace bb
