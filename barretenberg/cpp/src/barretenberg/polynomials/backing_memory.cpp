// === AUDIT STATUS ===
// internal:    { status: Complete, auditors: [Nishat], commit: 94f596f8b3bbbc216f9ad7dc33253256141156b2 }
// external_1:  { status: not started, auditors: [], commit: }
// external_2:  { status: not started, auditors: [], commit: }
// =====================

#include "barretenberg/polynomials/backing_memory.hpp"
#include "barretenberg/common/throw_or_abort.hpp"
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <string>
#if !defined(__wasm__) && !defined(_WIN32) && !defined(NO_MULTITHREADING) && !defined(BB_NO_EXCEPTIONS)
#include "barretenberg/common/thread.hpp"
#include <algorithm>
#include <cstdint>
#include <sys/mman.h>
#include <unistd.h>
#define BB_POLYNOMIAL_MEMORY_RELEASE 1
#endif

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
bool slow_low_memory =
    std::getenv("BB_SLOW_LOW_MEMORY") == nullptr ? false : std::string(std::getenv("BB_SLOW_LOW_MEMORY")) == "1";

// Storage budget is disabled for WASM builds unless ENABLE_WASM_BENCH is defined
#if !defined(__wasm__) || defined(ENABLE_WASM_BENCH)

#if defined(__wasm__) && defined(BB_NO_EXCEPTIONS)
// Simplified version for WASM builds without exception support
// For WASM benchmarking, we don't enforce storage budgets
size_t parse_size_string(const std::string& /* size_str */)
{
    // Return unlimited budget for WASM builds
    return std::numeric_limits<size_t>::max();
}
#else
// Full version with exception handling for native builds
// Parse storage size string (e.g., "500m", "2g", "1024k")
size_t parse_size_string(const std::string& size_str)
{
    if (size_str.empty()) {
        return std::numeric_limits<size_t>::max();
    }

    try {
        std::string str = size_str;

        // Convert to lowercase for case-insensitive comparison
        char suffix = static_cast<char>(std::tolower(static_cast<unsigned char>(str.back())));
        size_t multiplier = 1;

        // Check for unit suffix
        if (suffix == 'k') {
            multiplier = 1024ULL;
            str.pop_back();
        } else if (suffix == 'm') {
            multiplier = 1024ULL * 1024ULL;
            str.pop_back();
        } else if (suffix == 'g') {
            multiplier = 1024ULL * 1024ULL * 1024ULL;
            str.pop_back();
        } else if (std::isdigit(static_cast<unsigned char>(suffix)) == 0) {
            // Invalid suffix
            throw_or_abort("Invalid storage size format: '" + size_str + "'. Use format like '500m', '2g', or '1024k'");
        }

        // Check if remaining string is a valid number
        if (str.empty()) {
            throw_or_abort("Invalid storage size format: '" + size_str + "'. No numeric value provided");
        }

        size_t value = std::stoull(str);
        // Guard against value * multiplier silently wrapping modulo 2^64 for very large inputs
        if (multiplier != 0 && value > std::numeric_limits<size_t>::max() / multiplier) {
            throw_or_abort("Invalid storage size format: '" + size_str + "'. Value out of range");
        }
        return value * multiplier;
    } catch (const std::invalid_argument&) {
        throw_or_abort("Invalid storage size format: '" + size_str + "'. Not a valid number");
    } catch (const std::out_of_range&) {
        throw_or_abort("Invalid storage size format: '" + size_str + "'. Value out of range");
    }
}
#endif

namespace {
// Parse storage budget from environment variable (supports k/m/g suffixes like Docker)
size_t parse_storage_budget()
{
    const char* env_val = std::getenv("BB_STORAGE_BUDGET");
    if (env_val == nullptr) {
        return std::numeric_limits<size_t>::max(); // No limit by default
    }

    return parse_size_string(std::string(env_val));
}
} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
size_t storage_budget = parse_storage_budget();

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::atomic<size_t> current_storage_usage{ 0 };

#endif // !defined(__wasm__) || defined(ENABLE_WASM_BENCH)

namespace bb {
#ifdef BB_POLYNOMIAL_MEMORY_RELEASE
namespace {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local PolynomialMemoryRelease* active_release = nullptr;

// Splits the whole pages of all ranges evenly over the tasks; each task walks the ranges its share overlaps.
void release_in_parallel(const std::vector<std::pair<uintptr_t, size_t>>& ranges, size_t page)
{
    constexpr size_t BYTES_PER_TASK = size_t{ 16 } << 20;
    constexpr size_t MAX_TASKS = 16;
    std::vector<size_t> first_page(ranges.size());
    size_t total_pages = 0;
    for (size_t r = 0; r < ranges.size(); ++r) {
        first_page[r] = total_pages;
        total_pages += ranges[r].second;
    }
    const size_t num_tasks = std::min((total_pages * page) / BYTES_PER_TASK, MAX_TASKS);
    if (num_tasks < 2) {
        return;
    }
    parallel_for(num_tasks, [&](size_t t) {
        size_t lo = (total_pages * t) / num_tasks;
        const size_t hi = (total_pages * (t + 1)) / num_tasks;
        auto r =
            static_cast<size_t>(std::upper_bound(first_page.begin(), first_page.end(), lo) - first_page.begin()) - 1;
        while (lo < hi) {
            const size_t offset = lo - first_page[r];
            const size_t count = std::min(hi - lo, ranges[r].second - offset);
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            static_cast<void>(
                madvise(reinterpret_cast<void*>(ranges[r].first + (offset * page)), count * page, MADV_DONTNEED));
            lo += count;
            ++r;
        }
    });
}
} // namespace
#endif

bool PolynomialMemoryRelease::defer_free([[maybe_unused]] void* ptr,
                                         [[maybe_unused]] size_t bytes,
                                         [[maybe_unused]] void (*free_block)(void*)) noexcept
{
#ifdef BB_POLYNOMIAL_MEMORY_RELEASE
    PolynomialMemoryRelease* scope = active_release;
    if (scope == nullptr) {
        return false;
    }
    try {
        scope->blocks_.push_back({ ptr, bytes, free_block });
    } catch (...) {
        return false;
    }
    return true;
#else
    return false;
#endif
}

void PolynomialMemoryRelease::activate()
{
#ifdef BB_POLYNOMIAL_MEMORY_RELEASE
    if (!active_) {
        previous_ = active_release;
        active_release = this;
        active_ = true;
    }
#endif
}

PolynomialMemoryRelease::~PolynomialMemoryRelease()
{
#ifdef BB_POLYNOMIAL_MEMORY_RELEASE
    if (!active_) {
        return;
    }
    active_release = previous_;
    // Only whole pages strictly inside each block, so the allocator's chunk headers next to it stay intact.
    try {
        const auto page = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
        std::vector<std::pair<uintptr_t, size_t>> ranges;
        ranges.reserve(blocks_.size());
        for (const auto& block : blocks_) {
            const auto start = reinterpret_cast<uintptr_t>(block.ptr);
            const uintptr_t begin = (start + page - 1) & ~(page - 1);
            const uintptr_t end = (start + block.bytes) & ~(page - 1);
            if (end > begin) {
                ranges.emplace_back(begin, (end - begin) / page);
            }
        }
        release_in_parallel(ranges, page);
    } catch (...) {
        // Without the release the blocks are simply freed below.
    }
    for (const auto& block : blocks_) {
        block.free_block(block.ptr);
    }
#endif
}
} // namespace bb
