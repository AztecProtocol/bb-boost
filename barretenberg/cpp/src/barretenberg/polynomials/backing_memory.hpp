// === AUDIT STATUS ===
// internal:    { status: Complete, auditors: [Nishat], commit: 94f596f8b3bbbc216f9ad7dc33253256141156b2 }
// external_1:  { status: not started, auditors: [], commit: }
// external_2:  { status: not started, auditors: [], commit: }
// =====================

#pragma once

#include "barretenberg/common/release_pages.hpp"
#include "barretenberg/common/throw_or_abort.hpp"
#include "unistd.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <functional>
#include <memory>
#if !defined(__wasm__) && !defined(_WIN32)
#include <sys/mman.h>
#endif

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern bool slow_low_memory;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern size_t storage_budget;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern std::atomic<size_t> current_storage_usage;

// Parse storage size string (e.g., "500m", "2g", "1024k")
size_t parse_size_string(const std::string& size_str);

template <typename Fr> struct BackingMemory {
    // Common raw data pointer used by all storage types
    Fr* raw_data = nullptr;

#if !defined(__wasm__) && !defined(_WIN32)
    // File-backed data substruct with cleanup metadata
    struct FileBackedData {
        size_t file_size;
        std::string filename;
        int fd;
        Fr* raw_data_ptr;

        ~FileBackedData()
        {
            if (raw_data_ptr != nullptr && file_size > 0) {
                munmap(raw_data_ptr, file_size);
                current_storage_usage.fetch_sub(file_size);
            }
            if (fd >= 0) {
                close(fd);
            }
            if (!filename.empty()) {
                std::filesystem::remove(filename);
            }
        }
    };
    std::shared_ptr<FileBackedData> file_backed;
#endif
    // Aligned memory data substruct
    std::shared_ptr<Fr[]> aligned_memory;

    BackingMemory() = default;

    BackingMemory(const BackingMemory&) = default;
    BackingMemory& operator=(const BackingMemory&) = default;

    BackingMemory(BackingMemory&& other) noexcept
        : raw_data(other.raw_data)
#if !defined(__wasm__) && !defined(_WIN32)
        , file_backed(std::move(other.file_backed))
#endif
        , aligned_memory(std::move(other.aligned_memory))
    {
        other.raw_data = nullptr;
    }

    BackingMemory& operator=(BackingMemory&& other) noexcept
    {
        if (this != &other) {
            raw_data = other.raw_data;
#if !defined(__wasm__) && !defined(_WIN32)
            file_backed = std::move(other.file_backed);
#endif
            aligned_memory = std::move(other.aligned_memory);
            other.raw_data = nullptr;
        }
        return *this;
    }

    // Allocate memory, preferring file-backed if in low memory mode.
    // Memory is NOT zeroed — callers that need zeroed memory must do so themselves.
    static BackingMemory allocate(size_t size)
    {
        BackingMemory memory;
#if !defined(__wasm__) && !defined(_WIN32)
        if (slow_low_memory) {
            if (try_allocate_file_backed(memory, size)) {
                return memory;
            }
        }
#endif
        allocate_aligned(memory, size);
        return memory;
    }

    ~BackingMemory() = default;

    /**
     * @brief Before the listed handles are destroyed together, release the pages of every in-memory block that no
     * other handle owns, with one team of threads for all of them (see release_pages_in_parallel).
     * @details A block qualifies when its owner count equals the number of distinct listed handles to it, so memory
     * shared with an object outside the list is left alone, however often a handle is listed. File-backed memory and
     * blocks not allocated by allocate() are never released. The caller guarantees that no other thread can acquire a
     * new reference to a listed block during the call; in particular, no weak_ptr to one may exist. The released
     * blocks' contents are lost, so this must run immediately before the handles are destroyed.
     */
    template <size_t N>
    static void release_pages_of_sole_owned(const std::array<const BackingMemory*, N>& handles) noexcept
    {
        // A handle holds one reference however often it is listed.
        std::array<const BackingMemory*, N> unique = handles;
        std::sort(unique.begin(), unique.end(), std::less<>{});
        const auto num_unique = static_cast<size_t>(std::unique(unique.begin(), unique.end()) - unique.begin());
        // A deleter lives in its block's shared control block, so equal deleters mean the same block.
        std::array<AlignedDeleter*, N> deleters{};
        for (size_t i = 0; i < num_unique; ++i) {
            deleters[i] = unique[i]->aligned_deleter();
        }
        bb::PageBlockList<N> blocks;
        std::array<AlignedDeleter*, N> added{};
        size_t num_added = 0;
        for (size_t i = 0; i < num_unique; ++i) {
            AlignedDeleter* deleter = deleters[i];
            if (deleter == nullptr || deleter->pages_released.load(std::memory_order_relaxed) ||
                std::find(deleters.data(), deleters.data() + i, deleter) != deleters.data() + i) {
                continue;
            }
            const auto num_listed = std::count(deleters.data() + i, deleters.data() + num_unique, deleter);
            if (num_listed == unique[i]->aligned_memory.use_count() &&
                blocks.add(deleter->allocation, deleter->bytes)) {
                added[num_added++] = deleter;
                // Dropping a copy acquires the count, so earlier owners' accesses happen before the release.
                [[maybe_unused]] const std::shared_ptr<Fr[]> copy = unique[i]->aligned_memory;
            }
        }
        if (blocks.release()) {
            for (size_t i = 0; i < num_added; ++i) {
                added[i]->pages_released.store(true, std::memory_order_relaxed);
            }
        }
    }

  private:
    // Keeps the block's allocation and size, so that a batch can release the block's pages before its last handle is
    // destroyed. An aliasing handle shares this deleter but may point inside the allocation.
    struct AlignedDeleter {
        Fr* allocation = nullptr;
        size_t bytes = 0;
        // Relaxed suffices: it only lets the deleter skip a release a batch already made, and orders nothing else.
        std::atomic<bool> pages_released{ false };

        AlignedDeleter(Fr* allocated, size_t allocated_bytes) noexcept
            : allocation(allocated)
            , bytes(allocated_bytes)
        {}
        // shared_ptr needs a copyable deleter, and std::atomic is not copyable.
        AlignedDeleter(const AlignedDeleter& other) noexcept
            : allocation(other.allocation)
            , bytes(other.bytes)
            , pages_released(other.pages_released.load(std::memory_order_relaxed))
        {}

        void operator()(Fr* p) const
        {
            if (!pages_released.load(std::memory_order_relaxed)) {
                bb::release_pages_in_parallel(p, bytes);
            }
            delete[] p;
        }
    };

    AlignedDeleter* aligned_deleter() const { return std::get_deleter<AlignedDeleter>(aligned_memory); }

    // Use new Fr[] instead of std::make_shared<Fr[]>(n) to avoid serial
    // value-initialization (zeroing). Polynomial's constructor handles
    // zeroing in parallel where needed.
    static void allocate_aligned(BackingMemory& memory, size_t size)
    {
        if (size == 0) {
            memory.aligned_memory = nullptr;
            memory.raw_data = nullptr;
            return;
        }
        Fr* ptr = new Fr[size];
        memory.aligned_memory = std::shared_ptr<Fr[]>(ptr, AlignedDeleter{ ptr, size * sizeof(Fr) });
        memory.raw_data = ptr;
    }

#if !defined(__wasm__) && !defined(_WIN32)
    static bool try_allocate_file_backed(BackingMemory& memory, size_t size)
    {
        if (size == 0) {
            return false;
        }

        if (size > std::numeric_limits<size_t>::max() / sizeof(Fr)) {
            return false;
        }

        size_t required_bytes = size * sizeof(Fr);

        // Check and update storage usage to enforce budget
        size_t current_usage = current_storage_usage.load();
        while (true) {
            if (current_usage + required_bytes > storage_budget) {
                return false;
            }
            if (current_storage_usage.compare_exchange_weak(current_usage, current_usage + required_bytes)) {
                break;
            }
        }

        size_t file_size = required_bytes;
        static std::atomic<size_t> file_counter{ 0 };
        size_t id = file_counter.fetch_add(1);

        std::filesystem::path temp_dir;
        try {
            temp_dir = std::filesystem::temp_directory_path();
        } catch (const std::exception&) {
            temp_dir = std::filesystem::current_path();
        }

        std::string filename = temp_dir / ("poly-mmap-" + std::to_string(getpid()) + "-" + std::to_string(id));

        int fd = open(filename.c_str(), O_CREAT | O_RDWR | O_TRUNC | O_EXCL, 0600);
        if (fd < 0) {
            current_storage_usage.fetch_sub(required_bytes);
            return false;
        }

        if (ftruncate(fd, static_cast<off_t>(file_size)) != 0) {
            close(fd);
            std::filesystem::remove(filename);
            current_storage_usage.fetch_sub(required_bytes);
            return false;
        }

        void* addr = mmap(nullptr, file_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
        if (addr == MAP_FAILED) {
            close(fd);
            std::filesystem::remove(filename);
            current_storage_usage.fetch_sub(required_bytes);
            return false;
        }

        auto file_backed_data = std::make_shared<FileBackedData>();
        file_backed_data->file_size = file_size;
        file_backed_data->filename = filename;
        file_backed_data->fd = fd;
        file_backed_data->raw_data_ptr = static_cast<Fr*>(addr);

        memory.raw_data = static_cast<Fr*>(addr);
        memory.file_backed = std::move(file_backed_data);

        return true;
    }
#endif
};
