#pragma once
#if !defined(__wasm__) && !defined(_WIN32)
#include "barretenberg/common/thread.hpp"
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <filesystem>
#include <unistd.h>

namespace bb::srs {

/**
 * @brief pread exactly `size` bytes at `offset`, retrying short reads and EINTR. Returns false on error or EOF.
 */
inline bool pread_exact(int fd, uint8_t* dst, size_t size, size_t offset)
{
    while (size > 0) {
        const ssize_t got = ::pread(fd, dst, size, static_cast<off_t>(offset));
        if (got < 0 && errno == EINTR) {
            continue;
        }
        if (got <= 0) {
            return false;
        }
        dst += got;
        size -= static_cast<size_t>(got);
        offset += static_cast<size_t>(got);
    }
    return true;
}

/**
 * @brief Reads the first `size` bytes of a file into `dst`, which may be uninitialised, one contiguous slice per
 * thread, so the reads proceed in parallel and each page is first touched by the thread that fills it.
 * @return false if the file cannot be opened or holds fewer than `size` bytes.
 */
inline bool pread_prefix_in_parallel(const std::filesystem::path& path, uint8_t* dst, size_t size)
{
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return false;
    }
    std::atomic<bool> failed{ false };
    parallel_for([&](ThreadChunk chunk) {
        const auto range = chunk.range(size);
        if (!range.empty() && !pread_exact(fd, dst + *range.begin(), range.size(), *range.begin())) {
            failed.store(true, std::memory_order_relaxed);
        }
    });
    ::close(fd);
    return !failed.load();
}

} // namespace bb::srs
#endif
