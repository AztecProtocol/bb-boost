#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace bb {

struct MemoryRange {
    uintptr_t begin;
    uintptr_t end;
    bool operator==(const MemoryRange&) const = default;
};

/**
 * @brief The address ranges to release before exiting, chosen from the text of /proc/self/maps.
 *
 * @details Kept: private, writable mappings with no backing file (anonymous memory and the brk heap). A mapping that
 * contains any address in `pinned` is dropped whole; the parts of a kept mapping that overlap `excluded` are cut out.
 * Pieces under `min_bytes` are dropped. Pure: it reads only its arguments. Returns nullopt on a line it cannot parse.
 */
std::optional<std::vector<MemoryRange>> select_exit_release_ranges(std::string_view maps,
                                                                   std::span<const MemoryRange> excluded,
                                                                   std::span<const uintptr_t> pinned,
                                                                   size_t min_bytes);

/** The system calls the exit path depends on, replaceable so tests can make each one fail. */
struct ExitReleaseSystem {
    size_t (*count_threads)();          // the kernel's thread count for this thread group; 0 when unreadable
    bool (*binds_immediately)();        // the PLT of the object holding this code was bound at load time
    void* (*map_scratch)(size_t bytes); // nullptr on failure
    int (*start_thread)(void* thread_handle, void* stack, size_t stack_bytes, void* (*fn)(void*), void* arg);
};
ExitReleaseSystem default_exit_release_system();

/**
 * @brief Ends the process with `code`, first handing its private memory back to the kernel from several threads.
 *
 * @details At exit the kernel frees every resident page on one core; madvise(MADV_DONTNEED) on disjoint ranges runs
 * concurrently and leaves exit mostly empty mappings. The released memory includes the allocator's, so once the release
 * starts nothing may run but the release and _exit. Before releasing anything it therefore blocks signals; requires
 * the kernel's thread count to equal this thread plus its parallel_for workers, and then, once it has stopped them,
 * one (each within about 2 ms); requires immediate symbol binding; and writes what a normal exit would write (the
 * BB_BENCH report, buffered streams). The count covers this thread group only: the caller must not share its memory
 * with another process (CLONE_VM) or have asynchronous I/O in flight. Release threads
 * run on stacks in a private scratch mapping; this thread's stack and TLS and every loaded object's data are never
 * released.
 *
 * Returns false, having released nothing, restored the signal mask and written at most a verbose log line, when any of
 * that cannot be established or preparing it throws; the caller then exits the normal way. Never returns otherwise.
 */
bool release_memory_and_exit(int code, const ExitReleaseSystem& system = default_exit_release_system());

} // namespace bb
