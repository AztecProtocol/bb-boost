#include "barretenberg/common/exit_release.hpp"

#include <algorithm>
#include <charconv>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>

#if defined(__linux__) && !defined(__wasm__) && !defined(NO_MULTITHREADING)
#define BB_EXIT_RELEASE_SUPPORTED 1
#endif
// Sanitizers report from exit handlers, which _exit skips.
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#undef BB_EXIT_RELEASE_SUPPORTED
#endif
#if defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) || __has_feature(memory_sanitizer)
#undef BB_EXIT_RELEASE_SUPPORTED
#endif
#endif

#ifdef BB_EXIT_RELEASE_SUPPORTED
#include "barretenberg/common/bb_bench.hpp"
#include "barretenberg/common/log.hpp"
#include "barretenberg/common/thread.hpp"
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <link.h>
#include <new>
#include <pthread.h>
#include <sched.h>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// Coverage runtimes write their data from exit handlers, which _exit skips; with one linked in, exit the normal way.
extern "C" int __llvm_profile_write_file(void) __attribute__((weak));
extern "C" void __gcov_dump(void) __attribute__((weak));
extern ElfW(Dyn) _DYNAMIC[] __attribute__((weak));
#endif

namespace bb {

std::optional<std::vector<MemoryRange>> select_exit_release_ranges(std::string_view maps,
                                                                   std::span<const MemoryRange> excluded,
                                                                   std::span<const uintptr_t> pinned,
                                                                   size_t min_bytes)
{
    std::vector<MemoryRange> out;
    while (!maps.empty()) {
        const size_t eol = maps.find('\n');
        const std::string line(maps.substr(0, eol));
        maps.remove_prefix(eol == std::string_view::npos ? maps.size() : eol + 1);
        if (line.empty()) {
            continue;
        }
        // "begin-end perms offset dev inode [path]"
        uintptr_t begin = 0;
        uintptr_t end = 0;
        char perms[5] = {};
        int path_at = -1;
        if (std::sscanf(line.c_str(), "%" SCNxPTR "-%" SCNxPTR " %4s %*s %*s %*s%n", &begin, &end, perms, &path_at) !=
                3 ||
            path_at < 0 || end <= begin || std::strlen(perms) < 4) {
            return std::nullopt;
        }
        const size_t path_start = line.find_first_not_of(' ', static_cast<size_t>(path_at));
        const std::string_view path =
            path_start == std::string::npos ? std::string_view{} : std::string_view(line).substr(path_start);
        const bool private_writable = perms[1] == 'w' && perms[3] == 'p';
        const bool anonymous = path.empty() || path == "[heap]" || path.starts_with("[anon:");
        if (!private_writable || !anonymous ||
            std::ranges::any_of(pinned, [&](uintptr_t a) { return a >= begin && a < end; })) {
            continue;
        }
        std::vector<MemoryRange> pieces{ { begin, end } };
        for (const MemoryRange& cut : excluded) {
            std::vector<MemoryRange> next;
            for (const MemoryRange& piece : pieces) {
                if (cut.end <= piece.begin || cut.begin >= piece.end) {
                    next.push_back(piece);
                    continue;
                }
                if (cut.begin > piece.begin) {
                    next.push_back({ piece.begin, cut.begin });
                }
                if (cut.end < piece.end) {
                    next.push_back({ cut.end, piece.end });
                }
            }
            pieces = std::move(next);
        }
        for (const MemoryRange& piece : pieces) {
            if (piece.end - piece.begin >= min_bytes) {
                out.push_back(piece);
            }
        }
    }
    return out;
}

#ifndef BB_EXIT_RELEASE_SUPPORTED
ExitReleaseSystem default_exit_release_system()
{
    return { nullptr, nullptr, nullptr, nullptr };
}

bool release_memory_and_exit(int /*code*/, const ExitReleaseSystem& /*system*/)
{
    return false;
}
#else
namespace {
constexpr size_t MIN_MAPPING_BYTES = size_t{ 2 } << 20;
constexpr size_t MIN_TOTAL_BYTES = size_t{ 64 } << 20; // below this the serial exit is a few ms
constexpr size_t BYTES_PER_WORKER = size_t{ 32 } << 20;
constexpr uint32_t MAX_WORKERS = 16;
constexpr size_t WORKER_STACK_BYTES = size_t{ 256 } << 10;
constexpr size_t MAX_RANGES = 8192;
// Covers the calling thread's TCB (stack-protector canary included) next to its static TLS blocks.
constexpr uintptr_t TLS_WINDOW_BYTES = uintptr_t{ 64 } << 10;
// How long the thread count may take to reach what is expected (a joined thread stays counted until the kernel
// releases it) before the normal exit is taken instead.
constexpr auto THREAD_COUNT_WAIT = std::chrono::milliseconds(2);

enum : uint32_t { WAIT = 0, RELEASE = 1, ABORT = 2 };

// Everything the release threads read lives in the scratch mapping, which is never released.
struct ReleaseState {
    std::atomic<uint32_t> ready{ 0 };
    std::atomic<uint32_t> go{ WAIT };
    std::atomic<uint32_t> done{ 0 };
    std::atomic<uint32_t> failed{ 0 }; // madvise calls that returned an error (exit frees those pages)
    uint32_t num_workers = 0;
    uint64_t page_bytes = 0;
    uint64_t total_bytes = 0;
    size_t num_ranges = 0;
    MemoryRange ranges[MAX_RANGES];
};
struct WorkerArg {
    ReleaseState* state;
    uint32_t index;
};
constexpr size_t SCRATCH_BYTES =
    (MAX_WORKERS * WORKER_STACK_BYTES) + sizeof(ReleaseState) + (MAX_WORKERS * sizeof(WorkerArg));

// Releases worker `index`'s equal, page-aligned share of the ranges' bytes.
void release_share(ReleaseState& s, uint32_t index)
{
    const uint64_t pages = s.total_bytes / s.page_bytes;
    const uint64_t lo = (pages * index / s.num_workers) * s.page_bytes;
    const uint64_t hi = (pages * (index + 1) / s.num_workers) * s.page_bytes;
    uint64_t offset = 0;
    for (size_t i = 0; i < s.num_ranges && offset < hi; ++i) {
        const uint64_t bytes = s.ranges[i].end - s.ranges[i].begin;
        const uint64_t from = std::max(lo, offset);
        const uint64_t to = std::min(hi, offset + bytes);
        if (from < to &&
            madvise(reinterpret_cast<void*>(s.ranges[i].begin + (from - offset)), to - from, MADV_DONTNEED) != 0) {
            s.failed.fetch_add(1, std::memory_order_relaxed);
        }
        offset += bytes;
    }
}

void* release_worker(void* arg)
{
    const auto* a = static_cast<const WorkerArg*>(arg);
    ReleaseState& s = *a->state;
    s.ready.fetch_add(1, std::memory_order_release);
    uint32_t go = WAIT;
    while ((go = s.go.load(std::memory_order_acquire)) == WAIT) {
        sched_yield();
    }
    if (go == ABORT) {
        return nullptr; // nothing was released: an ordinary thread exit is safe
    }
    release_share(s, a->index);
    s.done.fetch_add(1, std::memory_order_release);
    // Signals are blocked, so this sleeps until _exit ends the process, touching only this thread's stack.
    for (;;) {
        pause();
    }
}

// st_nlink of /proc/self/task is 2 plus the kernel's thread count (proc_task_getattr), the Threads: of
// /proc/self/status; 0 when it cannot be read. It rises in clone before the new thread runs and drops only when an
// exiting thread is released, after it has left the address space (exit_mm). It counts this thread group only, not a
// process sharing the memory without being a thread (CLONE_VM without CLONE_THREAD) nor I/O in flight.
size_t count_threads_default()
{
    struct stat st{};
    return stat("/proc/self/task", &st) == 0 && st.st_nlink > 2 ? static_cast<size_t>(st.st_nlink - 2) : 0;
}

// Polls until the thread count reads `expected`. A matching count ends the wait whenever it comes; after one that does
// not, the wait gives up once THREAD_COUNT_WAIT has passed, so a slow stat or sleep can carry it past that. Yielding
// lets an exiting thread queued on this CPU run; an occasional short sleep frees the CPU when a yield does not.
bool wait_for_thread_count(const ExitReleaseSystem& system, size_t expected, size_t& counts)
{
    const auto deadline = std::chrono::steady_clock::now() + THREAD_COUNT_WAIT;
    for (counts = 1; system.count_threads() != expected; ++counts) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        if (counts % 8 == 0) {
            const timespec step{ 0, 20'000 };
            nanosleep(&step, nullptr);
        } else {
            sched_yield();
        }
    }
    return true;
}

// The calls made after the release (madvise, sched_yield, pause, write, _exit) go through this object's PLT; a lazy
// slot would enter the dynamic resolver, whose link maps live in anonymous memory the release may have zeroed. _DYNAMIC
// is the dynamic section of the object this code is linked into; a static binary has none and binds nothing lazily.
bool binds_immediately_default()
{
    bool lazy_slots = false;
    for (const ElfW(Dyn)* d = _DYNAMIC; d != nullptr && d->d_tag != DT_NULL; ++d) {
        if (d->d_tag == DT_BIND_NOW || (d->d_tag == DT_FLAGS && (d->d_un.d_val & DF_BIND_NOW) != 0) ||
            (d->d_tag == DT_FLAGS_1 && (d->d_un.d_val & DF_1_NOW) != 0)) {
            return true;
        }
        lazy_slots = lazy_slots || d->d_tag == DT_JMPREL;
    }
    return !lazy_slots;
}

void* map_scratch_default(size_t bytes)
{
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    return p == MAP_FAILED ? nullptr : p;
}

int start_thread_default(void* handle, void* stack, size_t stack_bytes, void* (*fn)(void*), void* arg)
{
    pthread_attr_t attr;
    if (pthread_attr_init(&attr) != 0) {
        return -1;
    }
    int rc = pthread_attr_setstack(&attr, stack, stack_bytes);
    if (rc == 0) {
        rc = pthread_create(static_cast<pthread_t*>(handle), &attr, fn, arg);
    }
    pthread_attr_destroy(&attr);
    return rc;
}

MemoryRange page_align_out(uintptr_t lo, uintptr_t hi, uintptr_t page)
{
    return { lo & ~(page - 1), (hi + page - 1) & ~(page - 1) };
}

// The data and bss of bb, libc and the loader hold the allocator's and the threads library's state.
int add_loaded_object(struct dl_phdr_info* info, size_t /*size*/, void* excluded)
{
    const auto page = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
    for (size_t i = 0; i < info->dlpi_phnum; ++i) {
        if (const auto& ph = info->dlpi_phdr[i]; ph.p_type == PT_LOAD) {
            const uintptr_t lo = info->dlpi_addr + ph.p_vaddr;
            static_cast<std::vector<MemoryRange>*>(excluded)->push_back(page_align_out(lo, lo + ph.p_memsz, page));
        }
    }
    return 0;
}

thread_local char tls_marker;
} // namespace

ExitReleaseSystem default_exit_release_system()
{
    if (__llvm_profile_write_file != nullptr || __gcov_dump != nullptr) {
        return { nullptr, nullptr, nullptr, nullptr };
    }
    return { count_threads_default, binds_immediately_default, map_scratch_default, start_thread_default };
}

namespace {
// Writes what the normal exit would have written, then releases and exits. noexcept: an exception while writing the
// BB_BENCH report ends the process here exactly as it would from ~GlobalBenchStatsContainer.
[[noreturn]] void release_and_exit(int code, ReleaseState& state, uint32_t started) noexcept
{
    detail::GLOBAL_BENCH_STATS.print_exit_report_if_requested();
    std::cout.flush();
    std::clog.flush();
    std::cerr.flush();
    std::fflush(nullptr);

    // From here this thread touches only its stack, its TLS, the scratch mapping and loaded objects' data.
    state.go.store(RELEASE, std::memory_order_release);
    release_share(state, 0);
    while (state.done.load(std::memory_order_acquire) != started) {
        sched_yield();
    }
    if (const uint32_t failed = state.failed.load(); failed != 0 && LogLevel::VERBOSE <= bb_log_level) {
        char note[64] = "exit: madvise failed ";
        char* end = std::to_chars(note + 21, note + 52, failed).ptr;
        std::memcpy(end, " times\n", 7);
        static_cast<void>(write(STDERR_FILENO, note, static_cast<size_t>(end + 7 - note)));
    }
    _exit(code);
}
} // namespace

bool release_memory_and_exit(int code, const ExitReleaseSystem& system)
{
    if (__llvm_profile_write_file != nullptr || __gcov_dump != nullptr) {
        return false;
    }
    // 1. Signals stay blocked until _exit, so no handler runs on this thread (it could start a thread or touch released
    //    memory); the release threads inherit the mask. Every return below restores it.
    sigset_t all;
    sigset_t previous;
    sigfillset(&all);
    if (pthread_sigmask(SIG_BLOCK, &all, &previous) != 0) {
        return false;
    }
    uint8_t* scratch = nullptr;
    size_t scratch_bytes = 0;
    const auto normal_exit = [&](const char* why) noexcept {
        if (scratch != nullptr) {
            munmap(scratch, scratch_bytes);
        }
        const bool restored = pthread_sigmask(SIG_SETMASK, &previous, nullptr) == 0;
        try {
            vinfo("exit: normal teardown (", why, restored ? ")" : "; signal mask not restored)");
        } catch (...) {
        }
        return false;
    };
    ReleaseState* state = nullptr;
    try {
        // 2. Only this thread may run. The kernel's count must first equal this thread plus its parallel_for workers,
        //    which shows that /proc counts threads (an emulated one that does not fails here), then come down to one
        //    once they are stopped and joined. With one thread, whose signals are blocked, no thread can run or start.
        //    bb starts no process sharing its memory and no asynchronous I/O, which the count would not show.
        const size_t workers = parallel_for_num_workers();
        size_t counts = 0;
        if (workers == 0 || !wait_for_thread_count(system, workers + 1, counts)) {
            return normal_exit("thread count does not match the workers");
        }
        parallel_for_stop_workers();
        const auto counting_since = std::chrono::steady_clock::now();
        if (!wait_for_thread_count(system, 1, counts)) {
            return normal_exit("another thread is alive");
        }
        const auto wait = std::chrono::steady_clock::now() - counting_since;
        if (!system.binds_immediately()) {
            return normal_exit("lazy symbol binding");
        }
        // 3. The scratch mapping, a whole number of pages, mapped before /proc/self/maps is read so the selection
        //    excludes it by address.
        const auto page = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
        scratch_bytes = (SCRATCH_BYTES + page - 1) & ~(page - 1);
        scratch = static_cast<uint8_t*>(system.map_scratch(scratch_bytes));
        if (scratch == nullptr) {
            return normal_exit("no scratch mapping");
        }
        // 4. What may not be released: loaded objects, the scratch mapping, and this thread's stack and TLS.
        const auto scratch_lo = reinterpret_cast<uintptr_t>(scratch);
        std::vector<MemoryRange> excluded;
        dl_iterate_phdr(add_loaded_object, &excluded);
        excluded.push_back({ scratch_lo, scratch_lo + scratch_bytes });
        int stack_marker = 0;
        const uintptr_t tls_lo =
            std::min(reinterpret_cast<uintptr_t>(&tls_marker), reinterpret_cast<uintptr_t>(&errno));
        const uintptr_t tls_hi =
            std::max(reinterpret_cast<uintptr_t>(&tls_marker), reinterpret_cast<uintptr_t>(&errno));
        excluded.push_back(page_align_out(tls_lo - TLS_WINDOW_BYTES, tls_hi + TLS_WINDOW_BYTES, page));
        const uintptr_t pinned[] = { reinterpret_cast<uintptr_t>(&stack_marker), tls_lo, tls_hi };
        std::stringstream maps;
        maps << std::ifstream("/proc/self/maps").rdbuf();
        const auto ranges = select_exit_release_ranges(maps.str(), excluded, pinned, MIN_MAPPING_BYTES);
        if (!ranges || ranges->size() > MAX_RANGES) {
            return normal_exit("memory map not usable");
        }
        uint64_t total = 0;
        for (const MemoryRange& r : *ranges) {
            total += r.end - r.begin;
        }
        if (total < MIN_TOTAL_BYTES) {
            return normal_exit("little memory to release");
        }
        state = new (scratch + (MAX_WORKERS * WORKER_STACK_BYTES)) ReleaseState();
        std::ranges::copy(*ranges, state->ranges);
        state->num_ranges = ranges->size();
        state->total_bytes = total;
        state->page_bytes = page;
        state->num_workers = static_cast<uint32_t>(
            std::clamp<uint64_t>(std::min<uint64_t>(total / BYTES_PER_WORKER, get_num_cpus()), 2, MAX_WORKERS));
        vinfo("exit: releasing ",
              total >> 20,
              " MiB in ",
              ranges->size(),
              " ranges on ",
              state->num_workers,
              " threads (one thread after ",
              counts,
              " counts, ",
              std::chrono::duration_cast<std::chrono::microseconds>(wait).count(),
              " us)");
    } catch (...) {
        return normal_exit("exception while preparing");
    }

    // 5. Release threads on stacks inside the scratch mapping.
    auto* args = reinterpret_cast<WorkerArg*>(state + 1);
    pthread_t handles[MAX_WORKERS];
    uint32_t started = 0;
    for (uint32_t i = 1; i < state->num_workers; ++i) {
        args[i] = { state, i };
        if (system.start_thread(&handles[started],
                                scratch + ((i - 1) * WORKER_STACK_BYTES),
                                WORKER_STACK_BYTES,
                                release_worker,
                                &args[i]) != 0) {
            state->go.store(ABORT, std::memory_order_release);
            for (uint32_t t = 0; t < started; ++t) {
                pthread_join(handles[t], nullptr);
            }
            return normal_exit("release thread did not start");
        }
        ++started;
    }
    while (state->ready.load(std::memory_order_acquire) != started) {
        sched_yield();
    }
    release_and_exit(code, *state, started);
}
#endif

} // namespace bb
