#pragma once

#include "barretenberg/stdlib_circuit_builders/circuit_splice.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <string>
#include <string_view>

// Scratch builds need native threads with a parallel_for pool and core count of their own (wasm shares the pool between
// threads, and OpenMP loops ignore the core count), pthreads to size their stacks, and exceptions.
#if !defined(NO_MULTITHREADING) && !defined(__wasm__) && !defined(_WIN32) && !defined(OMP_MULTITHREADING) &&           \
    defined(__cpp_exceptions) && !defined(BB_NO_EXCEPTIONS)
#define BB_SCRATCH_BUILDS_AVAILABLE 1
#include <pthread.h>
#else
#define BB_SCRATCH_BUILDS_AVAILABLE 0
#endif

/**
 * What every scratch build of a circuit uses: a builder constructed with SpliceRecording, built on a thread of its own
 * and spliced into the real builder by the building thread.
 */
namespace acir_format::scratch_build {

#if BB_SCRATCH_BUILDS_AVAILABLE

// Scratch builds start only with at least this many threads, and use at most all but the building thread.
constexpr size_t MIN_THREADS = 3;

/**
 * @brief Stops a scratch build at its next recording once requested from another thread.
 */
class ScratchCancellation final : public bb::SpliceCancellation {
  public:
    // A cancelled build throws "<owner>: the scratch build was cancelled". `before_check`, when set, runs on the
    // recording thread before each check.
    explicit ScratchCancellation(std::string_view owner, std::function<void()> before_check = {});

    void request() noexcept;
    void check() override;

  private:
    std::string message_;
    std::function<void()> before_check_;
    std::atomic<bool> requested_ = false;
};

/**
 * @brief A thread with a stack of the usual main-thread size, joined at most once.
 */
class ScratchThread {
  public:
    // Runs `body(arg)` on a new thread; false when it could not start.
    [[nodiscard]] bool start(void* (*body)(void*), void* arg);
    // Waits for the thread when it was started and not yet joined.
    void join();
    bool running() const { return running_; }

  private:
    pthread_t thread_{};
    bool running_ = false;
};

std::string refusal_reason(bb::SpliceRefusal refusal);

/**
 * @brief Returns to the system the memory that scratch threads have freed.
 * @details Freed blocks stay resident in the malloc arena of the thread that allocated them, which the building thread
 * does not allocate from. While a scratch builder lives, that memory lies in free blocks below the builder's own, which
 * glibc can return; once the builder is destroyed its arena is a single top block, which glibc trims only in the main
 * arena.
 */
void return_freed_memory();

// What `error` says, or a fixed text when it holds something other than a std::exception.
std::string message_of(const std::exception_ptr& error);

int64_t milliseconds_between(std::chrono::steady_clock::time_point from, std::chrono::steady_clock::time_point to);

#endif

} // namespace acir_format::scratch_build
