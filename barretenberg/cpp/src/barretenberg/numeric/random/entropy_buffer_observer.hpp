#pragma once
#include <cstddef>
#include <cstdint>

namespace bb::numeric {

/**
 * @brief Sees each refill of the default engine's per-thread entropy buffer and each byte range a draw consumes from
 * it, for tests.
 * @details Every callback runs on the thread that made the read or the draw, concurrently with other threads, so an
 * implementation must be thread-safe.
 */
class EntropyBufferObserver {
  public:
    EntropyBufferObserver() = default;
    virtual ~EntropyBufferObserver() = default;
    EntropyBufferObserver(const EntropyBufferObserver&) = delete;
    EntropyBufferObserver(EntropyBufferObserver&&) = delete;
    EntropyBufferObserver& operator=(const EntropyBufferObserver&) = delete;
    EntropyBufferObserver& operator=(EntropyBufferObserver&&) = delete;

    /**
     * @brief The calling thread refilled its buffer from the operating-system source.
     * @param fill Identifies this fill of this buffer; unique in the process and never 0.
     * @param size Number of bytes the fill holds.
     */
    virtual void on_fill(uint64_t fill, size_t size) = 0;

    /**
     * @brief The calling thread consumed bytes [offset, offset + length) of `fill`.
     * @details `fill` is 0 when the buffer was filled while no observer was installed.
     */
    virtual void on_consume(uint64_t fill, size_t offset, size_t length) = 0;
};

/**
 * @brief Install `observer` for the default engine (nullptr removes it) and return the observer it replaces.
 * @details The observer must outlive every draw made while it is installed. With none installed, a draw pays one
 * predictable branch.
 */
EntropyBufferObserver* set_entropy_buffer_observer(EntropyBufferObserver* observer);

} // namespace bb::numeric
