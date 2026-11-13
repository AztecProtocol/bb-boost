#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace bb {

/** The mmap threshold configure_process_malloc() pins: glibc's ceiling for its dynamic threshold on 64-bit targets. */
inline constexpr size_t PROCESS_MMAP_THRESHOLD = size_t{ 32 } << 20;

/** The trim threshold configure_process_malloc() sets: mallopt's -1, which glibc takes as no trimming at all. */
inline constexpr int PROCESS_TRIM_THRESHOLD = -1;

enum class MallocPolicy : uint8_t {
    APPLIED,             ///< heap trimming is off and the mmap threshold is pinned
    NOT_THIS_COMMAND,    ///< outside command_keeps_heap(): nothing was set, whatever is in force stays
    LEFT_TO_ENVIRONMENT, ///< the environment configures malloc, so nothing was set
    UNSUPPORTED,         ///< not a native glibc build, or a sanitizer build: nothing was set
    REFUSED,             ///< glibc rejected a setting
};

/** Sets one allocator parameter and returns 1 on success, as glibc's mallopt. */
using MallocParameterSetter = int (*)(int parameter, int value);

/**
 * @brief Whether an environment configures glibc's malloc: a GLIBC_TUNABLES entry named glibc.malloc.*, or any
 * variable whose name starts with MALLOC_ (glibc's aliases such as MALLOC_ARENA_MAX and MALLOC_TRIM_THRESHOLD_).
 * Other tunables, such as glibc.pthread.rseq, do not count.
 * @param environment A null-terminated array of NAME=value strings, as environ; null counts as empty.
 */
bool environment_configures_malloc(const char* const* environment);

/** Whether this build can apply the settings: native Linux on glibc, without a sanitizer. */
bool process_malloc_policy_supported();

/**
 * @brief Whether a bb command keeps freed heap memory: `avm_prove`, and `prove` with scheme ultra_honk, unless the
 * user asked for --slow_low_memory.
 * @details An empirical policy per prover family, read from the command and the scheme only, never from an input.
 * For AVM and UltraHonk proving, kept heap measured faster at a peak cost of a few MB to a few percent. For Chonk
 * (`prove --scheme chonk`), which proves many circuits in turn, it measured faster at a much larger peak cost, so
 * Chonk is left out. msgpack run serves many requests from one process, so it is left out too.
 * @param command The top-level subcommand as typed (`msgpack` for msgpack run); empty when there is none.
 * @param scheme The parsed --scheme value.
 * @param slow_low_memory Whether --slow_low_memory was given: a setting that trades memory for time never applies.
 */
bool command_keeps_heap(std::string_view command, std::string_view scheme, bool slow_low_memory);

/**
 * @brief When @p keeps_heap (see command_keeps_heap()), sets M_MMAP_THRESHOLD to PROCESS_MMAP_THRESHOLD and then
 * M_TRIM_THRESHOLD to PROCESS_TRIM_THRESHOLD through @p set. Sets nothing otherwise, nor when @p environment
 * configures malloc.
 * @details Freed heap memory then stays with the process until it exits, and serves later allocations of any size
 * that fit in it. A block of PROCESS_MMAP_THRESHOLD bytes or more gets its own mapping, unmapped by free(), only
 * when no free heap space holds it. A refused mmap threshold stops before the trim setting. glibc's mallopt is not
 * thread-safe: call this before the command starts its workers.
 */
MallocPolicy configure_process_malloc(bool keeps_heap, const char* const* environment, MallocParameterSetter set);

/**
 * @brief configure_process_malloc(), then its outcome on the verbose log. The settings are applied at every log
 * level; only the line depends on it.
 */
MallocPolicy configure_process_malloc_and_log(bool keeps_heap,
                                              const char* const* environment,
                                              MallocParameterSetter set);

/** configure_process_malloc_and_log() on this process's environment, through glibc's mallopt. */
MallocPolicy configure_process_malloc_and_log(bool keeps_heap);

/** One line for the verbose log: "malloc policy: " and the outcome. */
std::string_view malloc_policy_message(MallocPolicy policy);

} // namespace bb
