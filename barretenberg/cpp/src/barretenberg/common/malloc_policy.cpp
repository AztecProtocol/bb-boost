#include "barretenberg/common/malloc_policy.hpp"
#include "barretenberg/common/log.hpp"

#include <cstdlib>

#if defined(__linux__) && defined(__GLIBC__) && !defined(__wasm__)
#define BB_MALLOC_POLICY_SUPPORTED 1
#endif
// Sanitizer runtimes replace malloc, so glibc's settings would not reach the allocator in use.
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__) || defined(__SANITIZE_HWADDRESS__)
#undef BB_MALLOC_POLICY_SUPPORTED
#endif
#if defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) || __has_feature(memory_sanitizer) ||          \
    __has_feature(hwaddress_sanitizer)
#undef BB_MALLOC_POLICY_SUPPORTED
#endif
#endif

#ifdef BB_MALLOC_POLICY_SUPPORTED
#include <malloc.h>
#include <unistd.h>
#endif

namespace bb {

bool environment_configures_malloc(const char* const* environment)
{
    constexpr std::string_view tunables_prefix = "GLIBC_TUNABLES=";
    for (; environment != nullptr && *environment != nullptr; ++environment) {
        const std::string_view entry(*environment);
        if (entry.starts_with("MALLOC_")) {
            return true;
        }
        if (!entry.starts_with(tunables_prefix)) {
            continue;
        }
        // GLIBC_TUNABLES holds name=value pairs separated by ':'.
        std::string_view rest = entry.substr(tunables_prefix.size());
        while (!rest.empty()) {
            const size_t end = rest.find(':');
            if (rest.substr(0, end).starts_with("glibc.malloc.")) {
                return true;
            }
            if (end == std::string_view::npos) {
                break;
            }
            rest.remove_prefix(end + 1);
        }
    }
    return false;
}

bool process_malloc_policy_supported()
{
#ifdef BB_MALLOC_POLICY_SUPPORTED
    return true;
#else
    return false;
#endif
}

bool command_keeps_heap(std::string_view command, std::string_view scheme, bool slow_low_memory)
{
    return !slow_low_memory && (command == "avm_prove" || (command == "prove" && scheme == "ultra_honk"));
}

MallocPolicy configure_process_malloc(bool keeps_heap, const char* const* environment, MallocParameterSetter set)
{
    if (!keeps_heap) {
        return MallocPolicy::NOT_THIS_COMMAND;
    }
#ifdef BB_MALLOC_POLICY_SUPPORTED
    if (environment_configures_malloc(environment)) {
        return MallocPolicy::LEFT_TO_ENVIRONMENT;
    }
    // Each setting also turns off glibc's dynamic thresholds (mp_.no_dyn_threshold), which otherwise raise the mmap
    // threshold to each freed mapped block's size, up to 32 MiB, and the trim threshold to twice that. Both are set:
    // either alone freezes the other at its starting 128 KiB. The mmap threshold goes first, so that a refusal leaves
    // both untouched; glibc accepts any trim value.
    if (set(M_MMAP_THRESHOLD, static_cast<int>(PROCESS_MMAP_THRESHOLD)) != 1) {
        return MallocPolicy::REFUSED;
    }
    // mallopt's int -1 becomes (unsigned long)-1 inside glibc, the value its malloc.c documents as disabling
    // trimming. An explicit malloc_trim() still trims.
    if (set(M_TRIM_THRESHOLD, PROCESS_TRIM_THRESHOLD) != 1) {
        return MallocPolicy::REFUSED;
    }
    return MallocPolicy::APPLIED;
#else
    static_cast<void>(environment);
    static_cast<void>(set);
    return MallocPolicy::UNSUPPORTED;
#endif
}

MallocPolicy configure_process_malloc_and_log(bool keeps_heap,
                                              const char* const* environment,
                                              MallocParameterSetter set)
{
    // Outside vinfo, whose arguments are evaluated only when verbose logging is on.
    const MallocPolicy policy = configure_process_malloc(keeps_heap, environment, set);
    vinfo(malloc_policy_message(policy));
    return policy;
}

MallocPolicy configure_process_malloc_and_log(bool keeps_heap)
{
#ifdef BB_MALLOC_POLICY_SUPPORTED
    return configure_process_malloc_and_log(keeps_heap, environ, &mallopt);
#else
    return configure_process_malloc_and_log(keeps_heap, nullptr, nullptr);
#endif
}

std::string_view malloc_policy_message(MallocPolicy policy)
{
    switch (policy) {
    case MallocPolicy::APPLIED:
        return "malloc policy: applied (no heap trim, mmap threshold 32 MiB)";
    case MallocPolicy::NOT_THIS_COMMAND:
        return "malloc policy: unchanged (not applied to this command or with --slow_low_memory)";
    case MallocPolicy::LEFT_TO_ENVIRONMENT:
        return "malloc policy: left to the environment (GLIBC_TUNABLES glibc.malloc.* or MALLOC_*)";
    case MallocPolicy::UNSUPPORTED:
        return "malloc policy: not built in";
    case MallocPolicy::REFUSED:
        return "malloc policy: refused by glibc";
    }
    return "malloc policy: unknown";
}

} // namespace bb
