#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <optional>
#include <stdexcept>

#if defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(hwaddress_sanitizer) || __has_feature(memory_sanitizer) ||       \
    __has_feature(thread_sanitizer)
#define BB_ADDRESS_SPACE_SANITIZED 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_HWADDRESS__) || defined(__SANITIZE_THREAD__)
#define BB_ADDRESS_SPACE_SANITIZED 1
#endif

// Reservation needs 64-bit Linux with glibc, exceptions, and none of the address, hwaddress, memory and thread
// sanitizers, whose allocators can abort on a failed allocation instead of throwing.
#if defined(__linux__) && defined(__GLIBC__) && !defined(__wasm__) && UINTPTR_MAX == UINT64_MAX &&                     \
    defined(__cpp_exceptions) && !defined(BB_NO_EXCEPTIONS) && !defined(BB_ADDRESS_SPACE_SANITIZED)
#define BB_ADDRESS_SPACE_RESERVE 1
#include <fcntl.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace bb {

struct AddressSpaceFacts {
    std::optional<int> overcommit_mode; // /proc/sys/vm/overcommit_memory; nullopt when unreadable
    bool address_space_unlimited = false;
    bool allocator_env_set = false; // MALLOC_PERTURB_ or GLIBC_TUNABLES is in the environment
};

/**
 * @brief Heuristic (0) or always (1) overcommit, no RLIMIT_AS, and no allocator settings in the environment. Otherwise
 * a reservation is charged against the budget of every later real allocation, or the environment configures the
 * allocator (perturbation writes every block it hands out). None of these facts is read from the input.
 */
constexpr bool address_space_reservation_policy(const AddressSpaceFacts& facts) noexcept
{
    return facts.overcommit_mode.has_value() && (*facts.overcommit_mode == 0 || *facts.overcommit_mode == 1) &&
           facts.address_space_unlimited && !facts.allocator_env_set;
}

#ifdef BB_ADDRESS_SPACE_RESERVE
namespace address_space_detail {

inline std::optional<int> read_overcommit_mode() noexcept
{
    const int fd = ::open("/proc/sys/vm/overcommit_memory", O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return std::nullopt;
    }
    char text[8] = {};
    const ssize_t length = ::read(fd, text, sizeof(text));
    ::close(fd);
    if (length < 1 || text[0] < '0' || text[0] > '9' || (length > 1 && text[1] != '\n')) {
        return std::nullopt;
    }
    return text[0] - '0';
}

inline bool address_space_unlimited() noexcept
{
    rlimit limit{};
    return ::getrlimit(RLIMIT_AS, &limit) == 0 && limit.rlim_cur == RLIM_INFINITY;
}

} // namespace address_space_detail
#endif

/**
 * @brief address_space_reservation_policy over this process's facts, read once, at the first call: later changes to
 * RLIMIT_AS, the overcommit sysctl or the environment are not seen. Always false where reservation is compiled out.
 */
inline bool address_space_reservation_allowed() noexcept
{
#ifdef BB_ADDRESS_SPACE_RESERVE
    static const bool allowed = address_space_reservation_policy(
        { .overcommit_mode = address_space_detail::read_overcommit_mode(),
          .address_space_unlimited = address_space_detail::address_space_unlimited(),
          .allocator_env_set = std::getenv("MALLOC_PERTURB_") != nullptr || std::getenv("GLIBC_TUNABLES") != nullptr });
    return allowed;
#else
    return false;
#endif
}

/**
 * @brief Reserves `capacity` elements in each of `vecs`, in order, all or nothing; false, touching nothing, when any
 * of them already has capacity.
 * @details On std::bad_alloc or std::length_error every vector is left empty with zero capacity and false returned:
 * neither exception escapes, and nothing reserved stays allocated. Where reservation is compiled out, returns false.
 */
template <typename... Vectors>
bool try_reserve_address_space([[maybe_unused]] size_t capacity, [[maybe_unused]] Vectors&... vecs)
{
#ifdef BB_ADDRESS_SPACE_RESERVE
    if (((vecs.capacity() != 0) || ...)) {
        return false;
    }
    try {
        (vecs.reserve(capacity), ...);
        return true;
    } catch (const std::bad_alloc&) {    // NOLINT(bugprone-empty-catch): released below
    } catch (const std::length_error&) { // NOLINT(bugprone-empty-catch): released below
    }
    (Vectors(vecs.get_allocator()).swap(vecs), ...);
    return false;
#else
    return false;
#endif
}

} // namespace bb
