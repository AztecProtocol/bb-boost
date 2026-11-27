#pragma once
#include <barretenberg/ecc/scalar_multiplication/pippenger_constantine.hpp>
#include <algorithm>
#include <array>
#include <cstdint>

namespace bb::test {
namespace window_skip_test {
namespace rec = bb::scalar_multiplication::round_parallel_detail;
using Scalar = std::array<uint64_t, 4>;
inline std::array<uint32_t, 8> counters{};
inline uint16_t bit_length(const Scalar& s)
{
    for (size_t b = 254; b != 0; --b) {
        if (((s[(b - 1) / 64] >> ((b - 1) % 64)) & 1) != 0) { return static_cast<uint16_t>(b); }
    }
    return 0;
}
inline uint32_t reference(const Scalar& s, size_t base, size_t width)
{
    uint32_t raw = 0;
    for (size_t k = 0; k <= width; ++k) {
        const int64_t bit = static_cast<int64_t>(base + k) - 1;
        if (bit >= 0 && bit < 256) {
            raw |= static_cast<uint32_t>((s[static_cast<size_t>(bit) / 64] >> (bit % 64)) & 1) << k;
        }
    }
    const uint32_t sign = (raw >> width) & 1;
    const uint32_t mask = (uint32_t(1) << width) - 1;
    return (sign << 31) | (((((raw + 1) >> 1) - sign) ^ (0U - sign)) & mask);
}
inline uint32_t scalar_digit(const Scalar& s, size_t base, size_t width)
{
    const auto p = rec::compute_constantine_slice_params(base, width, 4);
    return rec::get_constantine_packed_digit(s.data(), p.lo_limb, p.hi_limb, p.lo_off, p.lo_bits,
                                            p.lo_mask, p.hi_mask, p.slice_localised_to_one_u64, width);
}
inline void quad_digits(const Scalar* s, size_t base, size_t width, uint32_t* out)
{
    const auto p = rec::compute_constantine_slice_params_u32(base, width, 8);
    const auto w = static_cast<uint32_t>(width);
    const rec::SimdU32x4 lo{ p.lo_mask, p.lo_mask, p.lo_mask, p.lo_mask };
    const rec::SimdU32x4 hi{ p.hi_mask, p.hi_mask, p.hi_mask, p.hi_mask };
    const rec::SimdU32x4 one{ 1, 1, 1, 1 };
    const uint32_t m = (uint32_t(1) << width) - 1;
    const rec::SimdU32x4 mask{ m, m, m, m };
    const auto* a = reinterpret_cast<const uint32_t*>(s[0].data());
    const auto* b = reinterpret_cast<const uint32_t*>(s[1].data());
    const auto* c = reinterpret_cast<const uint32_t*>(s[2].data());
    const auto* d = reinterpret_cast<const uint32_t*>(s[3].data());
    switch (rec::classify_slice_path_u32(p)) {
    case rec::ConstantineSlicePath::Localised:
        rec::store_constantine_packed_digits_x4_localised(out,a,b,c,d,p.lo_limb,p.lo_off,lo,one,mask,w); break;
    case rec::ConstantineSlicePath::Bottom:
        rec::store_constantine_packed_digits_x4_bottom(out,a,b,c,d,p.hi_limb,p.lo_bits,hi,one,mask,w); break;
    case rec::ConstantineSlicePath::Boundary:
        rec::store_constantine_packed_digits_x4_boundary(out,a,b,c,d,p.lo_limb,p.hi_limb,p.lo_off,p.lo_bits,lo,hi,one,mask,w); break;
    }
}
inline bool skip(uint16_t bits, size_t base)
{
#ifdef MSM_SKIP_TEST_BASELINE
    (void)bits; (void)base;
    return false;
#else
    return rec::window_above_scalar_bits(bits, base);
#endif
}
} // namespace window_skip_test

template <bool EnableSkip> uint32_t check_msm_window_skip()
{
    using namespace window_skip_test;
    counters.fill(0);
    // Exact Booth lookback equality: highest bit base-1 must NOT be skipped.
    for (size_t bits = 1; bits <= 254; ++bits) {
        Scalar s{}; s[(bits - 1) / 64] = uint64_t(1) << ((bits - 1) % 64);
        for (size_t width = 1; width <= 19; ++width) {
            for (const size_t base : { bits - 1, bits, std::min<size_t>(255, bits + 1) }) {
                const uint32_t want = reference(s, base, width);
                const uint32_t got = scalar_digit(s, base, width);
                if (got != want) { return 1; }
                if constexpr (EnableSkip) {
                    if (skip(static_cast<uint16_t>(bits), base) && (want & 0x7fffffffU) != 0) { return 2; }
#ifndef MSM_SKIP_TEST_BASELINE
                    if (base == bits && skip(static_cast<uint16_t>(bits), base)) { return 3; }
#endif
                }
                ++counters[0];
            }
        }
    }
    std::array<Scalar, 2112> scalars{};
    std::array<uint8_t, 2112> msbs{};
    std::array<uint8_t, 2112> active{};
    std::array<uint32_t, 2112> out_base{};
    uint32_t hash = 2166136261U;
    for (const size_t origin : { size_t(0), size_t(17) }) {
    for (const size_t length : {size_t(0),size_t(1),size_t(63),size_t(64),size_t(65),size_t(127),size_t(129),size_t(2047),size_t(2048),size_t(2049)}) {
        const size_t n = origin + length;
        for (unsigned pattern = 0; pattern < 7; ++pattern) {
            for (unsigned membership = 0; membership < 3; ++membership) {
                for (size_t i = origin; i < n; ++i) {
                    Scalar s{};
                    if (pattern != 0) { s[0] = 1 + i * 17; }
                    if (pattern == 2 || (pattern == 1 && i - origin < 4)) {
                        s[1] = 0x9e3779b97f4a7c15ULL ^ i; s[2] = 0x123456789abcdefULL; s[3] = uint64_t(1) << 61;
                    }
                    if (pattern == 3) { s = {}; s[(i % 254) / 64] = uint64_t(1) << (i % 254); }
                    if (pattern == 4) { s[1] = (uint64_t(1) << 63) | i; } // GLV-width components.
                    if (pattern == 5 && i % 2 == 0) { s = {}; }
                    if (pattern == 6 && i % 64 == 0) { s[3] = uint64_t(1) << 61; }
                    scalars[i] = s;
                    const uint16_t bits = bit_length(s);
                    msbs[i] = bits == 0 ? uint8_t(255) : static_cast<uint8_t>(bits - 1);
                    const bool excluded = membership == 1 ? i % 2 == 0 : membership == 2 && i % 64 == 0;
                    active[i] = bits != 0 && !excluded;
                    out_base[i] = membership == 2 && i % 7 == 0 ? (0x40000000U | static_cast<uint32_t>(i)) : static_cast<uint32_t>(i);
                }
                for (const size_t stride : {size_t(1),size_t(5),size_t(13),size_t(19)}) {
                    for (size_t base = 0; base < 254; base += stride) {
                        const size_t width = std::min(stride, size_t(254) - base);
                        for (size_t tile = origin; tile < n; tile += 2048) {
                            const size_t end = std::min(n, tile + 2048);
                            std::array<uint64_t, 32> masks{};
                            std::array<uint16_t, 32> bits{};
                            for (size_t i = tile; i < end; ++i) {
                                masks[(i-tile)/64] |= uint64_t(active[i]) << ((i-tile)%64);
                                if (active[i]) { bits[(i-tile)/64] = std::max(bits[(i-tile)/64], uint16_t(msbs[i]+1)); }
                            }
                            size_t i = tile;
                            while (i + 64 <= end) {
                                const size_t block = (i-tile)/64;
                                uint64_t fresh = 0;
                                for (size_t k = 0; k < 64; ++k) { fresh |= uint64_t(active[i+k]) << k; }
                                if (fresh != masks[block]) { return 4; }
                                const bool omitted = masks[block] == 0 || (EnableSkip && skip(bits[block], base));
                                std::array<uint32_t, 64> packed{};
                                if (!omitted) {
                                    for (size_t k = 0; k < 64; k += 4) { quad_digits(&scalars[i+k], base, width, packed.data()+k); }
                                    ++counters[2];
                                } else { ++counters[3]; }
                                for (size_t k = 0; k < 64; ++k) {
                                    if (!active[i+k]) { continue; }
                                    const uint32_t want = reference(scalars[i+k], base, width);
                                    if (omitted ? (want & 0x7fffffffU) != 0 : packed[k] != want) { return 5; }
                                    if ((want & 0x7fffffffU) != 0) {
                                        hash = (hash ^ want) * 16777619U;
                                        hash = (hash ^ out_base[i+k]) * 16777619U;
                                        ++counters[4];
                                    }
                                }
                                i += 64;
                            }
                            for (; i < end; ++i) {
                                if (!active[i]) { continue; }
                                const uint32_t want = reference(scalars[i], base, width);
                                const bool omitted = EnableSkip && skip(uint16_t(msbs[i]+1), base);
                                if (omitted ? (want & 0x7fffffffU) != 0 : scalar_digit(scalars[i], base, width) != want) { return 6; }
                                if ((want & 0x7fffffffU) != 0) {
                                    hash = (hash ^ want) * 16777619U;
                                    hash = (hash ^ out_base[i]) * 16777619U;
                                    ++counters[4];
                                }
                                ++counters[5];
                            }
                        }
                        ++counters[1];
                    }
                }
            }
        }
    }
    }
    counters[6] = hash;
    counters[7] = 1;
    return 0;
}
} // namespace bb::test
