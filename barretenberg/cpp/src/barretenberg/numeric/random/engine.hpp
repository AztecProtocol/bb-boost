// === AUDIT STATUS ===
// internal:    { status: Complete, auditors: [Luke], commit: }
// external_1:  { status: not started, auditors: [], commit: }
// external_2:  { status: not started, auditors: [], commit: }
// =====================

#pragma once
#include "../uint128/uint128.hpp"
#include "../uint256/uint256.hpp"
#include "../uintx/uintx.hpp"
#include "unistd.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <random>

namespace bb::numeric {

class RNG {
  public:
    /**
     * @brief Where an engine's stream of draws stands.
     * @details Empty for an engine whose draws have no order to name. A non-empty position is meaningful only to the
     * engine object that issued it.
     */
    struct Position {
        const RNG* issuer = nullptr;
        std::optional<std::mt19937_64> state;

        [[nodiscard]] bool empty() const { return !state.has_value(); }
        bool operator==(const Position& other) const = default;
    };

    /**
     * @brief The calling thread's substitute stream, installed by draw_from() for as long as this object lives.
     * @details Must be destroyed on the thread that created it.
     */
    class DrawScope {
      public:
        DrawScope() = default;
        virtual ~DrawScope() = default;
        DrawScope(const DrawScope&) = delete;
        DrawScope(DrawScope&&) = delete;
        DrawScope& operator=(const DrawScope&) = delete;
        DrawScope& operator=(DrawScope&&) = delete;

        /**
         * @brief Where the substitute stream stands now; empty when the engine installed none.
         */
        [[nodiscard]] virtual Position reached() const { return {}; }
    };

    /**
     * @brief Where the stream stands now; empty for an engine without an order.
     */
    [[nodiscard]] virtual Position position() const { return {}; }

    /**
     * @brief Where the stream would stand after `draws` ran from `start`, leaving this engine where it is.
     * @details `draws` may draw through its argument or through any reference to this engine; the values are
     * discarded. An engine without an order returns an empty position without calling `draws`.
     */
    [[nodiscard]] virtual Position after([[maybe_unused]] const Position& start,
                                         [[maybe_unused]] const std::function<void(RNG&)>& draws)
    {
        return {};
    }

    /**
     * @brief Make every draw the calling thread makes through this engine read a private stream started at `start`,
     * for as long as the returned scope lives.
     * @details References to this engine taken earlier are covered, and the engine's own stream is never read in the
     * meantime. One scope per thread at a time. An engine without an order installs nothing: the thread draws as it
     * always does.
     */
    [[nodiscard]] virtual std::unique_ptr<DrawScope> draw_from([[maybe_unused]] const Position& start)
    {
        return std::make_unique<DrawScope>();
    }

    /**
     * @brief Whether the stream stands exactly at `where`; always true for an engine without an order.
     */
    [[nodiscard]] virtual bool at([[maybe_unused]] const Position& where) const { return true; }

    /**
     * @brief Move the stream to `where`; does nothing for an engine without an order.
     */
    virtual void adopt([[maybe_unused]] const Position& where) {}

    virtual uint8_t get_random_uint8() = 0;

    virtual uint16_t get_random_uint16() = 0;

    virtual uint32_t get_random_uint32() = 0;

    virtual uint64_t get_random_uint64() = 0;

    virtual uint128_t get_random_uint128() = 0;

    virtual uint256_t get_random_uint256() = 0;

    virtual ~RNG() = default;
    RNG() noexcept = default;
    RNG(const RNG& other) = default;
    RNG(RNG&& other) = default;
    RNG& operator=(const RNG& other) = default;
    RNG& operator=(RNG&& other) = default;

    uint512_t get_random_uint512()
    {
        // Do not inline in constructor call. Evaluation order is important for cross-compiler consistency.
        auto lo = get_random_uint256();
        auto hi = get_random_uint256();
        return { lo, hi };
    }

    uint1024_t get_random_uint1024()
    {
        // Do not inline in constructor call. Evaluation order is important for cross-compiler consistency.
        auto lo = get_random_uint512();
        auto hi = get_random_uint512();
        return { lo, hi };
    }
};

RNG& get_debug_randomness(bool reset = false, std::uint_fast64_t seed = 12345);
RNG& get_randomness();

} // namespace bb::numeric
