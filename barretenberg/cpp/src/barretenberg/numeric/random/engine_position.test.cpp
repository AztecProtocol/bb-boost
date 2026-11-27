#include "barretenberg/common/assert.hpp"
#include "barretenberg/ecc/curves/bn254/g1.hpp"
#include "engine.hpp"
#include "entropy_buffer_observer.hpp"
#include <algorithm>
#include <condition_variable>
#include <gtest/gtest.h>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

using namespace bb;
using numeric::RNG;

namespace {

constexpr std::uint_fast64_t SEED = 0x5eed;

// Draws of every width, recorded as 64-bit words so two runs can be compared.
std::vector<uint64_t> mixed_draws(RNG& engine, size_t rounds)
{
    std::vector<uint64_t> out;
    for (size_t i = 0; i < rounds; i++) {
        out.push_back(engine.get_random_uint8());
        out.push_back(engine.get_random_uint16());
        out.push_back(engine.get_random_uint32());
        out.push_back(engine.get_random_uint64());
        const uint128_t wide = engine.get_random_uint128();
        out.push_back(static_cast<uint64_t>(wide));
        out.push_back(static_cast<uint64_t>(wide >> 64));
        const uint256_t widest = engine.get_random_uint256();
        out.insert(out.end(), widest.data, widest.data + 4);
    }
    return out;
}

std::vector<g1::element> group_draws(RNG& engine, size_t count)
{
    std::vector<g1::element> out;
    for (size_t i = 0; i < count; i++) {
        out.push_back(g1::element::random_element(&engine));
    }
    return out;
}

} // namespace

TEST(EnginePosition, DebugEngineScopeOnAnotherThreadMatchesSerialStream)
{
    constexpr size_t ROUNDS = 200;
    auto& engine = numeric::get_debug_randomness(true, SEED);
    const std::vector<uint64_t> serial = mixed_draws(engine, ROUNDS);
    const uint64_t serial_next = engine.get_random_uint64();

    numeric::get_debug_randomness(true, SEED);
    const RNG::Position start = engine.position();
    ASSERT_FALSE(start.empty());

    std::vector<uint64_t> on_thread;
    std::vector<uint64_t> through_accessor;
    RNG::Position reached;
    std::thread([&] {
        RNG& captured = numeric::get_debug_randomness();
        auto scope = engine.draw_from(start);
        on_thread = mixed_draws(captured, ROUNDS / 2);
        through_accessor = mixed_draws(numeric::get_debug_randomness(), ROUNDS - (ROUNDS / 2));
        reached = scope->reached();
    }).join();
    on_thread.insert(on_thread.end(), through_accessor.begin(), through_accessor.end());

    EXPECT_EQ(on_thread, serial);
    EXPECT_TRUE(engine.at(start));
    EXPECT_FALSE(engine.at(reached));

    std::vector<uint64_t> looked_ahead;
    EXPECT_EQ(engine.after(start, [&](RNG& r) { looked_ahead = mixed_draws(r, ROUNDS); }), reached);
    EXPECT_EQ(looked_ahead, serial);
    EXPECT_TRUE(engine.at(start));

    engine.adopt(reached);
    EXPECT_TRUE(engine.at(reached));
    EXPECT_EQ(engine.get_random_uint64(), serial_next);
}

TEST(EnginePosition, DebugEngineScopeIsPerThread)
{
    constexpr size_t ROUNDS = 50;
    auto& engine = numeric::get_debug_randomness(true, SEED);
    const std::vector<uint64_t> serial = mixed_draws(engine, 2 * ROUNDS);

    numeric::get_debug_randomness(true, SEED);
    const RNG::Position start = engine.position();
    std::mutex mutex;
    std::condition_variable cv;
    int stage = 0;
    std::vector<uint64_t> on_thread;
    std::thread worker([&] {
        auto scope = engine.draw_from(start);
        on_thread = mixed_draws(engine, ROUNDS);
        std::unique_lock lock(mutex);
        stage = 1;
        cv.notify_all();
        cv.wait(lock, [&] { return stage == 2; });
        const auto rest = mixed_draws(engine, ROUNDS);
        on_thread.insert(on_thread.end(), rest.begin(), rest.end());
    });
    std::vector<uint64_t> on_main;
    {
        std::unique_lock lock(mutex);
        cv.wait(lock, [&] { return stage == 1; });
        // The worker's scope stays installed while the main thread draws from the shared stream.
        on_main = mixed_draws(engine, ROUNDS);
        stage = 2;
        cv.notify_all();
    }
    worker.join();

    EXPECT_EQ(on_thread, serial);
    EXPECT_EQ(on_main,
              std::vector<uint64_t>(serial.begin(), serial.begin() + static_cast<std::ptrdiff_t>(on_main.size())));
}

TEST(EnginePosition, DebugEngineGroupElementsLandOnTheSameEndPosition)
{
    constexpr size_t COUNT = 16;
    auto& engine = numeric::get_debug_randomness(true, SEED);
    const RNG::Position start = engine.position();

    std::vector<g1::element> on_thread;
    RNG::Position reached;
    std::thread([&] {
        auto scope = engine.draw_from(start);
        on_thread = group_draws(engine, COUNT);
        reached = scope->reached();
    }).join();

    std::vector<g1::element> looked_ahead;
    EXPECT_EQ(engine.after(start, [&](RNG& r) { looked_ahead = group_draws(r, COUNT); }), reached);
    EXPECT_TRUE(engine.at(start));

    const std::vector<g1::element> serial = group_draws(engine, COUNT);
    EXPECT_TRUE(engine.at(reached));
    for (size_t i = 0; i < COUNT; i++) {
        EXPECT_TRUE(on_thread[i] == serial[i]);
        EXPECT_TRUE(looked_ahead[i] == serial[i]);
    }
}

TEST(EnginePosition, DebugEngineRejectsMisuse)
{
    auto& engine = numeric::get_debug_randomness(true, SEED);
    const RNG::Position empty;
    EXPECT_FALSE(engine.at(empty));
    EXPECT_THROW_WITH_MESSAGE(engine.adopt(empty), "position is empty");
    EXPECT_THROW_WITH_MESSAGE((void)engine.draw_from(empty), "position is empty");

    const RNG::Position start = engine.position();
    auto scope = engine.draw_from(start);
    EXPECT_THROW_WITH_MESSAGE((void)engine.draw_from(start), "already active on this thread");
    EXPECT_THROW_WITH_MESSAGE((void)engine.at(start), "inside a draw scope");
}

TEST(EnginePosition, DefaultEngineHasNoOrder)
{
    auto& engine = numeric::get_randomness();
    if (!engine.position().empty()) {
        GTEST_SKIP() << "get_randomness() returns an ordered engine in this build";
    }
    const RNG::Position start = engine.position();
    EXPECT_TRUE(engine.at(start));
    bool called = false;
    EXPECT_TRUE(engine.after(start, [&](RNG&) { called = true; }).empty());
    EXPECT_FALSE(called);
    {
        auto scope = engine.draw_from(start);
        EXPECT_NE(engine.get_random_uint256(), engine.get_random_uint256());
        EXPECT_TRUE(scope->reached().empty());
    }
    engine.adopt(start);
    EXPECT_TRUE(engine.at(start));
    EXPECT_TRUE(engine.position().empty());
}

namespace {

// Records which thread made each fill and the byte ranges each thread consumed, merging contiguous consumption.
class RecordingObserver final : public numeric::EntropyBufferObserver {
  public:
    struct Consumed {
        std::thread::id consumer;
        size_t begin;
        size_t end;
    };
    struct Fill {
        std::thread::id filler;
        size_t size = 0;
        std::vector<Consumed> consumed;
    };

    void on_fill(uint64_t fill, size_t size) override
    {
        std::lock_guard lock(mutex);
        if (!fills.emplace(fill, Fill{ std::this_thread::get_id(), size, {} }).second) {
            duplicate_fill_ids++;
        }
    }

    void on_consume(uint64_t fill, size_t offset, size_t length) override
    {
        std::lock_guard lock(mutex);
        const auto it = fills.find(fill);
        if (it == fills.end()) {
            unrecorded_fill_consumers.insert(std::this_thread::get_id());
            return;
        }
        auto& consumed = it->second.consumed;
        if (!consumed.empty() && consumed.back().consumer == std::this_thread::get_id() &&
            consumed.back().end == offset) {
            consumed.back().end += length;
        } else {
            consumed.push_back({ std::this_thread::get_id(), offset, offset + length });
        }
    }

    std::mutex mutex;
    std::map<uint64_t, Fill> fills;
    std::set<std::thread::id> unrecorded_fill_consumers;
    size_t duplicate_fill_ids = 0;
};

} // namespace

TEST(EnginePosition, DefaultEngineThreadsConsumeOnlyTheirOwnFills)
{
    auto& engine = numeric::get_randomness();
    if (!engine.position().empty()) {
        GTEST_SKIP() << "get_randomness() returns an ordered engine in this build";
    }
    // A 256-bit draw consumes 32 bytes; about 2.5 MiB per thread makes every thread refill its buffer at least twice.
    constexpr size_t BYTES_PER_DRAW = 32;
    constexpr size_t DRAWS_PER_THREAD = (5UL << 20) / 2 / BYTES_PER_DRAW;
    constexpr size_t ROUNDS = 3;
    constexpr size_t THREADS_PER_ROUND = 2;

    RecordingObserver observer;
    ASSERT_EQ(numeric::set_entropy_buffer_observer(&observer), nullptr);
    std::mutex ids_mutex;
    std::set<std::thread::id> workers;
    for (size_t round = 0; round < ROUNDS; round++) {
        std::vector<std::thread> threads;
        for (size_t t = 0; t < THREADS_PER_ROUND; t++) {
            // Half of the threads draw inside a scope, which must change nothing for this engine.
            const bool scoped = (t % 2) == 0;
            threads.emplace_back([&engine, &ids_mutex, &workers, scoped] {
                {
                    std::lock_guard lock(ids_mutex);
                    workers.insert(std::this_thread::get_id());
                }
                std::unique_ptr<RNG::DrawScope> scope;
                if (scoped) {
                    scope = engine.draw_from(engine.position());
                }
                for (size_t i = 0; i < DRAWS_PER_THREAD; i++) {
                    static_cast<void>(engine.get_random_uint256());
                }
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
    }
    numeric::set_entropy_buffer_observer(nullptr);

    EXPECT_EQ(observer.duplicate_fill_ids, 0U);
    for (const auto& worker : workers) {
        EXPECT_EQ(observer.unrecorded_fill_consumers.count(worker), 0U);
    }
    std::map<std::thread::id, size_t> fills_per_worker;
    size_t consumed_bytes = 0;
    for (auto& [id, fill] : observer.fills) {
        if (workers.contains(fill.filler)) {
            fills_per_worker[fill.filler]++;
        }
        auto ranges = fill.consumed;
        std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) { return a.begin < b.begin; });
        for (size_t i = 0; i < ranges.size(); i++) {
            EXPECT_EQ(ranges[i].consumer, fill.filler) << "fill " << id << " read by a thread that did not fill it";
            EXPECT_LE(ranges[i].end, fill.size);
            if (i + 1 < ranges.size()) {
                EXPECT_LE(ranges[i].end, ranges[i + 1].begin) << "fill " << id << " has overlapping reads";
            }
            if (workers.contains(ranges[i].consumer)) {
                consumed_bytes += ranges[i].end - ranges[i].begin;
            }
        }
    }
    EXPECT_EQ(fills_per_worker.size(), workers.size());
    for (const auto& [worker, count] : fills_per_worker) {
        EXPECT_GE(count, 2U);
    }
    EXPECT_EQ(consumed_bytes, ROUNDS * THREADS_PER_ROUND * DRAWS_PER_THREAD * BYTES_PER_DRAW);
}
