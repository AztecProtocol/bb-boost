#include "barretenberg/common/exit_release.hpp"
#include "barretenberg/common/thread.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <new>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace bb;

namespace {
constexpr uintptr_t MiB = uintptr_t{ 1 } << 20;

constexpr const char* MAPS = "00400000-00600000 r-xp 00000000 08:01 123          /usr/bin/bb\n"
                             "00600000-00a00000 rw-p 00200000 08:01 123          /usr/bin/bb\n"
                             "00a00000-04a00000 rw-p 00000000 00:00 0            [heap]\n"
                             "7f0000000000-7f0004000000 rw-p 00000000 00:00 0 \n"
                             "7f0004000000-7f0004100000 rw-p 00000000 00:00 0 \n"
                             "7f0005000000-7f0009000000 r--p 00000000 00:00 0 \n"
                             "7f0010000000-7f0014000000 rw-s 00000000 00:05 77   /dev/zero (deleted)\n"
                             "7f0020000000-7f0024000000 rw-p 00000000 00:00 0    [anon:glibc: malloc arena]\n"
                             "7ffc00000000-7ffc00800000 rw-p 00000000 00:00 0    [stack]\n"
                             "7ffc10000000-7ffc10002000 r--p 00000000 00:00 0    [vvar]\n";
} // namespace

TEST(ExitRelease, SelectsPrivateWritableAnonymousMappingsAboveTheFloor)
{
    // Dropped: file-backed (text and data), the 1 MiB mapping (floor), read-only, shared, [stack], [vvar].
    const auto ranges = select_exit_release_ranges(MAPS, {}, {}, 2 * MiB);
    ASSERT_TRUE(ranges.has_value());
    const std::vector<MemoryRange> expected{ { 0x00a00000, 0x04a00000 },
                                             { 0x7f0000000000, 0x7f0004000000 },
                                             { 0x7f0020000000, 0x7f0024000000 } };
    EXPECT_EQ(*ranges, expected);
}

TEST(ExitRelease, CutsExcludedRangesAndDropsPinnedMappings)
{
    const std::vector<MemoryRange> excluded{
        { 0x7f0001000000, 0x7f0001400000 }, // splits the anonymous mapping
        { 0x7f0003f00000, 0x7f0004000000 }, // trims its tail
        { 0x00b00000, 0x04900000 },         // leaves two 1 MiB pieces of [heap], both under the floor
    };
    const std::vector<uintptr_t> pinned{ 0x7f0020001000 }; // inside the arena mapping: dropped whole
    const auto ranges = select_exit_release_ranges(MAPS, excluded, pinned, 2 * MiB);
    ASSERT_TRUE(ranges.has_value());
    const std::vector<MemoryRange> expected{ { 0x7f0000000000, 0x7f0001000000 }, { 0x7f0001400000, 0x7f0003f00000 } };
    EXPECT_EQ(*ranges, expected);
}

TEST(ExitRelease, RejectsUnparsableMaps)
{
    EXPECT_FALSE(select_exit_release_ranges("not a maps line\n", {}, {}, 0).has_value());
    EXPECT_FALSE(select_exit_release_ranges("00a00000-00900000 rw-p 00000000 00:00 0\n", {}, {}, 0).has_value());
    EXPECT_FALSE(select_exit_release_ranges("00a00000-00b00000 rw-p\n", {}, {}, 0).has_value());
}

#if defined(__linux__) && !defined(__wasm__) && !defined(NO_MULTITHREADING)
// Every condition the fast exit cannot establish leaves the process exactly as it was, having written nothing, so the
// caller's normal return runs the destructors and writes the BB_BENCH report itself. Each fake below fails before
// anything is released; a fake that did not fail would end this test binary with exit code 99. A fake count of this
// thread plus its parallel_for workers passes the thread check, since stopping the workers brings it to one.
class ExitReleaseProcess : public ::testing::Test {
  protected:
    void SetUp() override
    {
        setenv("BB_BENCH", "1", 1);
        saved_cerr = std::cerr.rdbuf(captured.rdbuf());
        pthread_sigmask(SIG_SETMASK, nullptr, &mask_before);
        parallel_for(64, [](size_t) {});         // the call stops this thread's workers; they must come back
        ballast.assign(256 * MiB, uint8_t{ 1 }); // enough private memory to get past the size check
    }
    void TearDown() override
    {
        std::cerr.rdbuf(saved_cerr);
        unsetenv("BB_BENCH");
    }
    void expect_normal_exit_path(bool released)
    {
        EXPECT_FALSE(released);
        EXPECT_EQ(captured.str(), ""); // no BB_BENCH report: the destructor still owes it
        sigset_t mask_after;
        pthread_sigmask(SIG_SETMASK, nullptr, &mask_after);
        for (int sig = 1; sig < NSIG; ++sig) {
            EXPECT_EQ(sigismember(&mask_after, sig), sigismember(&mask_before, sig)) << "signal " << sig;
        }
        EXPECT_TRUE(std::ranges::all_of(ballast, [](uint8_t b) { return b == 1; }));
        std::atomic<size_t> iterations = 0;
        parallel_for(64, [&](size_t) { iterations++; });
        EXPECT_EQ(iterations.load(), 64U);
    }
    std::stringstream captured;
    std::streambuf* saved_cerr = nullptr;
    sigset_t mask_before{};
    std::vector<uint8_t> ballast;
};

TEST_F(ExitReleaseProcess, AnotherThreadAlive)
{
    std::atomic<bool> stop = false;
    std::thread other([&] {
        while (!stop.load()) {
            std::this_thread::yield();
        }
    });
    const bool released = release_memory_and_exit(99);
    stop.store(true);
    other.join();
    expect_normal_exit_path(released);
}

TEST_F(ExitReleaseProcess, UncountedThreadsLazyBindingScratchFailureAndThrowingPreparation)
{
    std::vector<ExitReleaseSystem> systems(4, default_exit_release_system());
    for (ExitReleaseSystem& system : systems) {
        system.count_threads = [] { return parallel_for_num_workers() + 1; };
    }
    systems[3].count_threads = [] { return size_t{ 1 }; }; // a /proc that does not count threads
    systems[0].binds_immediately = [] { return false; };
    systems[1].map_scratch = [](size_t) -> void* { return nullptr; };
    systems[2].map_scratch = [](size_t) -> void* { throw std::bad_alloc(); };
    for (const ExitReleaseSystem& system : systems) {
        expect_normal_exit_path(release_memory_and_exit(99, system));
    }
}

TEST_F(ExitReleaseProcess, SecondReleaseThreadFailsToStart)
{
    static int starts = 0;
    static std::atomic<pid_t> first_tid = 0;
    static void* (*first_fn)(void*) = nullptr;
    starts = 0;
    first_tid = 0;
    ExitReleaseSystem system = default_exit_release_system();
    system.count_threads = [] { return parallel_for_num_workers() + 1; };
    system.binds_immediately = [] { return true; };
    system.start_thread = [](void* handle, void* stack, size_t bytes, void* (*fn)(void*), void* arg) {
        if (++starts == 2) {
            return -1;
        }
        first_fn = fn;
        const auto recording = [](void* a) -> void* {
            first_tid = gettid();
            return first_fn(a);
        };
        return default_exit_release_system().start_thread(handle, stack, bytes, recording, arg);
    };
    const size_t cpus = get_num_cpus();
    set_parallel_for_concurrency(16); // at least three release threads, so a second start is attempted
    const bool released = release_memory_and_exit(99, system);
    set_parallel_for_concurrency(cpus);
    if (starts == 0) {
        expect_normal_exit_path(released);
        GTEST_SKIP() << "the release is compiled out (sanitizer or coverage build)";
    }
    EXPECT_EQ(starts, 2);
    ASSERT_NE(first_tid.load(), 0);
    // The thread that did start was joined: it leaves /proc once the kernel finishes its exit.
    const std::string task = "/proc/self/task/" + std::to_string(first_tid.load());
    for (int i = 0; i < 1000 && std::filesystem::exists(task); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_FALSE(std::filesystem::exists(task));
    expect_normal_exit_path(released);
}

TEST_F(ExitReleaseProcess, FastPathInAFreshProcessExitsWithItsCodeAfterItsOutput)
{
    const ExitReleaseSystem system = default_exit_release_system();
    if (system.count_threads == nullptr || !system.binds_immediately() || get_num_cpus() < 2) {
        GTEST_SKIP() << "the release is compiled out, or this binary binds lazily, or there is one CPU";
    }
    // The threadsafe style re-runs this binary up to the statement, so the child is a fresh process whose only other
    // threads are the workers SetUp started. The child reads the output path from the environment.
    const char* inherited = std::getenv("BB_EXIT_RELEASE_TEST_OUT");
    const std::string out =
        inherited != nullptr ? inherited : ::testing::TempDir() + "exit_release_" + std::to_string(getpid());
    setenv("BB_EXIT_RELEASE_TEST_OUT", out.c_str(), 1);
    std::string bytes(MiB, '\0');
    for (size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<char>(i * 131 % 251);
    }
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(
        {
            std::ofstream(out, std::ios::binary) << bytes;
            std::cerr.rdbuf(saved_cerr); // BB_BENCH is set: the report must reach the real stderr before the release
            release_memory_and_exit(0);
            std::_Exit(3); // the normal exit was taken
        },
        ::testing::ExitedWithCode(0),
        "enchmark");
    unsetenv("BB_EXIT_RELEASE_TEST_OUT");
    std::stringstream written;
    written << std::ifstream(out, std::ios::binary).rdbuf();
    EXPECT_TRUE(written.str() == bytes);
    std::filesystem::remove(out);
}
#endif
