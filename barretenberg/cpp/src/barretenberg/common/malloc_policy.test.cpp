#include "barretenberg/common/malloc_policy.hpp"
#include "barretenberg/common/log.hpp"

#include <array>
#include <cstdlib>
#include <gtest/gtest.h>
#include <string>
#include <utility>
#include <vector>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

using namespace bb;

namespace {

std::vector<std::pair<int, int>> recorded_settings;
int setter_result = 1;

int record_setting(int parameter, int value)
{
    recorded_settings.emplace_back(parameter, value);
    return setter_result;
}

// These tests record the settings through record_setting; none applies them to this process.
class MallocPolicyTest : public ::testing::Test {
  protected:
    void SetUp() override
    {
        recorded_settings.clear();
        setter_result = 1;
    }
};

const std::array<const char*, 2> UNRELATED_TUNABLE{ "GLIBC_TUNABLES=glibc.pthread.rseq=0", nullptr };

} // namespace

TEST(MallocPolicy, EnvironmentConfiguresMallocOnlyThroughMallocTunablesOrMallocVariables)
{
    const std::array<const char*, 3> plain{ "PATH=/usr/bin", "MALLOCATOR=1", nullptr };
    const std::array<const char*, 2> other_tunables{ "GLIBC_TUNABLES=glibc.pthread.rseq=0:glibc.cpu.x86_ibt=off",
                                                     nullptr };
    const std::array<const char*, 2> empty_tunables{ "GLIBC_TUNABLES=", nullptr };
    const std::array<const char*, 3> malloc_tunable{ "HOME=/root",
                                                     "GLIBC_TUNABLES=glibc.pthread.rseq=0:glibc.malloc.arena_max=2",
                                                     nullptr };
    const std::array<const char*, 2> malloc_variable{ "MALLOC_ARENA_MAX=4", nullptr };
    const std::array<const char*, 1> empty{ nullptr };

    EXPECT_FALSE(environment_configures_malloc(nullptr));
    EXPECT_FALSE(environment_configures_malloc(empty.data()));
    EXPECT_FALSE(environment_configures_malloc(plain.data()));
    EXPECT_FALSE(environment_configures_malloc(other_tunables.data()));
    EXPECT_FALSE(environment_configures_malloc(empty_tunables.data()));
    EXPECT_TRUE(environment_configures_malloc(malloc_tunable.data()));
    EXPECT_TRUE(environment_configures_malloc(malloc_variable.data()));
}

TEST(MallocPolicy, OnlyAvmProveAndUltraHonkProveWithoutLowMemoryKeepHeap)
{
    EXPECT_TRUE(command_keeps_heap("avm_prove", "ultra_honk", false));
    EXPECT_TRUE(command_keeps_heap("avm_prove", "", false));
    EXPECT_TRUE(command_keeps_heap("prove", "ultra_honk", false));

    EXPECT_FALSE(command_keeps_heap("avm_prove", "ultra_honk", true));
    EXPECT_FALSE(command_keeps_heap("prove", "ultra_honk", true));
    EXPECT_FALSE(command_keeps_heap("prove", "chonk", false));
    EXPECT_FALSE(command_keeps_heap("prove", "avm", false));
    EXPECT_FALSE(command_keeps_heap("msgpack", "ultra_honk", false)); // msgpack run
    EXPECT_FALSE(command_keeps_heap("write_vk", "ultra_honk", false));
    EXPECT_FALSE(command_keeps_heap("verify", "ultra_honk", false));
    EXPECT_FALSE(command_keeps_heap("", "ultra_honk", false));
}

TEST_F(MallocPolicyTest, ACommandOutsideTheRuleGetsNoSetting)
{
    EXPECT_EQ(configure_process_malloc(
                  command_keeps_heap("prove", "chonk", false), UNRELATED_TUNABLE.data(), &record_setting),
              MallocPolicy::NOT_THIS_COMMAND);
    EXPECT_TRUE(recorded_settings.empty());
}

TEST_F(MallocPolicyTest, AnEnvironmentThatConfiguresMallocGetsNoSetting)
{
    const std::array<const char*, 2> environment{ "MALLOC_TRIM_THRESHOLD_=131072", nullptr };
    EXPECT_EQ(configure_process_malloc(true, environment.data(), &record_setting),
              process_malloc_policy_supported() ? MallocPolicy::LEFT_TO_ENVIRONMENT : MallocPolicy::UNSUPPORTED);
    EXPECT_TRUE(recorded_settings.empty());
}

#if defined(__GLIBC__)
TEST_F(MallocPolicyTest, ACommandInTheRuleGetsTheMmapThresholdThenNoTrimming)
{
    if (!process_malloc_policy_supported()) {
        GTEST_SKIP() << "the settings are compiled out of this build";
    }
    EXPECT_EQ(configure_process_malloc(true, UNRELATED_TUNABLE.data(), &record_setting), MallocPolicy::APPLIED);
    const std::vector<std::pair<int, int>> expected{ { M_MMAP_THRESHOLD, static_cast<int>(PROCESS_MMAP_THRESHOLD) },
                                                     { M_TRIM_THRESHOLD, PROCESS_TRIM_THRESHOLD } };
    EXPECT_EQ(recorded_settings, expected);
}

TEST_F(MallocPolicyTest, ARefusedMmapThresholdStopsBeforeTheTrimSetting)
{
    if (!process_malloc_policy_supported()) {
        GTEST_SKIP() << "the settings are compiled out of this build";
    }
    setter_result = 0;
    EXPECT_EQ(configure_process_malloc(true, UNRELATED_TUNABLE.data(), &record_setting), MallocPolicy::REFUSED);
    const std::vector<std::pair<int, int>> expected{ { M_MMAP_THRESHOLD, static_cast<int>(PROCESS_MMAP_THRESHOLD) } };
    EXPECT_EQ(recorded_settings, expected);
}

// The CLI calls configure_process_malloc_and_log() at whatever log level the command set. At the default (INFO) the
// settings must still be applied, and nothing is printed; with verbose logging the outcome line is printed.
TEST_F(MallocPolicyTest, StartUpAppliesTheSettingsAtEveryLogLevelAndPrintsOnlyWhenVerbose)
{
    if (!process_malloc_policy_supported()) {
        GTEST_SKIP() << "the settings are compiled out of this build";
    }
    std::vector<std::string> lines;
    // Restores the log level and sink (which points at lines) even when an assertion returns early.
    struct RestoreLogging {
        LogLevel level = bb_log_level;
        LogFunction function = log_function;
        RestoreLogging() = default;
        RestoreLogging(const RestoreLogging&) = delete;
        RestoreLogging& operator=(const RestoreLogging&) = delete;
        RestoreLogging(RestoreLogging&&) = delete;
        RestoreLogging& operator=(RestoreLogging&&) = delete;
        ~RestoreLogging()
        {
            set_log_function(function);
            bb_log_level = level;
        }
    } restore;
    set_log_function([&lines](LogLevel /*level*/, const std::string& message) { lines.push_back(message); });

    bb_log_level = LogLevel::INFO;
    EXPECT_EQ(configure_process_malloc_and_log(true, UNRELATED_TUNABLE.data(), &record_setting), MallocPolicy::APPLIED);
    EXPECT_EQ(recorded_settings.size(), 2U);
    EXPECT_TRUE(lines.empty());

    recorded_settings.clear();
    bb_log_level = LogLevel::VERBOSE;
    EXPECT_EQ(configure_process_malloc_and_log(true, UNRELATED_TUNABLE.data(), &record_setting), MallocPolicy::APPLIED);
    EXPECT_EQ(recorded_settings.size(), 2U);
    ASSERT_EQ(lines.size(), 1U);
    EXPECT_EQ(lines[0].rfind(std::string(malloc_policy_message(MallocPolicy::APPLIED)), 0), 0U);
}

namespace {
[[noreturn]] void apply_through_glibc()
{
    std::_Exit(configure_process_malloc(true, UNRELATED_TUNABLE.data(), &mallopt) == MallocPolicy::APPLIED ? 0 : 1);
}
} // namespace

// glibc's mallopt accepts both values. Run in a fresh process (the threadsafe style re-runs this binary), so this
// test process keeps its own allocator settings.
TEST(MallocPolicy, GlibcAcceptsBothSettings)
{
    if (!process_malloc_policy_supported()) {
        GTEST_SKIP() << "the settings are compiled out of this build";
    }
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(apply_through_glibc(), ::testing::ExitedWithCode(0), "");
}
#endif
