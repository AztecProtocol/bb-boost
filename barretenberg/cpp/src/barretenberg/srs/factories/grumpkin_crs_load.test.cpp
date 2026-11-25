#include "barretenberg/api/file_io.hpp"
#include "barretenberg/common/log.hpp"
#include "barretenberg/common/serialize.hpp"
#include "barretenberg/common/thread.hpp"
#include "barretenberg/common/throw_or_abort.hpp"
#include "barretenberg/constants.hpp"
#include "barretenberg/ecc/curves/grumpkin/grumpkin.hpp"
#include "barretenberg/srs/factories/get_grumpkin_crs.hpp"
#include "barretenberg/srs/factories/grumpkin_crs_data.hpp"
#include "barretenberg/srs/factories/grumpkin_srs_gen.hpp"
#include <array>
#include <filesystem>
#include <functional>
#include <gtest/gtest.h>
#include <span>
#include <string>
#include <unistd.h>

using namespace bb;

namespace {
using Point = curve::Grumpkin::AffineElement;
constexpr size_t CHUNK = srs::GRUMPKIN_G1_CHUNK_SIZE_POINTS;
constexpr const char* FILE_NAME = "grumpkin_g1_v2.flat.dat";
// parallel_for concurrencies to load at; 0 leaves the setting alone, since without multithreading it cannot be set.
#ifdef NO_MULTITHREADING
constexpr std::array<size_t, 1> CONCURRENCIES{ 0 };
#else
constexpr std::array<size_t, 3> CONCURRENCIES{ 1, 3, 16 };
#endif

// The cached-file path as a single serial read of the whole-chunk prefix and a serial parse of it. As in bb v6.1.0, it
// isn't hashed here: the CRS constructor checks the points against the pinned chunk hashes.
std::vector<Point> serial_load(const std::filesystem::path& dir, size_t num_points)
{
    const size_t verify_points = ((num_points + CHUNK - 1) / CHUNK) * CHUNK;
    const size_t verify_bytes = verify_points * sizeof(Point);
    if (get_file_size(dir / FILE_NAME) < verify_bytes) {
        throw_or_abort(format("Grumpkin CRS cache needs ",
                              verify_points,
                              " points for integrity verification, but generation is not allowed"));
    }
    const auto data = read_file(dir / FILE_NAME, verify_bytes);
    std::vector<Point> points(verify_points);
    for (size_t i = 0; i < verify_points; ++i) {
        points[i] = from_buffer<Point>(data, i * sizeof(Point));
    }
    return points;
}

// Without exceptions a failed load aborts, so error paths are only exercised when exceptions are enabled.
std::string error_of(const std::function<void()>& load)
{
#ifdef BB_NO_EXCEPTIONS
    load();
    return "";
#else
    try {
        load();
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
#endif
}
} // namespace

// Files of several pinned chunks, built from the canonical generator: the whole file, one whose last chunk is partial,
// and one with a corrupted third chunk. For every request size and parallel_for concurrency the loader must return
// the serial path's points (the whole chunks the request needs), and fail exactly where the serial path fails.
class GrumpkinCrsLoad : public ::testing::Test {
  protected:
    static void SetUpTestSuite()
    {
        root = std::filesystem::temp_directory_path() / ("grumpkin-crs-load-" + std::to_string(getpid()));
        canonical = srs::generate_grumpkin_srs(srs::GRUMPKIN_G1_NUM_POINTS);
        const auto bytes = to_buffer(canonical);
        ASSERT_EQ(bytes.size(), srs::GRUMPKIN_G1_SIZE_BYTES);
        verify_grumpkin_crs_integrity(bytes);

        const auto write = [&](const char* name, std::span<const uint8_t> data) {
            std::filesystem::create_directories(root / name);
            write_file(root / name / FILE_NAME, data);
        };
        write("full", bytes);
        write("partial", std::span(bytes).first((2 * CHUNK + CHUNK / 2) * sizeof(Point)));
        auto corrupt = bytes;
        corrupt[(2 * CHUNK + 5) * sizeof(Point)] ^= 1;
        write("corrupt", corrupt);
    }
    static void TearDownTestSuite() { std::filesystem::remove_all(root); }

    static void expect_same(const char* file, size_t num_points)
    {
        const auto dir = root / file;
        const size_t original_concurrency = get_num_cpus();
        const std::string serial_error = error_of([&] { serial_load(dir, num_points); });
        for (const size_t concurrency : CONCURRENCIES) {
            if (concurrency != 0) {
                set_parallel_for_concurrency(concurrency);
            }
            const std::string context = std::string(file) + ", num_points " + std::to_string(num_points) +
                                        ", concurrency " + std::to_string(concurrency);
            if (!serial_error.empty()) {
                EXPECT_EQ(error_of([&] { get_grumpkin_g1_data(dir, num_points, /*allow_download=*/false); }),
                          serial_error)
                    << context;
                continue;
            }
            const auto points = get_grumpkin_g1_data(dir, num_points, /*allow_download=*/false);
            EXPECT_EQ(points, serial_load(dir, num_points)) << context;
        }
        if (CONCURRENCIES[0] != 0) {
            set_parallel_for_concurrency(original_concurrency);
        }
        if (serial_error.empty() && std::string(file) != "corrupt") {
            const auto serial = serial_load(dir, num_points);
            ASSERT_GE(serial.size(), num_points) << file;
            EXPECT_TRUE(std::equal(serial.begin(), serial.end(), canonical.begin())) << file;
        }
    }

    static inline std::filesystem::path root;
    static inline std::vector<Point> canonical;
};

TEST_F(GrumpkinCrsLoad, WholeFileMatchesSerialPath)
{
    for (const size_t n : { size_t{ 1 },
                            size_t{ 1 } << CONST_ECCVM_LOG_N,
                            CHUNK,
                            CHUNK + 1,
                            2 * CHUNK + 7,
                            3 * CHUNK,
                            srs::GRUMPKIN_G1_NUM_POINTS }) {
        expect_same("full", n);
    }
}

TEST_F(GrumpkinCrsLoad, PartialLastChunkMatchesSerialPath)
{
#ifdef BB_NO_EXCEPTIONS
    GTEST_SKIP() << "a short cache aborts without exceptions";
#endif
    for (const size_t n : { size_t{ 1 }, CHUNK + 1, 2 * CHUNK, 2 * CHUNK + 1 }) {
        expect_same("partial", n);
    }
    EXPECT_FALSE(error_of([&] { serial_load(root / "partial", 2 * CHUNK + 1); }).empty());
}

TEST_F(GrumpkinCrsLoad, CorruptChunkMatchesSerialPath)
{
#ifdef BB_NO_EXCEPTIONS
    GTEST_SKIP() << "a corrupt chunk aborts without exceptions";
#endif
    for (const size_t n : { size_t{ 1 } << CONST_ECCVM_LOG_N, 2 * CHUNK, 2 * CHUNK + 1, srs::GRUMPKIN_G1_NUM_POINTS }) {
        expect_same("corrupt", n);
    }
    // Loading doesn't hash in v6.1.0 (the CRS constructor does): a corrupted cache loads, or fails to parse, exactly as
    // the serial path does.
}
