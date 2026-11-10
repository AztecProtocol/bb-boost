#include "barretenberg/common/get_bytecode.hpp"
#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

#ifndef __wasm__
#include <libdeflate.h>
#include <memory>

namespace {

std::vector<uint8_t> gzip(const std::vector<uint8_t>& data)
{
    auto compressor =
        std::unique_ptr<libdeflate_compressor, void (*)(libdeflate_compressor*)>{ libdeflate_alloc_compressor(6),
                                                                                  libdeflate_free_compressor };
    std::vector<uint8_t> out(libdeflate_gzip_compress_bound(compressor.get(), data.size()));
    out.resize(libdeflate_gzip_compress(compressor.get(), data.data(), data.size(), out.data(), out.size()));
    return out;
}

std::vector<uint8_t> pattern(size_t size, uint8_t seed)
{
    std::vector<uint8_t> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<uint8_t>((i * 31) + seed);
    }
    return data;
}

// Scratch storage the test can inspect afterwards: filled with a sentinel, so bytes the attempt never wrote keep it.
constexpr uint8_t SENTINEL = 0xA5;
std::vector<uint8_t>& observed_scratch()
{
    static std::vector<uint8_t> storage;
    return storage;
}
GzipScratch sentinel_scratch(size_t size)
{
    observed_scratch().assign(size, SENTINEL);
    return { observed_scratch().data(), [](uint8_t*) {} };
}
GzipScratch failing_scratch(size_t /*size*/)
{
    return { nullptr, [](uint8_t*) {} };
}

} // namespace

TEST(GzipDecompress, SingleMemberIsSizedFromItsTrailer)
{
    const auto data = pattern(size_t{ 3 } << 20, 7);
    const auto out = gzip_decompress(gzip(data));
    EXPECT_EQ(out, data);
    EXPECT_EQ(out.capacity(), data.size());
}

// libdeflate inflates only the first member, while the trailer at the end of the input is the last member's. The
// attempt sized from that trailer must write no more than the first member, so it costs only the pages it fills.
TEST(GzipDecompress, ConcatenatedMembersTouchOnlyTheFirstMembersShareOfTheTrailerSizedAttempt)
{
    const auto first = pattern(size_t{ 1 } << 10, 1);
    const auto last = pattern(size_t{ 16 } << 20, 2);
    auto gz = gzip(first);
    const auto gz_last = gzip(last);
    gz.insert(gz.end(), gz_last.begin(), gz_last.end());

    EXPECT_EQ(gzip_decompress(gz, sentinel_scratch), first);
    const auto& scratch = observed_scratch();
    ASSERT_EQ(scratch.size(), last.size());
    // The inflater's fast path may overrun its output position by a few hundred bytes within the buffer.
    constexpr size_t OVERRUN_SLACK = 4096;
    EXPECT_TRUE(std::all_of(scratch.begin() + static_cast<std::ptrdiff_t>(first.size() + OVERRUN_SLACK),
                            scratch.end(),
                            [](uint8_t b) { return b == SENTINEL; }));
}

TEST(GzipDecompress, UnavailableScratchFallsBackToDoubling)
{
    const auto data = pattern(size_t{ 3 } << 20, 9);
    EXPECT_EQ(gzip_decompress(gzip(data), failing_scratch), data);
}
#endif
