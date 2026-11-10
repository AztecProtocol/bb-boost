#include "get_bytecode.hpp"
#include "barretenberg/common/throw_or_abort.hpp"
#include "base64.hpp"
#include <string>
#include <vector>

#ifndef __wasm__
#include <fstream>
#include <libdeflate.h>
#include <new>
#include <nlohmann/json.hpp>
#include <stdexcept>
#endif

#ifndef __wasm__
namespace {

// A deflate stream expands its input at most 1032-fold (RFC 1951: a 258-byte match costs at least 2 bits), so a size
// claim above that bound is wrong and is not tried.
constexpr size_t DEFLATE_MAX_EXPANSION = 1032;
constexpr size_t GZIP_TRAILER_ISIZE_BYTES = 4;
constexpr size_t GZIP_MIN_MEMBER_BYTES = 18; // 10-byte header + 8-byte trailer

GzipScratch allocate_gzip_scratch(size_t size)
{
    // new[] of uint8_t default-initialises: pages stay untouched, so not resident, until something is written to them.
    return { new (std::nothrow) uint8_t[size], [](uint8_t* p) { delete[] p; } };
}

} // namespace
#endif

std::vector<uint8_t> gzip_decompress([[maybe_unused]] const std::vector<uint8_t>& compressed,
                                     [[maybe_unused]] GzipScratch (*scratch)(size_t))
{
#ifdef __wasm__
    throw_or_abort("gzip_decompress not supported in WASM");
#else
    auto decompressor =
        std::unique_ptr<libdeflate_decompressor, void (*)(libdeflate_decompressor*)>{ libdeflate_alloc_decompressor(),
                                                                                      libdeflate_free_decompressor };

    // A gzip member ends with ISIZE, the uncompressed size modulo 2^32 (RFC 1952, section 2.3.1). Inflating once into
    // storage of that size avoids restarting from the first byte after every doubling of a small guess. In a
    // multi-member stream ISIZE describes the last member while libdeflate inflates only the first, so the result is
    // kept only when that one member spans the whole input. The storage is not zero-filled, so a wrong guess costs only
    // the pages the first member writes; if it cannot be had, or the guess is wrong, the doubling loop runs as before.
    if (compressed.size() >= GZIP_MIN_MEMBER_BYTES) {
        const uint8_t* isize_bytes = compressed.data() + compressed.size() - GZIP_TRAILER_ISIZE_BYTES;
        const size_t isize = static_cast<size_t>(isize_bytes[0]) | (static_cast<size_t>(isize_bytes[1]) << 8) |
                             (static_cast<size_t>(isize_bytes[2]) << 16) | (static_cast<size_t>(isize_bytes[3]) << 24);
        if (isize > 0 && (isize + DEFLATE_MAX_EXPANSION - 1) / DEFLATE_MAX_EXPANSION <= compressed.size()) {
            const GzipScratch buffer = (scratch != nullptr ? scratch : allocate_gzip_scratch)(isize);
            size_t consumed = 0;
            size_t actual_size = 0;
            if (buffer != nullptr &&
                libdeflate_gzip_decompress_ex(decompressor.get(),
                                              compressed.data(),
                                              compressed.size(),
                                              buffer.get(),
                                              isize,
                                              &consumed,
                                              &actual_size) == LIBDEFLATE_SUCCESS &&
                consumed == compressed.size()) {
                return std::vector<uint8_t>(buffer.get(), buffer.get() + actual_size);
            }
        }
    }

    std::vector<uint8_t> decompressed;
    decompressed.resize(1024ULL * 128ULL); // Initial size guess

    for (;;) {
        size_t actual_size = 0;
        libdeflate_result result = libdeflate_gzip_decompress(decompressor.get(),
                                                              compressed.data(),
                                                              compressed.size(),
                                                              decompressed.data(),
                                                              decompressed.size(),
                                                              &actual_size);

        if (result == LIBDEFLATE_INSUFFICIENT_SPACE) {
            decompressed.resize(decompressed.size() * 2);
            continue;
        }
        if (result == LIBDEFLATE_BAD_DATA) {
            throw std::runtime_error("Invalid gzip data");
        }
        decompressed.resize(actual_size);
        break;
    }
    return decompressed;
#endif
}

std::vector<uint8_t> decode_bytecode(const std::string& base64_bytecode)
{
    // Decode base64 and decompress using libdeflate for gzip
    std::string decoded = base64_decode(base64_bytecode, false);
    std::vector<uint8_t> gzipped(decoded.begin(), decoded.end());
    return gzip_decompress(gzipped);
}

std::vector<uint8_t> get_bytecode_from_json([[maybe_unused]] const std::string& json_path)
{
#ifdef __wasm__
    throw_or_abort("get_bytecode_from_json not supported in WASM");
#else
    std::ifstream json_file(json_path);
    if (!json_file.is_open()) {
        throw std::runtime_error("Failed to open JSON file: " + json_path);
    }

    nlohmann::json json_data = nlohmann::json::parse(json_file);
    std::string base64_bytecode = json_data["bytecode"];

    return decode_bytecode(base64_bytecode);
#endif
}

std::vector<uint8_t> gunzip([[maybe_unused]] const std::string& path)
{
#ifdef __wasm__
    throw_or_abort("gunzip not supported in WASM");
#else
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    // One bulk read when the size is known; pipes and other unseekable inputs are streamed.
    file.seekg(0, std::ios::end);
    const std::streamoff file_size = file.tellg();
    if (file_size < 0) {
        file.clear();
        std::vector<uint8_t> compressed((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        return gzip_decompress(compressed);
    }
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> compressed(static_cast<size_t>(file_size));
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    if (!file.read(reinterpret_cast<char*>(compressed.data()), file_size)) {
        throw std::runtime_error("Failed to read file: " + path);
    }
    return gzip_decompress(compressed);
#endif
}

std::vector<uint8_t> get_bytecode([[maybe_unused]] const std::string& bytecodePath)
{
#ifdef __wasm__
    throw_or_abort("get_bytecode not supported in WASM");
#else
    if (bytecodePath == "-") {
        return { (std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>() };
    }
    std::filesystem::path filePath = bytecodePath;
    if (filePath.extension() == ".json") {
        // Try reading json files as if they are a Nargo build artifact
        return get_bytecode_from_json(bytecodePath);
    }

    // For other extensions, assume file is a raw ACIR program
    return gunzip(bytecodePath);
#endif
}
