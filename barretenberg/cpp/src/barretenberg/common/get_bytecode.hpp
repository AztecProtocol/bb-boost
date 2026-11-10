#pragma once
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

// Storage for the decompression attempt sized from the gzip trailer; nullptr if it cannot be obtained.
using GzipScratch = std::unique_ptr<uint8_t[], void (*)(uint8_t*)>;

/**
 * @brief Decompresses the first member of a gzip stream, as libdeflate_gzip_decompress does.
 * @param scratch Supplies the storage of the attempt sized from the gzip trailer (by default uninitialised new[], or
 * nullptr when that fails). Tests substitute their own to see what that attempt writes.
 */
std::vector<uint8_t> gzip_decompress(const std::vector<uint8_t>& compressed, GzipScratch (*scratch)(size_t) = nullptr);

// Parse JSON file and extract base64-encoded gzipped bytecode
std::vector<uint8_t> get_bytecode_from_json(const std::string& json_path);

// Decode base64-encoded gzipped bytecode string
std::vector<uint8_t> decode_bytecode(const std::string& base64_bytecode);

// Decompress gzipped file using libdeflate
std::vector<uint8_t> gunzip(const std::string& path);

// Get bytecode from various file formats
std::vector<uint8_t> get_bytecode(const std::string& bytecode_path);

// On Windows, std::filesystem::path doesn't implicitly convert to std::string.
#ifdef _WIN32
inline std::vector<uint8_t> get_bytecode(const std::filesystem::path& bytecode_path)
{
    return get_bytecode(bytecode_path.string());
}
#endif
