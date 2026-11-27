#pragma once
#include <barretenberg/ecc/curves/bn254/g1.hpp>
#include <barretenberg/ecc/curves/bn254/g2.hpp>
#include <filesystem>
#include <fstream>
#include <ios>
#include <memory>

namespace bb {
std::vector<g1::affine_element> get_bn254_g1_data(const std::filesystem::path& path,
                                                  size_t num_points,
                                                  bool allow_download = true);

// Overload with custom URLs for testing fallback behavior
std::vector<g1::affine_element> get_bn254_g1_data(const std::filesystem::path& path,
                                                  size_t num_points,
                                                  bool allow_download,
                                                  const std::string& primary_url,
                                                  const std::string& fallback_url);

/**
 * @brief The first `size` BN254 G1 CRS points in an owned array, as loaded by load_bn254_g1_points.
 */
struct Bn254G1Points {
    std::shared_ptr<g1::affine_element[]> points;
    size_t size = 0;
};

/**
 * @brief Same points as get_bn254_g1_data, in storage that is never zero-filled first: from the cached uncompressed
 * file, threads deserialize straight into freshly allocated memory.
 */
Bn254G1Points load_bn254_g1_points(const std::filesystem::path& path, size_t num_points, bool allow_download = true);

g2::affine_element get_bn254_g2_data(const std::filesystem::path& path);
} // namespace bb
