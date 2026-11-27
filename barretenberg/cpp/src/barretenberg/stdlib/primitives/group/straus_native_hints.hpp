#pragma once

#include "barretenberg/ecc/curves/grumpkin/grumpkin.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace bb::stdlib {

/**
 * @brief The native table reads of the Straus loop of a fixed-base batch multiplication.
 * @details Round i of num_rounds reads, for each point j in order, tables[j][slices[j][num_rounds - 1 - i]]: tables[j]
 * holds the 2^table_bits entries of point j and slices[j] the num_rounds slices of its scalar, least significant first.
 * offset_generators[0] starts the accumulator, and offset_generators[j + 1] is the offset in every entry of tables[j].
 */
struct StrausTableReads {
    std::span<const bb::grumpkin::g1::affine_element> offset_generators;
    std::span<const bb::grumpkin::g1::affine_element* const> tables;
    std::span<const uint64_t* const> slices;
    size_t num_rounds = 0;
    size_t table_bits = 0;
};

/**
 * @brief The native values the in-circuit Straus loop takes as hints.
 * @details operation_transcript[k] is the accumulator after the k-th operation of the loop: in each round after the
 * first, table_bits doublings, then one addition per point. offset_generator_accumulator is the sum of the offset
 * generator terms the final accumulator carries.
 */
struct StrausNativeHints {
    std::unique_ptr<bb::grumpkin::g1::element[]> operation_transcript;
    size_t transcript_size = 0;
    bb::grumpkin::g1::element offset_generator_accumulator;
};

/**
 * Below this many table reads, or with fewer threads than STRAUS_HINTS_MIN_THREADS, the hints are computed serially.
 * The threaded computation adds every table entry twice, so it needs about four threads to halve the time, and the
 * serial chain of 2^16 reads is short against a circuit of that size; the bound also keeps every multiplication but
 * the largest on the serial code.
 */
inline constexpr size_t STRAUS_HINTS_MIN_PARALLEL_READS = size_t{ 1 } << 16;
inline constexpr size_t STRAUS_HINTS_MIN_THREADS = 4;

/** Whether compute_straus_hints uses several threads for num_reads table reads at the current thread count. */
bool straus_hints_use_threads(size_t num_reads);

/** The hints, computed serially or on several threads by straus_hints_use_threads; the values do not depend on it. */
StrausNativeHints compute_straus_hints(const StrausTableReads& reads);

/** The hints, computed one operation after the other. */
StrausNativeHints compute_straus_hints_serial(const StrausTableReads& reads);

/**
 * @brief The hints, computed on several threads.
 * @details Each round's reads are cut into chunks of a fixed number of points. The chunk sums are computed in parallel,
 * the accumulator before each chunk follows serially from them and the doublings, and each chunk then fills its
 * entries in parallel. Group addition is associative and every formula used is complete on Grumpkin (infinity, equal
 * and opposite points), so every entry is the serial entry's group element, and its affine coordinates, which are
 * what becomes a hint, are unique. A point at infinity has no unique coordinates and its hint carries its raw
 * projective ones, so if any entry is at infinity the serial computation is returned instead.
 */
StrausNativeHints compute_straus_hints_parallel(const StrausTableReads& reads);

} // namespace bb::stdlib
