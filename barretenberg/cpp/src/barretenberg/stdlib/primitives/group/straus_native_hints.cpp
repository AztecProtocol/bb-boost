#include "barretenberg/stdlib/primitives/group/straus_native_hints.hpp"

#include "barretenberg/common/thread.hpp"
#include "barretenberg/numeric/general/general.hpp"

#include <algorithm>
#include <vector>

namespace bb::stdlib {
namespace {

using Group = bb::grumpkin::g1;
using Element = Group::element;
using AffineElement = Group::affine_element;

// Fixed so that the chunks, and with them every intermediate value, do not depend on the thread count.
constexpr size_t POINTS_PER_CHUNK = 1024;

const AffineElement& table_read(const StrausTableReads& reads, const size_t round, const size_t point)
{
    return reads.tables[point][static_cast<size_t>(reads.slices[point][reads.num_rounds - 1 - round])];
}

StrausNativeHints allocate_hints(const StrausTableReads& reads)
{
    StrausNativeHints hints;
    hints.transcript_size = (reads.num_rounds * reads.tables.size()) + ((reads.num_rounds - 1) * reads.table_bits);
    hints.operation_transcript = std::make_unique_for_overwrite<Element[]>(hints.transcript_size);
    // Overwritten by the caller, but a return that is not elided copies it.
    hints.offset_generator_accumulator = Group::point_at_infinity;
    return hints;
}

} // namespace

bool straus_hints_use_threads(const size_t num_reads)
{
    return num_reads >= STRAUS_HINTS_MIN_PARALLEL_READS && get_num_cpus() >= STRAUS_HINTS_MIN_THREADS;
}

StrausNativeHints compute_straus_hints(const StrausTableReads& reads)
{
    return straus_hints_use_threads(reads.num_rounds * reads.tables.size()) ? compute_straus_hints_parallel(reads)
                                                                            : compute_straus_hints_serial(reads);
}

StrausNativeHints compute_straus_hints_serial(const StrausTableReads& reads)
{
    const size_t num_points = reads.tables.size();
    StrausNativeHints hints = allocate_hints(reads);
    Element* transcript = hints.operation_transcript.get();
    size_t num_operations = 0;

    // Every round adds the same offset generators, so the offset generator accumulator adds their sum once per round.
    Element offset_generator_sum = Group::point_at_infinity;
    for (size_t j = 0; j < num_points; ++j) {
        offset_generator_sum += reads.offset_generators[j + 1];
    }

    Element accumulator = reads.offset_generators[0];
    Element offset_generator_accumulator = reads.offset_generators[0];
    for (size_t i = 0; i < reads.num_rounds; ++i) {
        if (i != 0) {
            for (size_t j = 0; j < reads.table_bits; ++j) {
                accumulator = accumulator.dbl();
                transcript[num_operations++] = accumulator;
                offset_generator_accumulator = offset_generator_accumulator.dbl();
            }
        }
        for (size_t j = 0; j < num_points; ++j) {
            accumulator += table_read(reads, i, j);
            transcript[num_operations++] = accumulator;
        }
        offset_generator_accumulator += offset_generator_sum;
    }
    hints.offset_generator_accumulator = offset_generator_accumulator;
    return hints;
}

StrausNativeHints compute_straus_hints_parallel(const StrausTableReads& reads)
{
    const size_t num_points = reads.tables.size();
    const size_t num_rounds = reads.num_rounds;
    const size_t table_bits = reads.table_bits;
    const size_t chunks_per_round = numeric::ceil_div(num_points, POINTS_PER_CHUNK);
    const size_t num_chunks = num_rounds * chunks_per_round;
    // Round i's additions start at entry i * (num_points + table_bits), right after its doublings.
    const auto first_addition = [&](const size_t round) { return round * (num_points + table_bits); };
    const auto first_point = [&](const size_t chunk) { return (chunk % chunks_per_round) * POINTS_PER_CHUNK; };
    const auto end_point = [&](const size_t chunk) {
        return std::min(first_point(chunk) + POINTS_PER_CHUNK, num_points);
    };

    std::vector<Element> offset_generator_sums(chunks_per_round);
    parallel_for(chunks_per_round, [&](const size_t chunk) {
        Element sum = Group::point_at_infinity;
        for (size_t j = first_point(chunk); j < end_point(chunk); ++j) {
            sum += reads.offset_generators[j + 1];
        }
        offset_generator_sums[chunk] = sum;
    });
    std::vector<Element> chunk_sums(num_chunks);
    parallel_for(num_chunks, [&](const size_t chunk) {
        const size_t round = chunk / chunks_per_round;
        Element sum = Group::point_at_infinity;
        for (size_t j = first_point(chunk); j < end_point(chunk); ++j) {
            sum += table_read(reads, round, j);
        }
        chunk_sums[chunk] = sum;
    });

    StrausNativeHints hints = allocate_hints(reads);
    Element* transcript = hints.operation_transcript.get();
    Element offset_generator_sum = Group::point_at_infinity;
    for (const Element& sum : offset_generator_sums) {
        offset_generator_sum += sum;
    }
    std::vector<Element> chunk_starts(num_chunks);
    bool any_infinity = false;
    Element accumulator = reads.offset_generators[0];
    Element offset_generator_accumulator = reads.offset_generators[0];
    for (size_t i = 0; i < num_rounds; ++i) {
        if (i != 0) {
            for (size_t j = 0; j < table_bits; ++j) {
                accumulator = accumulator.dbl();
                transcript[first_addition(i) - table_bits + j] = accumulator;
                any_infinity |= accumulator.is_point_at_infinity();
                offset_generator_accumulator = offset_generator_accumulator.dbl();
            }
        }
        for (size_t k = 0; k < chunks_per_round; ++k) {
            chunk_starts[(i * chunks_per_round) + k] = accumulator;
            accumulator += chunk_sums[(i * chunks_per_round) + k];
        }
        offset_generator_accumulator += offset_generator_sum;
    }
    hints.offset_generator_accumulator = offset_generator_accumulator;

    std::vector<uint8_t> chunk_has_infinity(num_chunks, 0);
    parallel_for(num_chunks, [&](const size_t chunk) {
        const size_t round = chunk / chunks_per_round;
        Element* round_additions = transcript + first_addition(round);
        Element value = chunk_starts[chunk];
        bool has_infinity = false;
        for (size_t j = first_point(chunk); j < end_point(chunk); ++j) {
            value += table_read(reads, round, j);
            round_additions[j] = value;
            has_infinity |= value.is_point_at_infinity();
        }
        chunk_has_infinity[chunk] = static_cast<uint8_t>(has_infinity);
    });
    if (any_infinity || std::ranges::any_of(chunk_has_infinity, [](const uint8_t flag) { return flag != 0; })) {
        return compute_straus_hints_serial(reads);
    }
    return hints;
}

} // namespace bb::stdlib
