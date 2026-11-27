#include "barretenberg/numeric/random/engine.hpp"
#include "barretenberg/translator_vm/translator_proving_key.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <functional>
#include <string>
#include <vector>

using namespace bb;

namespace {
using Key = TranslatorProvingKey;
using Flavor = Key::Flavor;
using FF = Key::FF;

constexpr size_t NUM_RANGE_CONSTRAINT_GROUPS = Flavor::NUM_CONCATENATED_POLYS - 1;
constexpr size_t VALUES_PER_LANE = Flavor::MINI_CIRCUIT_SIZE - 1 - Flavor::NUM_MASKED_ROWS_END;
constexpr size_t CIRCUIT_SIZE = Flavor::MINI_CIRCUIT_SIZE * Flavor::CONCATENATION_GROUP_SIZE;
constexpr uint64_t MAX_MICRO_LIMB = (uint64_t{ 1 } << Flavor::MICRO_LIMB_BITS) - 1;
// Row of lane 15 whose slot is the last one before the runway of steps.
constexpr size_t LAST_ROW_BEFORE_RUNWAY = Key::dyadic_circuit_size_without_masking - Flavor::SORTED_STEPS_COUNT -
                                          ((Flavor::CONCATENATION_GROUP_SIZE - 1) * VALUES_PER_LANE);

// Value of row k of lane j in range-constraint group i (rows exclude the masking rows).
using Pattern = std::function<FF(size_t i, size_t j, size_t k)>;

struct Case {
    std::string name;
    Pattern pattern;
    // Wire polynomials to reshape before filling: {group, lane, number of rows}.
    std::vector<std::array<size_t, 3>> reshaped_lanes = {};
};

void fill_range_constraint_groups(Key& key, const Case& c)
{
    auto& engine = numeric::get_debug_randomness();
    auto groups = key.proving_key->polynomials.get_groups_to_be_concatenated();
    for (size_t i = 0; i < NUM_RANGE_CONSTRAINT_GROUPS; i++) {
        for (size_t j = 0; j < Flavor::CONCATENATION_GROUP_SIZE; j++) {
            size_t num_rows = Flavor::MINI_CIRCUIT_SIZE - 1;
            for (const auto& [group, lane, rows] : c.reshaped_lanes) {
                if (group == i && lane == j) {
                    num_rows = rows;
                }
            }
            groups[i][j] = Key::Polynomial(num_rows, CIRCUIT_SIZE, /*start_index=*/1);
            auto& wire = groups[i][j];
            const size_t masking_start = wire.end_index() - Flavor::NUM_MASKED_ROWS_END;
            for (size_t k = wire.start_index(); k < masking_start; k++) {
                wire.at(k) = c.pattern(i, j, k);
            }
            for (size_t k = masking_start; k < wire.end_index(); k++) {
                wire.at(k) = FF::random_element(&engine);
            }
        }
    }
}

// Gives the masking rows of the concatenated range-constraint polynomials distinct values, and fills the ordered
// polynomials with a sentinel, so that the comparison covers every coefficient.
void fill_masking_rows_and_sentinels(Key& key, const FF& sentinel)
{
    auto concatenated = key.proving_key->polynomials.get_concatenated();
    for (size_t i = 0; i < NUM_RANGE_CONSTRAINT_GROUPS; i++) {
        for (size_t j = 0; j < Flavor::CONCATENATION_GROUP_SIZE; j++) {
            for (size_t k = 0; k < Flavor::NUM_MASKED_ROWS_END; k++) {
                const size_t row =
                    (j * Flavor::MINI_CIRCUIT_SIZE) + Flavor::MINI_CIRCUIT_SIZE - Flavor::NUM_MASKED_ROWS_END + k;
                concatenated[i].at(row) =
                    FF(uint64_t{ 1000000 } + (((i * Flavor::CONCATENATION_GROUP_SIZE) + j) * 8) + k);
            }
        }
    }
    for (auto& ordered : key.proving_key->polynomials.get_ordered_range_constraints()) {
        for (size_t pos = ordered.start_index(); pos < ordered.end_index(); pos++) {
            ordered.at(pos) = sentinel;
        }
    }
}

/**
 * @brief Every coefficient of the ordered range-constraint polynomials, by the comparison sort. Each receives the
 * sorted sequence of its multiset without the first element at [1, circuit_size - MAX_RANDOM_VALUES_PER_ORDERED), and
 * its share of the concatenated polynomials' masking values at the end; the coefficients between keep the sentinel.
 * Lanes are written in order, so an overlong lane is overwritten by the next one as in the sorting vector.
 */
std::array<std::vector<FF>, Flavor::NUM_CONCATENATED_POLYS> reference_ordered_values(Key& key, const FF& sentinel)
{
    auto groups = key.proving_key->polynomials.get_groups_to_be_concatenated();
    const auto steps = Key::get_sorted_steps();
    const size_t vector_size = Key::dyadic_circuit_size_without_masking;
    const size_t free_space = vector_size - steps.size();
    const size_t end = key.proving_key->polynomials.get_polynomial_size() - Flavor::MAX_RANDOM_VALUES_PER_ORDERED;

    std::array<std::vector<uint64_t>, Flavor::NUM_CONCATENATED_POLYS> sorted;
    std::vector<uint64_t> extra(vector_size, 0);
    for (size_t i = 0; i < NUM_RANGE_CONSTRAINT_GROUPS; i++) {
        std::vector<uint64_t> values(vector_size, 0);
        size_t extra_offset = i * steps.size();
        for (size_t j = 0; j < Flavor::CONCATENATION_GROUP_SIZE; j++) {
            const auto& wire = groups[i][j];
            for (size_t k = wire.start_index(); k < wire.end_index() - Flavor::NUM_MASKED_ROWS_END; k++) {
                const size_t idx = (j * VALUES_PER_LANE) + (k - wire.start_index());
                const uint64_t value = static_cast<uint32_t>(uint256_t(wire[k]).data[0]);
                if (idx < free_space) {
                    values[idx] = value;
                } else {
                    extra[extra_offset++] = value;
                }
            }
        }
        std::copy(steps.begin(), steps.end(), values.begin() + static_cast<std::ptrdiff_t>(free_space));
        std::sort(values.begin(), values.end());
        sorted[i] = std::move(values);
    }
    std::copy(steps.begin(),
              steps.end(),
              extra.begin() + static_cast<std::ptrdiff_t>(NUM_RANGE_CONSTRAINT_GROUPS * steps.size()));
    std::sort(extra.begin(), extra.end());
    sorted[NUM_RANGE_CONSTRAINT_GROUPS] = std::move(extra);

    std::vector<FF> masking_values;
    auto concatenated = key.proving_key->polynomials.get_concatenated();
    for (size_t i = 0; i < NUM_RANGE_CONSTRAINT_GROUPS; i++) {
        for (size_t j = 0; j < Flavor::CONCATENATION_GROUP_SIZE; j++) {
            for (size_t k = 0; k < Flavor::NUM_MASKED_ROWS_END; k++) {
                masking_values.push_back(concatenated[i][(j * Flavor::MINI_CIRCUIT_SIZE) + Flavor::MINI_CIRCUIT_SIZE -
                                                         Flavor::NUM_MASKED_ROWS_END + k]);
            }
        }
    }
    const size_t per_ordered = masking_values.size() / Flavor::NUM_CONCATENATED_POLYS;
    const size_t remainder = masking_values.size() % Flavor::NUM_CONCATENATED_POLYS;

    std::array<std::vector<FF>, Flavor::NUM_CONCATENATED_POLYS> result;
    for (size_t i = 0; i < Flavor::NUM_CONCATENATED_POLYS; i++) {
        result[i].assign(CIRCUIT_SIZE, sentinel);
        result[i][0] = FF(0);
        for (size_t pos = 1; pos < end; pos++) {
            result[i][pos] = FF(sorted[i][pos]);
        }
        // Dealt round-robin, as split_concatenated_random_coefficients_to_ordered does (bb v6.1.0): masking value
        // (m * NUM_CONCATENATED_POLYS) + i is ordered polynomial i's m-th.
        const size_t num_masking = per_ordered + (i < remainder ? 1 : 0);
        for (size_t m = 0; m < num_masking; m++) {
            result[i][CIRCUIT_SIZE - num_masking + m] = masking_values[(m * Flavor::NUM_CONCATENATED_POLYS) + i];
        }
    }
    return result;
}

std::vector<Case> adversarial_cases()
{
    auto& engine = numeric::get_debug_randomness();
    const auto random_micro_limb = [&engine](size_t, size_t, size_t) {
        return FF(engine.get_random_uint16() & MAX_MICRO_LIMB);
    };
    // Position of row k of lane j in its group's sorting vector, increasing across the group.
    const auto flat_index = [](size_t j, size_t k) { return (j * VALUES_PER_LANE) + k; };
    return {
        { "all equal at the maximum", [](size_t, size_t, size_t) { return FF(MAX_MICRO_LIMB); } },
        { "all zero", [](size_t, size_t, size_t) { return FF(0); } },
        { "empty tail",
          [&engine](size_t, size_t, size_t k) {
              return k < 64 ? FF(engine.get_random_uint16() & MAX_MICRO_LIMB) : FF(0);
          } },
        { "already sorted", [flat_index](size_t, size_t j, size_t k) { return FF(flat_index(j, k) / 8); } },
        { "reversed",
          [flat_index](size_t, size_t j, size_t k) { return FF(MAX_MICRO_LIMB - (flat_index(j, k) / 8)); } },
        { "maximum range", random_micro_limb },
        // Each value names its group and lane, so a value counted in the wrong polynomial, or an overflow entry
        // written to the wrong offset, changes a multiset.
        { "group and lane signatures",
          [](size_t i, size_t j, size_t k) { return FF((i << 12) | (j << 8) | (k & 0xff)); } },
        // 2^32 + 5 truncates to 5, as the sorting vectors store it.
        { "value truncating into the range",
          [&engine](size_t i, size_t j, size_t k) {
              return (i == 3 && j == 0 && k == 1) ? FF((uint64_t{ 1 } << 32) + 5)
                                                  : FF(engine.get_random_uint16() & MAX_MICRO_LIMB);
          } },
        // Out-of-range values below the runway of group 1, at the last slot before the runway of group 3, and in the
        // overflow of group 2, which feeds the fifth polynomial.
        { "out of range",
          [&engine](size_t i, size_t j, size_t k) {
              constexpr size_t LAST_LANE = Flavor::CONCATENATION_GROUP_SIZE - 1;
              if (i == 1 && j == 0 && k <= 3) {
                  const std::array<FF, 3> out_of_range{ FF(MAX_MICRO_LIMB + 1), FF(uint64_t{ 0xffffffff }), FF(-1) };
                  return out_of_range[k - 1];
              }
              if ((i == 2 && j == LAST_LANE && k == VALUES_PER_LANE) ||
                  (i == 3 && j == LAST_LANE && k == LAST_ROW_BEFORE_RUNWAY)) {
                  return FF(MAX_MICRO_LIMB + 1);
              }
              return FF(engine.get_random_uint16() & MAX_MICRO_LIMB);
          } },
        // A short lane leaves zero slots below the runway; an overlong one overwrites the next lane's first slots.
        { "short and overlong lanes",
          random_micro_limb,
          { { 0, 3, Flavor::MINI_CIRCUIT_SIZE - 1 - 2000 }, { 2, 6, Flavor::MINI_CIRCUIT_SIZE - 1 + 50 } } },
    };
}
} // namespace

// Every coefficient of the ordered range-constraint polynomials equals the comparison sort of their multisets,
// including ties, padding zeros, the overflow into the fifth polynomial, layouts or values outside the counting path,
// and the masking tail.
TEST(TranslatorOrderedRangeSort, MatchesComparisonSort)
{
    Key key;
    key.proving_key = std::make_shared<Key::ProvingKey>();
    const FF sentinel = FF(uint64_t{ 0x5e7715e1 });
    for (const auto& c : adversarial_cases()) {
        fill_range_constraint_groups(key, c);
        fill_masking_rows_and_sentinels(key, sentinel);
        const auto expected = reference_ordered_values(key, sentinel);
        key.compute_translator_range_constraint_ordered_polynomials();

        auto ordered = key.proving_key->polynomials.get_ordered_range_constraints();
        for (size_t i = 0; i < Flavor::NUM_CONCATENATED_POLYS; i++) {
            EXPECT_EQ(ordered[i].start_index(), 1U) << c.name;
            EXPECT_EQ(ordered[i].end_index(), CIRCUIT_SIZE) << c.name;
            size_t mismatches = 0;
            for (size_t pos = 0; pos < expected[i].size(); pos++) {
                mismatches += static_cast<size_t>(ordered[i][pos] != expected[i][pos]);
            }
            EXPECT_EQ(mismatches, 0U) << c.name << ": ordered_range_constraints_" << i;
        }
    }
}
