#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "barretenberg/common/thread.hpp"
#include "barretenberg/common/tuple.hpp"
#include "barretenberg/vm2/common/field.hpp"
#include "barretenberg/vm2/common/map.hpp"
#include "barretenberg/vm2/common/map_hashes.hpp"
#include "barretenberg/vm2/common/stringify.hpp"
#include "barretenberg/vm2/generated/columns.hpp"
#include "barretenberg/vm2/tracegen/lib/interaction_builder.hpp"
#include "barretenberg/vm2/tracegen/lib/permutation_builder.hpp"
#include "barretenberg/vm2/tracegen/trace_container.hpp"

namespace bb::avm2::tracegen {

/**
 * Suppose you have n permutations (P1, ..., Pn) such that
 * - The destination tuple columns (and therefore table) are the same for all of them (e.g., mem.address, mem.value).
 * - There is a "global" destination table selector (mem.sel)
 * - The source selectors are different, and already set (e.g., execution.sel_addressing_base,
 * execution.sel_addressing_indirect_0..., sha256.mem_op_1...)
 * - The destination selectors are different, but UNSET. E.g., mem.sel_addressing_base,
 * mem.sel_addressing_indirect_0...).
 *
 * You want to find, for each source row of each permutation, a row in the destination
 * table and set the right destination selector. For the next source row or permutation, you should not be able to
 * consider the "already used" destination rows, because this is a permutation.
 *
 * The rows of a tuple are handed out in a fixed order: the k-th source row with that tuple, taking the permutations in
 * order and each one's rows in increasing order, gets the k-th largest destination row with that tuple. A destination
 * row has exactly one tuple, so split() can hand disjoint sets of tuples to separate jobs that set the same selectors.
 */
template <typename... PermutationSettings_> class MultiPermutationBuilder : public InteractionBuilderInterface {
  public:
    MultiPermutationBuilder(Column dst_table_selector)
        : dst_table_selector(dst_table_selector)
    {}
    ~MultiPermutationBuilder() override = default;

    void process(TraceContainer& trace) override
    {
        std::optional<Failure> failure = assign_rows(trace);
        if (split_state != nullptr) {
            split_state->record(std::move(failure));
        } else if (failure.has_value()) {
            throw std::runtime_error(failure->message);
        }
    }

    void finish() override
    {
        if (split_state != nullptr) {
            split_state->raise();
        }
    }

    std::vector<std::unique_ptr<InteractionBuilderInterface>> split(const TraceContainer& trace) override
    {
        if (trace.get_column_rows(dst_table_selector) < MIN_ROWS_TO_SPLIT) {
            return {};
        }
        return split_into(std::min(MAX_PARTS, bb::get_num_cpus() / CPUS_PER_PART));
    }

    // Jobs for disjoint sets of tuples, assigned by a hash of the tuple's first value (equal tuples have equal first
    // values); none if parts < 2 or this builder is a part. Missing tuples are recorded and raised by finish(); an
    // allocation failure, an assertion or another job's exception still propagates as it did for the whole builder.
    // Parts balance when first values are spread over the rows; tuples sharing one first value all go to one part.
    std::vector<std::unique_ptr<InteractionBuilderInterface>> split_into(size_t parts) const
    {
        if (split_state != nullptr || parts < 2) {
            return {};
        }
        auto state = std::make_shared<SplitState>(parts);
        std::vector<std::unique_ptr<InteractionBuilderInterface>> jobs;
        jobs.reserve(parts);
        for (size_t part = 0; part < parts; ++part) {
            jobs.push_back(
                std::unique_ptr<MultiPermutationBuilder>(new MultiPermutationBuilder(dst_table_selector, part, state)));
        }
        return jobs;
    }

  private:
    static constexpr size_t MAX_PARTS = 8;
    static constexpr size_t CPUS_PER_PART = 8;
    static constexpr uint32_t MIN_ROWS_TO_SPLIT = 1U << 16;

    struct Failure {
        size_t permutation;
        uint32_t row;
        std::string message;
    };

    // Shared by the parts of a split builder. raise() throws what the whole builder would have thrown: the first
    // failure in processing order. A part's failures follow only from its own tuples' rows, and each part stops at
    // its first one, so that failure is the earliest of the parts' first failures.
    class SplitState {
      public:
        explicit SplitState(size_t num_parts)
            : num_parts(num_parts)
        {}
        void record(std::optional<Failure> failure)
        {
            std::lock_guard lock(mutex);
            if (failure.has_value() &&
                (!first_failure.has_value() || std::tie(failure->permutation, failure->row) <
                                                   std::tie(first_failure->permutation, first_failure->row))) {
                first_failure = std::move(failure);
            }
        }
        void raise()
        {
            std::lock_guard lock(mutex);
            if (first_failure.has_value()) {
                throw std::runtime_error(first_failure->message);
            }
        }
        const size_t num_parts;

      private:
        std::mutex mutex;
        std::optional<Failure> first_failure;
    };

    MultiPermutationBuilder(Column dst_table_selector, size_t part, std::shared_ptr<SplitState> split_state)
        : dst_table_selector(dst_table_selector)
        , part(part)
        , num_parts(split_state->num_parts)
        , split_state(std::move(split_state))
    {}

    bool in_part(const FF& first_value) const { return std::hash<FF>{}(first_value) % num_parts == part; }

    // Sets the destination selectors of this part's tuples, stopping at the first source row whose tuple has no
    // destination row left.
    std::optional<Failure> assign_rows(TraceContainer& trace)
    {
        // Index the destination columns.
        init(trace);

        // Set the destination selector for each permutation.
        std::optional<Failure> failure;
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            ((failure =
                  set_destination_selector<std::tuple_element_t<Is, std::tuple<PermutationSettings_...>>>(trace, Is))
                 .has_value() ||
             ...);
        }(std::index_sequence_for<PermutationSettings_...>{});

        // Standard permutation processing for each permutation should be done here if we were to
        // support a strict mode or re-introduce a non-empty PermutationBuilder::process().
        return failure;
    }

    template <typename PermutationSettings>
    std::optional<Failure> set_destination_selector(TraceContainer& trace, size_t permutation)
    {
        // Find each source tuple in the destination table, and set a 1 in the destination selector.
        std::optional<Failure> failure;
        trace.visit_column(PermutationSettings::SRC_SELECTOR, [&](uint32_t row, const FF&) {
            if (failure.has_value() ||
                (num_parts > 1 && !in_part(trace.get_column_or_shift(PermutationSettings::SRC_COLUMNS[0], row)))) {
                return;
            }
            auto src_values = get_multiple_as_array(trace, PermutationSettings::SRC_COLUMNS, row);
            auto index_it = row_idx.find(src_values);
            if (index_it == row_idx.end() || index_it->second.empty()) {
                failure = Failure{ permutation,
                                   row,
                                   "Failed setting selectors for " + std::string(PermutationSettings::NAME) +
                                       ". Could not find tuple in destination.\nSRC tuple (row " + std::to_string(row) +
                                       "): " + column_values_to_string(src_values, PermutationSettings::SRC_COLUMNS) };
                return;
            }
            // Get one of the available rows for the tuple.
            auto& possible_dst_rows = index_it->second;
            uint32_t dst_row = possible_dst_rows.back();
            trace.set(PermutationSettings::DST_SELECTOR, dst_row, 1);
            // We remove the used row from the list of possible rows.
            // Since this is a permutation, you can use a row only once.
            possible_dst_rows.pop_back();
            // NOTE: we don't remove the key src_values from the map if it's empty, because that triggers a rehash.
        });
        return failure;
    }

    void init(TraceContainer& trace)
    {
        // Index the destination columns.
        // For each row in the destination table, extract the value of columns (a, b, c, ...)
        // and add the row number to the index. See the comment on `row_idx` for more details.
        row_idx.reserve(trace.get_column_rows(dst_table_selector) / num_parts);
        trace.visit_column(dst_table_selector, [&](uint32_t row, const FF&) {
            if (num_parts > 1 && !in_part(trace.get_column_or_shift(DST_COLUMNS[0], row))) {
                return;
            }
            row_idx[get_multiple_as_array(trace, DST_COLUMNS, row)].push_back(row);
        });
    }

    // All permutations are expected to have the same destination columns.
    using REPRESENTATIVE_PERM = std::tuple_element_t<0, flat_tuple::tuple<PermutationSettings_...>>;
    static constexpr auto DST_COLUMNS = REPRESENTATIVE_PERM::DST_COLUMNS; // std::array.
    static constexpr size_t COLUMNS_PER_SET = DST_COLUMNS.size();
    static_assert(((DST_COLUMNS == PermutationSettings_::DST_COLUMNS) && ...));
    // The destination selectors, however, are not the same.
    // We need an extra "destination TABLE selector" which is the selector of the whole table.
    Column dst_table_selector;
    // This builder handles the tuples in_part(): all of them unless it is a part of a split builder.
    size_t part = 0;
    size_t num_parts = 1;
    std::shared_ptr<SplitState> split_state;

    // In a permutation (or lookup) you are trying to find a src tuple of values
    // (a, b, c, ...) in some destination table. That is, you want a row number in the destination table.
    // The following map contains (a, b, c, ...) -> [row_number_1, row_number_2, ...].
    // That is, you can efficiently find all the rows in the destination table that match the src tuple.
    // Keys are stored as owning value arrays (see get_multiple_as_array in interaction_builder.hpp for rationale).
    unordered_flat_map<std::array<FF, COLUMNS_PER_SET>, /*rows*/ std::vector<uint32_t>> row_idx;
};

} // namespace bb::avm2::tracegen
