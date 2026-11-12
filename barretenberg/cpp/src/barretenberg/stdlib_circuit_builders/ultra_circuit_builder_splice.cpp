#include "ultra_circuit_builder.hpp"

#include "barretenberg/common/assert.hpp"
#include "barretenberg/common/thread.hpp"
#include "barretenberg/numeric/uint256/uint256.hpp"
#include "bounded_range_sort.hpp"

#include <algorithm>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bb {

namespace {

// Each row-copy task covers at least this many rows, ending on a tile boundary so that tasks share no tile.
constexpr size_t MIN_ROWS_PER_TASK = size_t{ 1 } << 12;
// The values of new variables are copied in chunks of at least this many: a chunk outweighs a parallel dispatch.
constexpr size_t MIN_VARIABLES_PER_CHUNK = size_t{ 1 } << 15;

// Consecutive source positions copied to consecutive destination positions.
struct Run {
    size_t source;
    size_t destination;
    size_t count;
};

// The runs of [begin, end) outside the half-open intervals `dropped`, numbered consecutively from destination 0.
std::vector<Run> runs_outside(const size_t begin, const size_t end, std::vector<std::pair<size_t, size_t>> dropped)
{
    std::ranges::sort(dropped);
    std::vector<Run> runs;
    size_t next = begin;
    size_t destination = 0;
    for (const auto& [lo, hi] : dropped) {
        if (lo > next) {
            runs.push_back({ .source = next, .destination = destination, .count = lo - next });
            destination += lo - next;
        }
        next = std::max(next, hi);
    }
    if (end > next) {
        runs.push_back({ .source = next, .destination = destination, .count = end - next });
    }
    return runs;
}

size_t total_count(const std::vector<Run>& runs)
{
    return runs.empty() ? 0 : runs.back().destination + runs.back().count;
}

// Calls body(source, destination, count) for the parts of `runs` covering destinations [begin, end), in order.
template <typename Body>
void for_each_run_part(const std::vector<Run>& runs, const size_t begin, const size_t end, const Body& body)
{
    if (begin >= end) {
        return;
    }
    // The first run starts at destination 0, so the run holding `begin` precedes the first run starting after it.
    auto it = std::ranges::upper_bound(runs, begin, {}, &Run::destination);
    for (--it; it != runs.end() && it->destination < end; ++it) {
        const size_t lo = std::max(begin, it->destination);
        const size_t hi = std::min(end, it->destination + it->count);
        body(it->source + (lo - it->destination), lo, hi - lo);
    }
}

bool has_bit(const std::vector<uint64_t>& bits, const uint32_t index)
{
    return ((bits[index >> 6] >> (index & 63)) & 1) != 0;
}

} // namespace

template <typename ExecutionTrace>
SpliceResult UltraCircuitBuilder_<ExecutionTrace>::splice(const UltraCircuitBuilder_& source,
                                                          const std::span<const uint32_t> seed_indices)
    requires(!ExecutionTrace::IS_MEGA_ARITHMETIZATION)
{
    const auto refuse = [](const SpliceRefusal refusal) { return SpliceResult{ .refusal = refusal }; };

    if (!source.splice_log_.has_value() || source.splice_log_->num_seeds != seed_indices.size() ||
        splice_log_.has_value() || this->circuit_finalized || source.circuit_finalized || this->is_write_vk_mode() ||
        source.is_write_vk_mode()) {
        return refuse(SpliceRefusal::BuilderState);
    }
    const SpliceLog& log = *source.splice_log_;
    if (this->failed() || source.failed()) {
        return refuse(SpliceRefusal::Failed);
    }
    if (!source.public_inputs().empty() || source.blocks.pub_inputs.size() != 0 || source.public_inputs_finalized()) {
        return refuse(SpliceRefusal::PublicInput);
    }
    if (source.blocks.elliptic.size() != 0 || source.blocks.lookup.size() != 0) {
        return refuse(SpliceRefusal::RowForm);
    }
    const PairingPointsTagging& source_tagging = source.pairing_points_tagging;
    if (!source.lookup_tables.empty() || !source.ipa_proof.empty() || source.pending_ipa_proof.valid() ||
        !source.variable_names.empty() || source_tagging.has_public_pairing_points() ||
        (source_tagging.num_tags() != 0 && this->pairing_points_tagging.has_public_pairing_points())) {
        return refuse(SpliceRefusal::UnknownForm);
    }
    // Every source tag belongs to a recorded range list, and every source constant but zero was recorded. The tag
    // permutation fixes DEFAULT_TAG and swaps the two tags of each recorded range list, and holds nothing else.
    const std::unordered_map<uint32_t, uint32_t>& source_tau = source.tau();
    const auto tau_maps = [&source_tau](const uint32_t from, const uint32_t to) {
        const auto it = source_tau.find(from);
        return it != source_tau.end() && it->second == to;
    };
    if (source.get_current_tag() != 2 * log.range_lists.size() || source.range_lists.size() != log.range_lists.size() ||
        source.constant_variable_indices.size() != log.constants.size() + 1 ||
        source_tau.size() != 1 + (2 * log.range_lists.size()) || !tau_maps(DEFAULT_TAG, DEFAULT_TAG)) {
        return refuse(SpliceRefusal::UnknownForm);
    }
    std::vector<const RangeList*> source_lists(log.range_lists.size());
    for (size_t l = 0; l < log.range_lists.size(); ++l) {
        const auto& recorded = log.range_lists[l];
        const auto it = source.range_lists.find(recorded.target_range);
        if (it == source.range_lists.end() || it->second.range_tag != recorded.range_tag ||
            it->second.tau_tag != recorded.range_tag + 1 ||
            it->second.variable_indices.size() < recorded.staircase_size ||
            !tau_maps(recorded.range_tag, recorded.range_tag + 1) ||
            !tau_maps(recorded.range_tag + 1, recorded.range_tag)) {
            return refuse(SpliceRefusal::UnknownForm);
        }
        source_lists[l] = &it->second;
    }

    const size_t num_seeds = seed_indices.size();
    const std::vector<FF>& source_values = source.get_variables();
    // The real variable here of each input, which must be untagged and in a copy cycle of its own.
    std::vector<uint32_t> input_classes;
    input_classes.reserve(num_seeds + 1);
    for (size_t i = 0; i < num_seeds; ++i) {
        const uint32_t index = seed_indices[i];
        BB_ASSERT_LT(index, this->get_num_variables(), "splice: seed index out of range");
        if (source_values[i] != this->get_variable(index)) {
            return refuse(SpliceRefusal::InputValue);
        }
        const uint32_t real_index = this->real_variable_index[index];
        if (this->real_variable_tags[real_index] != DEFAULT_TAG) {
            return refuse(SpliceRefusal::InputTag);
        }
        input_classes.push_back(real_index);
    }
    std::vector<uint32_t> sorted_seeds(seed_indices.begin(), seed_indices.end());
    std::ranges::sort(sorted_seeds);
    // The source compares variable indices, so no seed may coincide with another variable the source shares.
    const auto is_seed = [&](const uint32_t index) { return std::ranges::binary_search(sorted_seeds, index); };

    const size_t num_source_variables = source.get_num_variables();
    std::vector<uint32_t> index_map(num_source_variables);
    std::ranges::copy(seed_indices, index_map.begin());
    index_map[source.zero_idx()] = this->zero_idx();
    // Source variables that are this builder's: the zero, then constants cached here.
    std::vector<uint32_t> shared_variables{ source.zero_idx() };

    // A source constant whose value is cached here is the cached variable, without its fixing row or used-witness
    // entry.
    std::vector<std::pair<size_t, size_t>> dropped_variables;
    std::vector<std::pair<size_t, size_t>> dropped_rows;
    std::vector<size_t> dropped_used_witnesses;
    std::vector<uint8_t> constant_cached(log.constants.size(), 0);
    for (size_t c = 0; c < log.constants.size(); ++c) {
        const auto& constant = log.constants[c];
        const auto it = constant_variable_indices.find(source_values[constant.variable]);
        if (it == constant_variable_indices.end()) {
            continue;
        }
        constant_cached[c] = 1;
        index_map[constant.variable] = it->second;
        shared_variables.push_back(constant.variable);
        dropped_variables.emplace_back(constant.variable, constant.variable + 1);
        dropped_rows.emplace_back(constant.arithmetic_row, constant.arithmetic_row + 1);
        dropped_used_witnesses.push_back(constant.used_witness);
    }
    for (const uint32_t shared : shared_variables) {
        if (is_seed(index_map[shared])) {
            return refuse(SpliceRefusal::InputCycle);
        }
    }

    // A source range list whose range has a list here is that list: its staircase is this list's staircase, without
    // its rows.
    std::vector<RangeList*> target_lists(log.range_lists.size(), nullptr);
    for (size_t l = 0; l < log.range_lists.size(); ++l) {
        const auto& recorded = log.range_lists[l];
        const auto it = range_lists.find(recorded.target_range);
        if (it == range_lists.end()) {
            continue;
        }
        const std::vector<uint32_t>& target_staircase = it->second.variable_indices;
        const std::vector<uint32_t>& source_staircase = source_lists[l]->variable_indices;
        const size_t staircase_size = recorded.staircase_size;
        if (target_staircase.size() < staircase_size) {
            return refuse(SpliceRefusal::UnknownForm);
        }
        const size_t first = source_staircase[0];
        for (size_t q = 0; q < staircase_size; ++q) {
            if (source_staircase[q] != first + q) {
                return refuse(SpliceRefusal::UnknownForm);
            }
            if (is_seed(target_staircase[q])) {
                return refuse(SpliceRefusal::InputCycle);
            }
            index_map[first + q] = target_staircase[q];
        }
        target_lists[l] = &it->second;
        dropped_variables.emplace_back(first, first + staircase_size);
        dropped_rows.emplace_back(recorded.arithmetic_row,
                                  recorded.arithmetic_row + ((staircase_size + NUM_WIRES - 1) / NUM_WIRES));
    }

    // A shared variable in a source class that a tag reaches decides range constraints, which read the class tag: it
    // has to be untagged here and in a copy cycle of its own among the inputs, as it is in the source.
    std::vector<uint64_t> tagged_classes((num_source_variables + 63) / 64, 0);
    for (const auto& event : log.events) {
        if (event.kind == SpliceLog::Event::Kind::Tag) {
            const uint32_t source_class = source.real_variable_index[event.a];
            tagged_classes[source_class >> 6] |= uint64_t{ 1 } << (source_class & 63);
        }
    }
    for (const uint32_t shared : shared_variables) {
        if (!has_bit(tagged_classes, source.real_variable_index[shared])) {
            continue;
        }
        const uint32_t real_index = this->real_variable_index[index_map[shared]];
        if (this->real_variable_tags[real_index] != DEFAULT_TAG) {
            return refuse(SpliceRefusal::InputTag);
        }
        input_classes.push_back(real_index);
    }
    std::ranges::sort(input_classes);
    if (std::ranges::adjacent_find(input_classes) != input_classes.end()) {
        return refuse(SpliceRefusal::InputCycle);
    }

    // Every precondition holds: from here on the builder changes.

    // The tags of a source list are its list's here, or a fresh pair, allocated in the order the lists were created.
    std::vector<uint32_t> tag_map(source.get_current_tag() + 1, DEFAULT_TAG);
    for (size_t l = 0; l < log.range_lists.size(); ++l) {
        const uint32_t range_tag = log.range_lists[l].range_tag;
        if (target_lists[l] != nullptr) {
            tag_map[range_tag] = target_lists[l]->range_tag;
            tag_map[range_tag + 1] = target_lists[l]->tau_tag;
        } else {
            const uint32_t new_range_tag = get_new_tag();
            const uint32_t new_tau_tag = get_new_tag();
            set_tau_transposition(new_range_tag, new_tau_tag);
            tag_map[range_tag] = new_range_tag;
            tag_map[range_tag + 1] = new_tau_tag;
        }
    }

    // The kept source variables, in source order, become new variables here.
    const std::vector<Run> variable_runs =
        runs_outside(log.num_prefix_variables, num_source_variables, std::move(dropped_variables));
    const size_t num_new_variables = total_count(variable_runs);
    const size_t first_new_variable = this->get_num_variables();
    this->add_tagged_variables(num_new_variables, DEFAULT_TAG, [&](const std::span<FF> values) {
        range_sort::Chunks::of_min_size(num_new_variables, MIN_VARIABLES_PER_CHUNK)
            .run([&](size_t /*chunk*/, const size_t begin, const size_t end) {
                for_each_run_part(
                    variable_runs, begin, end, [&](const size_t from, const size_t to, const size_t count) {
                        std::copy_n(source_values.data() + from, count, values.data() + to);
                        std::iota(index_map.data() + from,
                                  index_map.data() + from + count,
                                  static_cast<uint32_t>(first_new_variable + to));
                    });
            });
    });

    // The kept source rows of each block are appended to it, wires through the index map, copied by tasks over
    // contiguous row ranges of all blocks.
    struct BlockCopy {
        ExecutionTraceBlockStorage<FF>* target;
        const ExecutionTraceBlockStorage<FF>* source;
        std::vector<Run> runs;
        size_t first_row;
        // The target's id of each of the source's selector value ids (blocks dedup their selector values).
        typename ExecutionTraceBlockStorage<FF>::ValueIdMaps value_ids;
    };
    const size_t memory_offset = blocks.memory.size();
    const size_t rom_offset = rom_ram_logic.rom_arrays.size();
    auto target_blocks = blocks.get();
    auto source_blocks = source.blocks.get();
    std::vector<BlockCopy> copies;
    size_t num_rows = 0;
    size_t num_dropped_rows = 0;
    for (size_t b = 0; b < target_blocks.size(); ++b) {
        const ExecutionTraceBlockStorage<FF>& from = source_blocks[b];
        const bool is_arithmetic = &from == &source.blocks.arithmetic;
        const size_t begin = is_arithmetic ? log.num_prefix_arithmetic_rows : 0;
        std::vector<Run> runs = runs_outside(
            begin, from.size(), is_arithmetic ? std::move(dropped_rows) : std::vector<std::pair<size_t, size_t>>{});
        const size_t count = total_count(runs);
        num_dropped_rows += from.size() - begin - count;
        if (count == 0) {
            continue;
        }
        num_rows += count;
        copies.push_back({ .target = &target_blocks[b], .source = &from, .runs = std::move(runs), .first_row = 0 });
    }
    const size_t rows_per_task = std::max(MIN_ROWS_PER_TASK, (num_rows + get_num_cpus() - 1) / get_num_cpus());
    struct Task {
        size_t copy;
        size_t begin;
        size_t end;
    };
    std::vector<Task> tasks;
    for (size_t c = 0; c < copies.size(); ++c) {
        BlockCopy& copy = copies[c];
        const size_t count = total_count(copy.runs);
        copy.first_row = copy.target->grow_rows_for_overwrite(count);
        // Serial: interning may add values to the target's tables, which the concurrent copies below only read.
        copy.value_ids = copy.target->intern_values_of(*copy.source);
        for (size_t begin = 0; begin < count;) {
            const size_t aligned_end = (copy.first_row + begin + rows_per_task + GATE_TILE_MASK) & ~GATE_TILE_MASK;
            const size_t end = std::min(count, aligned_end - copy.first_row);
            tasks.push_back({ .copy = c, .begin = begin, .end = end });
            begin = end;
        }
    }
    const auto run_task = [&](const size_t t) {
        const Task& task = tasks[t];
        const BlockCopy& copy = copies[task.copy];
        for_each_run_part(copy.runs, task.begin, task.end, [&](const size_t from, const size_t to, const size_t count) {
            copy.target->copy_translated_rows(
                *copy.source, from, copy.first_row + to, count, index_map.data(), copy.value_ids);
        });
    };
    if (tasks.size() == 1) {
        run_task(0);
    } else if (!tasks.empty()) {
        parallel_for(tasks.size(), run_task);
    }
    this->increment_num_gates(source.num_gates() - log.num_prefix_gates - num_dropped_rows);
    // A ROM-LogUp row carries its array id in q_c; source arrays follow the arrays here.
    if (rom_offset != 0) {
        for (const uint32_t gate : source.rom_logup_records) {
            const uint64_t rom_id = uint256_t(source.blocks.memory.q_c()[gate]).data[0];
            blocks.memory.q_c().set(memory_offset + gate, FF(rom_id + rom_offset));
        }
    }

    for (size_t l = 0; l < log.range_lists.size(); ++l) {
        const std::vector<uint32_t>& from = source_lists[l]->variable_indices;
        if (target_lists[l] != nullptr) {
            std::vector<uint32_t>& to = target_lists[l]->variable_indices;
            const size_t staircase_size = log.range_lists[l].staircase_size;
            to.reserve(to.size() + from.size() - staircase_size);
            for (size_t q = staircase_size; q < from.size(); ++q) {
                to.push_back(index_map[from[q]]);
            }
        } else {
            const uint32_t range_tag = log.range_lists[l].range_tag;
            RangeList list{ .target_range = log.range_lists[l].target_range,
                            .range_tag = tag_map[range_tag],
                            .tau_tag = tag_map[range_tag + 1],
                            .variable_indices = {} };
            list.variable_indices.reserve(from.size());
            for (const uint32_t variable : from) {
                list.variable_indices.push_back(index_map[variable]);
            }
            range_lists.emplace(list.target_range, std::move(list));
        }
    }

    // Copy cycles and tags, replayed in source order. A tag write equals assign_tag's: the inputs it can reach are
    // untagged here.
    const std::string assert_equal_message = "assert_equal";
    for (const auto& event : log.events) {
        if (event.kind == SpliceLog::Event::Kind::AssertEqual) {
            CircuitBuilderBase<FF>::assert_equal(index_map[event.a], index_map[event.b], assert_equal_message);
        } else {
            this->real_variable_tags[this->real_variable_index[index_map[event.a]]] = tag_map[event.b];
        }
    }

    for (size_t c = 0; c < log.constants.size(); ++c) {
        if (constant_cached[c] == 0) {
            const uint32_t variable = log.constants[c].variable;
            constant_variable_indices.emplace(source_values[variable], index_map[variable]);
        }
    }

    cached_partial_non_native_field_multiplications.reserve(
        cached_partial_non_native_field_multiplications.size() +
        source.cached_partial_non_native_field_multiplications.size());
    for (auto entry : source.cached_partial_non_native_field_multiplications) {
        for (size_t limb = 0; limb < 4; ++limb) {
            entry.a[limb] = index_map[entry.a[limb]];
            entry.b[limb] = index_map[entry.b[limb]];
        }
        entry.lo_0 = index_map[entry.lo_0];
        entry.hi_0 = index_map[entry.hi_0];
        entry.hi_1 = index_map[entry.hi_1];
        cached_partial_non_native_field_multiplications.push_back(entry);
    }

    const auto map_cell = [&](const uint32_t cell) {
        return cell == UNINITIALIZED_MEMORY_RECORD ? cell : index_map[cell];
    };
    for (RomTranscript array : source.rom_ram_logic.rom_arrays) {
        for (auto& cell : array.state) {
            cell = { map_cell(cell[0]), map_cell(cell[1]) };
        }
        for (auto& record : array.records) {
            record.index_witness = index_map[record.index_witness];
            record.value_column1_witness = index_map[record.value_column1_witness];
            record.value_column2_witness = index_map[record.value_column2_witness];
            record.record_witness = index_map[record.record_witness];
            record.gate_index += memory_offset;
        }
        rom_ram_logic.rom_arrays.push_back(std::move(array));
    }
    for (RamTranscript array : source.rom_ram_logic.ram_arrays) {
        for (auto& cell : array.state) {
            cell = map_cell(cell);
        }
        for (auto& record : array.records) {
            record.index_witness = index_map[record.index_witness];
            record.timestamp_witness = index_map[record.timestamp_witness];
            record.value_witness = index_map[record.value_witness];
            record.record_witness = index_map[record.record_witness];
            record.gate_index += memory_offset;
        }
        rom_ram_logic.ram_arrays.push_back(std::move(array));
    }
    const auto append_gates = [&](const std::vector<uint32_t>& from, std::vector<uint32_t>& to) {
        for (const uint32_t gate : from) {
            to.push_back(static_cast<uint32_t>(gate + memory_offset));
        }
    };
    append_gates(source.memory_read_records, memory_read_records);
    append_gates(source.memory_write_records, memory_write_records);
    append_gates(source.rom_logup_records, rom_logup_records);

    const uint32_t pairing_tag_offset = this->pairing_points_tagging.append_shifted(source_tagging);

    // Constants were recorded in creation order, so their used-witness positions ascend.
    size_t next_dropped = 0;
    for (size_t k = log.num_prefix_used_witnesses; k < source.used_witnesses.size(); ++k) {
        if (next_dropped < dropped_used_witnesses.size() && dropped_used_witnesses[next_dropped] == k) {
            ++next_dropped;
            continue;
        }
        used_witnesses.push_back(index_map[source.used_witnesses[k]]);
    }
    for (const uint32_t variable : source.finalize_witnesses) {
        finalize_witnesses.insert(index_map[variable]);
    }

    SpliceResult result;
    result.index_map = std::move(index_map);
    result.pairing_tag_offset = pairing_tag_offset;
    return result;
}

template SpliceResult UltraCircuitBuilder_<UltraExecutionTraceBlocks>::splice(
    const UltraCircuitBuilder_<UltraExecutionTraceBlocks>& source, std::span<const uint32_t> seed_indices);

} // namespace bb
