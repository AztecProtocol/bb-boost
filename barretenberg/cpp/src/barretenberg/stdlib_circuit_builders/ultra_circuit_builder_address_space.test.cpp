#include "barretenberg/common/address_space_reserve.hpp"
#include "barretenberg/stdlib_circuit_builders/mega_circuit_builder.hpp"
#include "barretenberg/stdlib_circuit_builders/ultra_circuit_builder.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace bb {
namespace {

using Builder = UltraCircuitBuilder;
using Reservation = Builder::VariablesReservation;

// The capacities of the per-variable vectors a test can reach (next_var_index and prev_var_index are private).
template <typename B> std::array<size_t, 3> capacities(const B& builder)
{
    return { builder.get_variables().capacity(),
             builder.real_variable_index.capacity(),
             builder.real_variable_tags.capacity() };
}

// Checks what the constructor reserved: an attempt exactly when this process allows one, and the reservation or else
// at least `hint` entries in each reachable per-variable vector.
void check_reservation(const Builder& builder, size_t hint)
{
    const Reservation state = builder.variables_reservation();
    EXPECT_EQ(state != Reservation::NotAttempted, address_space_reservation_allowed());
    for (const size_t capacity : capacities(builder)) {
        if (state == Reservation::Reserved) {
            EXPECT_GE(capacity, Builder::VARIABLES_ADDRESS_SPACE_ENTRIES);
        } else {
            EXPECT_GE(capacity, hint);
            EXPECT_LT(capacity, Builder::VARIABLES_ADDRESS_SPACE_ENTRIES);
        }
    }
}

// Adds variables until `builder` holds `num_variables`, then checks that the variables `values` were given at
// construction still hold them and, where the builder holds the reservation, that the storage never moved.
template <typename B> void grow_and_check(B& builder, const std::vector<fr>& values, size_t num_variables)
{
    const bool reserved = builder.variables_reservation() == Reservation::Reserved;
    const fr* const storage = builder.get_variables().data();
    while (builder.get_num_variables() < num_variables) {
        builder.add_variable(fr(builder.get_num_variables()));
    }
    for (size_t i = 0; i < values.size(); ++i) {
        EXPECT_EQ(builder.get_variable(static_cast<uint32_t>(i)), values[i]);
    }
    if (reserved) {
        EXPECT_EQ(builder.get_variables().data(), storage);
    }
}

// 1000 witnesses, whose hint is max(4 * 1000, 65536) variables.
std::vector<fr> acir_witness()
{
    std::vector<fr> witness(1000);
    for (size_t i = 0; i < witness.size(); ++i) {
        witness[i] = fr(7 * i + 1);
    }
    return witness;
}

TEST(UltraCircuitBuilderAddressSpace, AcirBuilderGrowsPastItsHint)
{
    const std::vector<fr> witness = acir_witness();
    Builder builder(witness, /*public_inputs=*/std::vector<uint32_t>{}, /*is_write_vk_mode=*/false);
    check_reservation(builder, 65536);
    grow_and_check(builder, witness, (size_t{ 1 } << 17) + 1);
}

// A Mega circuit from ACIR keeps the witness-count hint whatever this process allows.
TEST(UltraCircuitBuilderAddressSpace, MegaAcirBuilderKeepsItsHint)
{
    const std::vector<fr> witness = acir_witness();
    MegaCircuitBuilder builder(
        std::make_shared<ECCOpQueue>(), witness, /*public_inputs=*/std::vector<uint32_t>{}, /*is_write_vk_mode=*/false);
    EXPECT_EQ(builder.variables_reservation(), Reservation::NotAttempted);
    for (const size_t capacity : capacities(builder)) {
        EXPECT_GE(capacity, 65536U);
        EXPECT_LT(capacity, Builder::VARIABLES_ADDRESS_SPACE_ENTRIES);
    }
    grow_and_check(builder, witness, (size_t{ 1 } << 17) + 1);
}

// The reservation state is not part of a builder's identity: the call changes only `b`'s capacity and state.
TEST(UltraCircuitBuilderAddressSpace, ReservationStateIsNotCompared)
{
    const Builder a;
    Builder b;
    b.reserve_variables_address_space(1);
    EXPECT_EQ(a.variables_reservation() != b.variables_reservation(), address_space_reservation_allowed());
    EXPECT_TRUE(static_cast<const CircuitBuilderBase<fr>&>(a) == static_cast<const CircuitBuilderBase<fr>&>(b));
}

TEST(UltraCircuitBuilderAddressSpace, SeedBuilderGrowsPastItsHint)
{
    const std::vector<fr> seeds{ fr(3), fr(5), fr(7) };
    Builder builder(SpliceRecording{}, seeds);
    // The hint for three seeds is four variables.
    check_reservation(builder, 4);
    grow_and_check(builder, seeds, 4096);
}

} // namespace
} // namespace bb
