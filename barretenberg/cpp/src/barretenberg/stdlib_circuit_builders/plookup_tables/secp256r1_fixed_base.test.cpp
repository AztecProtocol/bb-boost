#include "./secp256r1_fixed_base.hpp"

#include "barretenberg/common/thread.hpp"
#include "barretenberg/ecc/groups/precomputed_generators.hpp"
#include "barretenberg/ecc/groups/precomputed_generators_secp256r1_impl.hpp"

#include <gtest/gtest.h>

using namespace bb::plookup::secp256r1_fixed_base;

namespace {
using G1 = table::G1;
using Element = table::Element;
using AffineElement = table::AffineElement;
using Fr = G1::Fr;
} // namespace

// Every entry is k · 2^(bit_offset(w)) · G + 2^w · H, computed here by scalar multiplication, and the total offset is
// (2^NUM_WINDOWS - 1) · H. Reads run from several threads at once, as the builders' lookups do.
TEST(Secp256r1FixedBaseTable, EntriesAreTheWindowMultiples)
{
    constexpr AffineElement H = bb::get_precomputed_generators<G1, "biggroup table offset generator", 1UL, 0UL>()[0];
    std::array<size_t, table::NUM_WINDOWS> bit_offset{};
    for (size_t w = 1; w < table::NUM_WINDOWS; ++w) {
        bit_offset[w] = bit_offset[w - 1] + table::window_bits(w - 1);
    }
    std::array<bool, table::NUM_WINDOWS> window_ok{};
    bb::parallel_for(table::NUM_WINDOWS, [&](size_t w) {
        const Element offset = Element(H) * Fr(uint256_t(1) << w);
        bool ok = true;
        for (size_t k = 0; k < table::table_size(w); ++k) {
            const Element expected = Element(G1::affine_one) * Fr(uint256_t(k) << bit_offset[w]) + offset;
            ok = ok && table::get_entry(w, k) == AffineElement(expected);
        }
        window_ok[w] = ok;
    });
    for (size_t w = 0; w < table::NUM_WINDOWS; ++w) {
        EXPECT_TRUE(window_ok[w]) << "window " << w;
    }
    const Fr total = Fr((uint256_t(1) << table::NUM_WINDOWS) - 1);
    EXPECT_EQ(table::total_offset(), AffineElement(Element(H) * total));
}
