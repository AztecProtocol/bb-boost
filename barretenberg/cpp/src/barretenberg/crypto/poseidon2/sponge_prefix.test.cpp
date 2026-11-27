#include "barretenberg/crypto/poseidon2/poseidon2_params.hpp"
#include "barretenberg/numeric/random/engine.hpp"
#include "poseidon2.hpp"
#include <cstring>
#include <gtest/gtest.h>
#include <span>
#include <vector>

using namespace bb;

namespace {
auto& engine = numeric::get_debug_randomness();

using Sponge = crypto::Poseidon2<crypto::Poseidon2Bn254ScalarFieldParams>::Sponge;

bool same_limbs(const fr& a, const fr& b)
{
    return std::memcmp(a.data, b.data, sizeof(a.data)) == 0;
}
} // namespace

// Every split of every input length around the rate (3), including an empty input, an empty prefix and an empty rest,
// must give hash_internal's result limb for limb.
TEST(Poseidon2SpongePrefix, FinishMatchesHashInternalForEverySplit)
{
    for (size_t length = 0; length <= 13; ++length) {
        std::vector<fr> input(length);
        for (auto& element : input) {
            element = fr::random_element(&engine);
        }
        const fr expected = Sponge::hash_internal(input);
        const std::span<const fr> all(input);
        for (size_t split = 0; split <= length; ++split) {
            SCOPED_TRACE("length " + std::to_string(length) + " split " + std::to_string(split));
            const Sponge::AbsorbedPrefix prefix(length, all.first(split));
            EXPECT_EQ(prefix.remaining_length(), length - split);
            EXPECT_TRUE(same_limbs(prefix.finish(all.subspan(split)), expected));
            // finish works on a copy of the sponge, so it can be called again with the same result.
            EXPECT_TRUE(same_limbs(prefix.finish(all.subspan(split)), expected));
        }
    }
}
