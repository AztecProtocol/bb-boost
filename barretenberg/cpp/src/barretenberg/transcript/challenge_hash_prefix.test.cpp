#include "barretenberg/ecc/curves/bn254/g1.hpp"
#include "barretenberg/numeric/random/engine.hpp"
#include "barretenberg/transcript/transcript.hpp"
#include <array>
#include <cstring>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace bb {
namespace {
auto& engine = numeric::get_debug_randomness();

bool same_limbs(const fr& a, const fr& b)
{
    return std::memcmp(a.data, b.data, sizeof(a.data)) == 0;
}

std::vector<fr> random_frs(size_t count)
{
    std::vector<fr> result(count);
    for (auto& element : result) {
        element = fr::random_element(&engine);
    }
    return result;
}
} // namespace

TEST(ChallengeHashPrefix, HashesOnlyTheInputItWasAbsorbedFor)
{
    const std::vector<fr> input = random_frs(10);
    const std::vector<fr> prefix(input.begin(), input.begin() + 4);
    const ChallengeHashPrefix absorbed(prefix, input.size());

    const auto hash = absorbed.hash_if_extended_by(input);
    ASSERT_TRUE(hash.has_value());
    EXPECT_TRUE(same_limbs(*hash, NativePoseidon2::hash(input)));

    std::vector<fr> longer = input;
    longer.push_back(fr(1));
    EXPECT_FALSE(absorbed.hash_if_extended_by(longer).has_value());

    std::vector<fr> changed = input;
    changed[3] += fr(1);
    EXPECT_FALSE(absorbed.hash_if_extended_by(changed).has_value());

    const ChallengeHashPrefix empty_prefix({}, input.size());
    const auto unprefixed_hash = empty_prefix.hash_if_extended_by(input);
    ASSERT_TRUE(unprefixed_hash.has_value());
    EXPECT_TRUE(same_limbs(*unprefixed_hash, NativePoseidon2::hash(input)));
}

// A transcript offered a matching prefix, or one absorbed for the wrong input length, produces the same challenges
// and proof as one that is offered nothing.
TEST(ChallengeHashPrefix, TranscriptChallengesDoNotChange)
{
    const g1::affine_element commitment = g1::affine_element::random_element(&engine);
    constexpr size_t COMMITMENT_FIELDS = FrCodec::calc_num_fields<g1::affine_element>();

    for (const int offset : { 0, 1, -1 }) {
        SCOPED_TRACE("length offset " + std::to_string(offset));
        NativeTranscript plain;
        NativeTranscript hinted;
        const auto both = [&](const auto& action) {
            action(plain);
            action(hinted);
        };

        // A first round, so that the hinted challenge also hashes a previous challenge.
        const std::vector<fr> first_round = random_frs(5);
        both([&](NativeTranscript& t) {
            for (size_t i = 0; i < first_round.size(); ++i) {
                t.add_to_hash_buffer("a_" + std::to_string(i), first_round[i]);
            }
        });
        EXPECT_TRUE(same_limbs(plain.get_challenge<fr>("alpha"), hinted.get_challenge<fr>("alpha")));

        const std::vector<fr> known = random_frs(7);
        both([&](NativeTranscript& t) {
            for (size_t i = 0; i < known.size(); ++i) {
                t.add_to_hash_buffer("b_" + std::to_string(i), known[i]);
            }
        });
        std::vector<fr> prefix = hinted.next_challenge_input_so_far();
        const size_t input_length = prefix.size() + 2 * COMMITMENT_FIELDS + static_cast<size_t>(offset);
        hinted.set_next_challenge_prefix(ChallengeHashPrefix(std::move(prefix), input_length));
        both([&](NativeTranscript& t) {
            t.send_to_verifier("c_0", commitment);
            t.send_to_verifier("c_1", commitment);
        });

        const auto plain_beta_gamma = plain.get_challenges<fr>(std::array<std::string, 2>{ "beta", "gamma" });
        const auto hinted_beta_gamma = hinted.get_challenges<fr>(std::array<std::string, 2>{ "beta", "gamma" });
        EXPECT_TRUE(same_limbs(plain_beta_gamma[0], hinted_beta_gamma[0]));
        EXPECT_TRUE(same_limbs(plain_beta_gamma[1], hinted_beta_gamma[1]));

        both([&](NativeTranscript& t) { t.send_to_verifier("d", fr(42)); });
        EXPECT_TRUE(same_limbs(plain.get_challenge<fr>("zeta"), hinted.get_challenge<fr>("zeta")));
        EXPECT_EQ(plain.export_proof(), hinted.export_proof());
    }
}

} // namespace bb
