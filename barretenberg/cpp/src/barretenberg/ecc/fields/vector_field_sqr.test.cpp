#include "barretenberg/ecc/fields/vector_field.hpp"

#include "barretenberg/ecc/curves/bn254/fq.hpp"
#include "barretenberg/ecc/curves/bn254/fr.hpp"

#include <array>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

namespace {

// sqr / self_sqr / sqr_for_affine run a dedicated squaring kernel under WASM SIMD; it must return the exact
// limbs of v * v, not just the same residue, on coarse [0, 2p) lanes and boundary values alike.
template <class VecT> void expect_sqr_limbs_match_self_mul()
{
    using F = typename VecT::Field;
    const F neg_one = -F::one();
    const auto two_p_minus_one = F::modulus + F::modulus - bb::numeric::uint256_t(1);
    const F raw_p(F::modulus.data[0], F::modulus.data[1], F::modulus.data[2], F::modulus.data[3]);
    const F raw_two_p_minus_one(
        two_p_minus_one.data[0], two_p_minus_one.data[1], two_p_minus_one.data[2], two_p_minus_one.data[3]);

    std::vector<VecT> inputs{ VecT(std::array<F, 5>{ F::zero(), F::one(), neg_one, raw_p, raw_two_p_minus_one }),
                              VecT(std::array<F, 5>{ raw_two_p_minus_one, raw_p, neg_one, F::one(), F::zero() }) };
    for (int trial = 0; trial < 256; ++trial) {
        std::array<F, 5> a;
        std::array<F, 5> b;
        for (size_t i = 0; i < 5; ++i) {
            a[i] = F::random_element();
            b[i] = F::random_element();
        }
        inputs.push_back(VecT(a) + VecT(b));
    }

    const auto expect_same_limbs = [](const std::array<F, 5>& expected, const std::array<F, 5>& got) {
        for (size_t i = 0; i < 5; ++i) {
            EXPECT_EQ(std::memcmp(expected[i].data, got[i].data, sizeof(expected[i].data)), 0) << "lane " << i;
        }
    };
    for (const VecT& v : inputs) {
        const auto expected = (v * v).to_array();
        expect_same_limbs(expected, v.sqr().to_array());
        VecT self = v;
        self.self_sqr();
        expect_same_limbs(expected, self.to_array());
        if constexpr (bb::vector_field_simd_enabled) {
            expect_same_limbs(expected, v.sqr_for_affine().to_array());
        }
    }
}

} // namespace

TEST(VectorFieldTest, SqrLimbsMatchSelfMul)
{
    expect_sqr_limbs_match_self_mul<bb::VectorField<bb::Bn254FrParams>>();
}

TEST(VectorFieldFqTest, SqrLimbsMatchSelfMul)
{
    expect_sqr_limbs_match_self_mul<bb::VectorField<bb::Bn254FqParams>>();
}
