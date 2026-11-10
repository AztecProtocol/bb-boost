#include "barretenberg/circuit_checker/circuit_checker.hpp"
#include "barretenberg/numeric/random/engine.hpp"
#include "barretenberg/numeric/uintx/uintx.hpp"
#include "barretenberg/stdlib/primitives/bigfield/bigfield.hpp"
#include "barretenberg/stdlib/primitives/circuit_builders/circuit_builders.hpp"
#include "barretenberg/stdlib/primitives/curves/bn254.hpp"
#include "barretenberg/stdlib/primitives/curves/secp256k1.hpp"
#include "barretenberg/stdlib/primitives/curves/secp256r1.hpp"
#include <array>
#include <gtest/gtest.h>
#include <vector>

// Pins the native shortcuts of bigfield against straightforward compositions of the public API: every constant,
// witness, gate and copy constraint must be exactly what the plain composition produces.

using namespace bb;

namespace {
auto& engine = numeric::get_debug_randomness();

template <typename Builder> void expect_same_field(const stdlib::field_t<Builder>& a, const stdlib::field_t<Builder>& b)
{
    EXPECT_EQ(a.get_context(), b.get_context());
    EXPECT_EQ(a.is_constant(), b.is_constant());
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(a.additive_constant.data[i], b.additive_constant.data[i]);
        EXPECT_EQ(a.multiplicative_constant.data[i], b.multiplicative_constant.data[i]);
    }
    EXPECT_EQ(a.get_origin_tag(), b.get_origin_tag());
}

template <typename Builder> void expect_same_circuit(const Builder& a, const Builder& b)
{
    ASSERT_EQ(a.get_variables().size(), b.get_variables().size());
    for (size_t i = 0; i < a.get_variables().size(); ++i) {
        for (size_t limb = 0; limb < 4; ++limb) {
            EXPECT_EQ(a.get_variables()[i].data[limb], b.get_variables()[i].data[limb]);
        }
    }
    EXPECT_EQ(a.blocks, b.blocks);
    EXPECT_EQ(a.real_variable_index, b.real_variable_index);
    EXPECT_EQ(static_cast<const CircuitBuilderBase<fr>&>(a), static_cast<const CircuitBuilderBase<fr>&>(b));
    EXPECT_EQ(a.failed(), b.failed());
}
} // namespace

template <typename BigField> class BigfieldCircuitIdentity : public testing::Test {
  public:
    using fq_ct = BigField;
    using Builder = typename fq_ct::field_ct::Builder;
    using field_ct = stdlib::field_t<Builder>;
    using witness_ct = stdlib::witness_t<Builder>;
    using Limb = typename fq_ct::Limb;
    static constexpr uint64_t L = fq_ct::NUM_LIMB_BITS;

    static void expect_same_bigfield(const fq_ct& a, const fq_ct& b)
    {
        EXPECT_EQ(a.get_context(), b.get_context());
        for (size_t i = 0; i < fq_ct::NUM_LIMBS; ++i) {
            expect_same_field(a.get_limb(i).element, b.get_limb(i).element);
            EXPECT_EQ(a.get_limb(i).maximum_value, b.get_limb(i).maximum_value);
        }
        expect_same_field(a.get_prime_basis_limb(), b.get_prime_basis_limb());
    }

    // The limbs `bigfield(ctx, value)` is specified to hold: constant limbs of the 68-bit slices of `value`, whose
    // maximum values come from the generic Limb constructor.
    static void expect_constant_layout(Builder* ctx, const fq_ct& actual, const uint256_t& value)
    {
        EXPECT_EQ(actual.get_context(), ctx);
        for (size_t i = 0; i < fq_ct::NUM_LIMBS; ++i) {
            const Limb reference(field_ct(ctx, fr(value.slice(i * L, (i + 1) * L))));
            expect_same_field(actual.get_limb(i).element, reference.element);
            EXPECT_EQ(actual.get_limb(i).maximum_value, reference.maximum_value);
        }
        expect_same_field(actual.get_prime_basis_limb(), field_ct(ctx, fr(value)));
    }

    static void test_constant_constructors()
    {
        Builder builder;
        for (Builder* ctx : std::array<Builder*, 2>{ nullptr, &builder }) {
            expect_constant_layout(ctx, fq_ct(ctx), uint256_t(0));
            // fr(0) is the all-zero representation, which is what the zero constructor writes.
            EXPECT_EQ(fr(0).data[0] | fr(0).data[1] | fr(0).data[2] | fr(0).data[3], 0ULL);

            std::vector<uint256_t> values{ 0, 1, (uint256_t(1) << L) - 1, uint256_t(1) << L, fq_ct::modulus - 1 };
            for (size_t i = 0; i < 8; ++i) {
                values.emplace_back(engine.get_random_uint256() % fq_ct::modulus);
            }
            for (const auto& value : values) {
                expect_constant_layout(ctx, fq_ct(ctx, value), value);
            }
        }
        expect_constant_layout(nullptr, fq_ct::one(), uint256_t(1));
        expect_constant_layout(nullptr, fq_ct::zero(), uint256_t(0));
        EXPECT_EQ(builder.get_num_variables(), Builder().get_num_variables());
    }
};

using CircuitTypes = testing::Types<typename stdlib::bn254<UltraCircuitBuilder>::BaseField,
                                    typename stdlib::secp256k1<UltraCircuitBuilder>::BaseField,
                                    typename stdlib::secp256k1<UltraCircuitBuilder>::ScalarField,
                                    typename stdlib::secp256r1<UltraCircuitBuilder>::BaseField,
                                    typename stdlib::secp256r1<UltraCircuitBuilder>::ScalarField>;
TYPED_TEST_SUITE(BigfieldCircuitIdentity, CircuitTypes);

TYPED_TEST(BigfieldCircuitIdentity, ConstantConstructors)
{
    TestFixture::test_constant_constructors();
}
