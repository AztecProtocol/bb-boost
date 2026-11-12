#include "barretenberg/stdlib/primitives/builder_binding.hpp"
#include "barretenberg/commitment_schemes/claim.hpp"
#include "barretenberg/srs/global_crs.hpp"
#include "barretenberg/stdlib/primitives/curves/bn254.hpp"
#include "barretenberg/stdlib/primitives/curves/grumpkin.hpp"
#include "barretenberg/stdlib/proof/proof.hpp"
#include "barretenberg/ultra_honk/ultra_verifier.hpp"
#include <gtest/gtest.h>

namespace bb::stdlib {
namespace {

using Builder = UltraCircuitBuilder;
using field_ct = field_t<Builder>;
using bool_ct = bool_t<Builder>;
using witness_ct = witness_t<Builder>;
using Bn254 = bn254<Builder>;
using fq_ct = Bn254::BaseField;
using G1 = Bn254::Group;
using cycle_group_ct = cycle_group<Builder>;
using PairingPointsCt = recursion::PairingPoints<Bn254>;
using OpeningClaimCt = OpeningClaim<grumpkin<Builder>>;
using VerifierOutput = recursion::honk::UltraRecursiveVerifierOutput<Builder>;

// Builder pointers of every member, read through public accessors only so the walk is checked independently.
using Contexts = std::vector<const Builder*>;

void collect_contexts(const field_ct& value, Contexts& out)
{
    out.push_back(value.get_context());
}

void collect_contexts(const bool_ct& value, Contexts& out)
{
    out.push_back(value.get_context());
}

void collect_contexts(const fq_ct& value, Contexts& out)
{
    out.push_back(value.context);
    for (size_t i = 0; i < fq_ct::NUM_LIMBS; i++) {
        collect_contexts(value.get_limb(i).element, out);
    }
    collect_contexts(value.get_prime_basis_limb(), out);
}

void collect_contexts(const G1& value, Contexts& out)
{
    collect_contexts(value.x(), out);
    collect_contexts(value.y(), out);
    collect_contexts(value.is_point_at_infinity(), out);
}

void collect_contexts(const cycle_group_ct& value, Contexts& out)
{
    out.push_back(value.get_context());
    collect_contexts(value.x(), out);
    collect_contexts(value.y(), out);
    collect_contexts(value.is_point_at_infinity(), out);
}

void collect_contexts(const PairingPointsCt& value, Contexts& out)
{
    collect_contexts(value.P0(), out);
    collect_contexts(value.P1(), out);
}

void collect_contexts(const OpeningClaimCt& value, Contexts& out)
{
    collect_contexts(value.opening_pair.challenge, out);
    collect_contexts(value.opening_pair.evaluation, out);
    collect_contexts(value.commitment, out);
}

template <typename T> void collect_contexts(const std::vector<T>& values, Contexts& out)
{
    for (const auto& value : values) {
        collect_contexts(value, out);
    }
}

template <typename T, size_t N> void collect_contexts(const std::array<T, N>& values, Contexts& out)
{
    for (const auto& value : values) {
        collect_contexts(value, out);
    }
}

void collect_contexts(const VerifierOutput& value, Contexts& out)
{
    collect_contexts(value.points_accumulator, out);
    collect_contexts(value.ipa_claim, out);
    collect_contexts(value.ipa_proof, out);
    collect_contexts(value.kernel_return_data, out);
    collect_contexts(value.ecc_op_tables, out);
    collect_contexts(value.transcript_hash, out);
}

template <typename T> Contexts contexts_of(const T& value)
{
    Contexts out;
    collect_contexts(value, out);
    return out;
}

// Every pointer that was on `from` is on `to`, and every null pointer stayed null.
void expect_moved(const Contexts& before, const Contexts& after, const Builder* from, const Builder* to)
{
    ASSERT_EQ(before.size(), after.size());
    for (size_t i = 0; i < before.size(); i++) {
        if (before[i] == nullptr) {
            EXPECT_EQ(after[i], nullptr) << "member " << i;
        } else {
            EXPECT_EQ(before[i], from) << "member " << i;
            EXPECT_EQ(after[i], to) << "member " << i;
        }
    }
}

void expect_only_on(const Contexts& contexts, const Builder* builder)
{
    for (const auto* context : contexts) {
        EXPECT_TRUE(context == nullptr || context == builder);
    }
}

struct IndexRecorder {
    std::vector<uint32_t> witnesses;
    std::vector<uint32_t> tags;
    void context(Builder*& /*context*/) {}
    void witness(uint32_t& index) { witnesses.push_back(index); }
    void pairing_tag(uint32_t& tag) { tags.push_back(tag); }
};

template <typename T> IndexRecorder record_indices(T& object)
{
    IndexRecorder recorder;
    BuilderBindingWalk<Builder>::visit(object, recorder);
    return recorder;
}

// One value of every kind a recursion constraint can return, on one builder.
struct Values {
    field_ct witness;
    field_ct scaled;
    field_ct constant_on_builder;
    field_ct constant_without_builder;
    bool_ct flag;
    bool_ct inverted_flag;
    bool_ct constant_flag;
    fq_ct big;
    fq_ct big_constant;
    G1 point;
    cycle_group_ct grumpkin_point;
    PairingPointsCt pairing_points;
    VerifierOutput output;

    template <typename F> void for_each(F&& f)
    {
        f(witness);
        f(scaled);
        f(constant_on_builder);
        f(constant_without_builder);
        f(flag);
        f(inverted_flag);
        f(constant_flag);
        f(big);
        f(big_constant);
        f(point);
        f(grumpkin_point);
        f(pairing_points);
        f(output);
    }

    // Both pairing-point objects register one tag each.
    static constexpr uint32_t NUM_PAIRING_TAGS = 2;

    static Values make(Builder& builder)
    {
        Values v;
        v.witness = field_ct::from_witness(&builder, fr(5));
        v.scaled = v.witness * fr(3) + fr(7);
        v.constant_on_builder = field_ct(&builder, fr(9));
        v.constant_without_builder = field_ct(fr(11));
        v.flag = bool_ct(witness_ct(&builder, true));
        v.inverted_flag = !v.flag;
        v.constant_flag = bool_ct(&builder, false);
        v.big = fq_ct::from_witness(&builder, fq::random_element());
        v.big_constant = fq_ct(&builder, uint256_t(123456789));
        v.point = G1::from_witness(&builder, g1::affine_element::random_element());
        v.grumpkin_point = cycle_group_ct::from_witness(&builder, cycle_group_ct::AffineElement::random_element());
        v.pairing_points = PairingPointsCt(G1::from_witness(&builder, g1::affine_element::random_element()),
                                           G1::from_witness(&builder, g1::affine_element::random_element()));
        v.output.points_accumulator = PairingPointsCt(G1::from_witness(&builder, g1::affine_element::random_element()),
                                                      G1::from_witness(&builder, g1::affine_element::random_element()));
        v.output.ipa_claim =
            OpeningClaimCt{ { fq_ct::from_witness(&builder, fq::random_element()),
                              fq_ct::from_witness(&builder, fq::random_element()) },
                            cycle_group_ct::from_witness(&builder, cycle_group_ct::AffineElement::random_element()) };
        v.output.ipa_proof = Proof<Builder>(builder, HonkProof{ fr(1), fr(2), fr::random_element() });
        return v;
    }
};

void expect_same_values(const PairingPointsCt& a, const PairingPointsCt& b)
{
    EXPECT_EQ(a.P0().get_value(), b.P0().get_value());
    EXPECT_EQ(a.P1().get_value(), b.P1().get_value());
    EXPECT_EQ(a.is_populated(), b.is_populated());
    EXPECT_EQ(a.is_default(), b.is_default());
}

void expect_same_values(const Values& a, const Values& b)
{
    EXPECT_EQ(a.witness.get_value(), b.witness.get_value());
    EXPECT_EQ(a.scaled.get_value(), b.scaled.get_value());
    EXPECT_EQ(a.scaled.additive_constant, b.scaled.additive_constant);
    EXPECT_EQ(a.scaled.multiplicative_constant, b.scaled.multiplicative_constant);
    EXPECT_EQ(a.witness.get_origin_tag(), b.witness.get_origin_tag());
    EXPECT_EQ(a.constant_on_builder.get_value(), b.constant_on_builder.get_value());
    EXPECT_EQ(a.constant_without_builder.get_value(), b.constant_without_builder.get_value());
    EXPECT_EQ(a.flag.get_value(), b.flag.get_value());
    EXPECT_EQ(a.inverted_flag.get_value(), b.inverted_flag.get_value());
    EXPECT_EQ(a.constant_flag.get_value(), b.constant_flag.get_value());
    EXPECT_EQ(a.big.get_value(), b.big.get_value());
    EXPECT_EQ(a.big_constant.get_value(), b.big_constant.get_value());
    for (size_t i = 0; i < fq_ct::NUM_LIMBS; i++) {
        EXPECT_EQ(a.big.get_limb(i).maximum_value, b.big.get_limb(i).maximum_value);
        EXPECT_EQ(a.big.get_limb(i).element.get_value(), b.big.get_limb(i).element.get_value());
    }
    EXPECT_EQ(a.big.get_prime_basis_limb().get_value(), b.big.get_prime_basis_limb().get_value());
    EXPECT_EQ(a.point.get_value(), b.point.get_value());
    EXPECT_EQ(a.point.is_point_at_infinity().get_value(), b.point.is_point_at_infinity().get_value());
    EXPECT_EQ(a.grumpkin_point.get_value(), b.grumpkin_point.get_value());
    expect_same_values(a.pairing_points, b.pairing_points);
    expect_same_values(a.output.points_accumulator, b.output.points_accumulator);
    EXPECT_EQ(a.output.ipa_claim.opening_pair.challenge.get_value(),
              b.output.ipa_claim.opening_pair.challenge.get_value());
    EXPECT_EQ(a.output.ipa_claim.opening_pair.evaluation.get_value(),
              b.output.ipa_claim.opening_pair.evaluation.get_value());
    EXPECT_EQ(a.output.ipa_claim.commitment.get_value(), b.output.ipa_claim.commitment.get_value());
    EXPECT_EQ(a.output.ipa_proof.get_value(), b.output.ipa_proof.get_value());
}

struct Counts {
    size_t gates;
    size_t variables;
    bool operator==(const Counts& other) const = default;
};

Counts counts(const Builder& builder)
{
    return { builder.num_gates(), builder.get_num_variables() };
}

class BuilderBindingTests : public testing::Test {
  public:
    static void SetUpTestSuite() { bb::srs::init_file_crs_factory(bb::srs::bb_crs_path()); }
};

} // namespace

TEST_F(BuilderBindingTests, RebindsEveryMemberToTheRealBuilder)
{
    constexpr size_t EXTRA_VARIABLES = 7;
    constexpr uint32_t TAG_OFFSET = 3;

    Builder scratch;
    Values values = Values::make(scratch);
    const Values original = values;

    // The real builder holds every scratch variable after variables and pairing-point tags of its own.
    Builder real;
    for (size_t i = 0; i < EXTRA_VARIABLES; i++) {
        real.add_variable(fr::random_element());
    }
    for (uint32_t i = 0; i < TAG_OFFSET + Values::NUM_PAIRING_TAGS; i++) {
        real.pairing_points_tagging.create_pairing_point_tag();
    }
    std::vector<uint32_t> real_index_of;
    for (uint32_t i = 0; i < scratch.get_num_variables(); i++) {
        real_index_of.push_back(real.add_variable(scratch.get_variable(i)));
    }
    ASSERT_GT(real_index_of.front(), 0U);

    std::vector<Contexts> contexts_before;
    std::vector<IndexRecorder> indices_before;
    size_t on_scratch_before = 0;
    values.for_each([&](auto& member) {
        contexts_before.push_back(contexts_of(member));
        indices_before.push_back(record_indices(member));
        on_scratch_before += count_contexts_on(member, &scratch);
    });
    EXPECT_GT(on_scratch_before, 0U);

    const Counts scratch_counts = counts(scratch);
    const Counts real_counts = counts(real);
    values.for_each([&](auto& member) { rebind_to_builder(member, &real, real_index_of, TAG_OFFSET); });
    expect_same_values(original, values);
    EXPECT_EQ(counts(scratch), scratch_counts);
    EXPECT_EQ(counts(real), real_counts);

    size_t member = 0;
    values.for_each([&](auto& value) {
        expect_moved(contexts_before[member], contexts_of(value), &scratch, &real);
        EXPECT_EQ(count_contexts_on(value, &scratch), 0U) << "member " << member;

        const IndexRecorder after = record_indices(value);
        const IndexRecorder& before = indices_before[member];
        ASSERT_EQ(after.witnesses.size(), before.witnesses.size());
        for (size_t i = 0; i < before.witnesses.size(); i++) {
            if (before.witnesses[i] == IS_CONSTANT) {
                EXPECT_EQ(after.witnesses[i], IS_CONSTANT);
            } else {
                EXPECT_EQ(after.witnesses[i], real_index_of[before.witnesses[i]]);
                EXPECT_EQ(real.get_variable(after.witnesses[i]), scratch.get_variable(before.witnesses[i]));
            }
        }
        ASSERT_EQ(after.tags.size(), before.tags.size());
        for (size_t i = 0; i < before.tags.size(); i++) {
            EXPECT_EQ(after.tags[i], before.tags[i] + TAG_OFFSET);
        }
        member++;
    });
    EXPECT_EQ(values.constant_without_builder.get_context(), nullptr);
    EXPECT_TRUE(values.constant_on_builder.is_constant());
    EXPECT_EQ(values.constant_on_builder.get_context(), &real);
}

TEST_F(BuilderBindingTests, ArithmeticOnReboundValuesAllocatesOnTheRealBuilder)
{
    Builder scratch;
    Values values = Values::make(scratch);
    Builder real;
    for (uint32_t i = 0; i < Values::NUM_PAIRING_TAGS; i++) {
        real.pairing_points_tagging.create_pairing_point_tag();
    }
    std::vector<uint32_t> real_index_of;
    for (uint32_t i = 0; i < scratch.get_num_variables(); i++) {
        real_index_of.push_back(real.add_variable(scratch.get_variable(i)));
    }
    values.for_each([&](auto& member) { rebind_to_builder(member, &real, real_index_of, 0); });

    const Counts scratch_counts = counts(scratch);
    const Counts real_counts = counts(real);

    const field_ct product = values.witness * values.scaled;
    const bool_ct conjunction = values.flag & values.inverted_flag;
    const fq_ct big_square = values.big * values.big;
    const G1 doubled = values.point.dbl();
    const cycle_group_ct grumpkin_doubled = values.grumpkin_point.dbl();
    PairingPointsCt fresh(G1::from_witness(&real, g1::affine_element::random_element()),
                          G1::from_witness(&real, g1::affine_element::random_element()));
    values.pairing_points.aggregate(fresh);

    EXPECT_EQ(counts(scratch), scratch_counts);
    EXPECT_GT(real.num_gates(), real_counts.gates);
    EXPECT_GT(real.get_num_variables(), real_counts.variables);
    EXPECT_EQ(product.get_value(), fr(5 * (5 * 3 + 7)));
    EXPECT_FALSE(conjunction.get_value());
    expect_only_on(contexts_of(product), &real);
    expect_only_on(contexts_of(conjunction), &real);
    expect_only_on(contexts_of(big_square), &real);
    expect_only_on(contexts_of(doubled), &real);
    expect_only_on(contexts_of(grumpkin_doubled), &real);
    expect_only_on(contexts_of(values.pairing_points), &real);
    EXPECT_EQ(count_contexts_on(values.pairing_points, &scratch), 0U);
}

TEST_F(BuilderBindingTests, RebindsGoblinPointsOnMega)
{
    using MegaG1 = bn254<MegaCircuitBuilder>::Group;
    using MegaOutput = recursion::honk::UltraRecursiveVerifierOutput<MegaCircuitBuilder>;

    MegaCircuitBuilder scratch;
    MegaOutput output;
    output.kernel_return_data = MegaG1::from_witness(&scratch, g1::affine_element::random_element());
    output.transcript_hash = field_t<MegaCircuitBuilder>::from_witness(&scratch, fr::random_element());
    const auto point = output.kernel_return_data.get_value();
    const auto hash = output.transcript_hash.get_value();

    MegaCircuitBuilder real;
    real.add_variable(fr::random_element());
    std::vector<uint32_t> real_index_of;
    for (uint32_t i = 0; i < scratch.get_num_variables(); i++) {
        real_index_of.push_back(real.add_variable(scratch.get_variable(i)));
    }
    EXPECT_GT(count_contexts_on(output, &scratch), 0U);
    rebind_to_builder(output, &real, real_index_of, 0);

    EXPECT_EQ(count_contexts_on(output, &scratch), 0U);
    for (const auto& limb : output.kernel_return_data.x().limbs) {
        EXPECT_EQ(limb.get_context(), &real);
    }
    for (const auto& limb : output.kernel_return_data.y().limbs) {
        EXPECT_EQ(limb.get_context(), &real);
    }
    EXPECT_EQ(output.transcript_hash.get_context(), &real);
    EXPECT_EQ(output.kernel_return_data.get_value(), point);
    EXPECT_EQ(output.transcript_hash.get_value(), hash);
}

} // namespace bb::stdlib
