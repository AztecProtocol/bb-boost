#pragma once
#include "barretenberg/common/assert.hpp"
#include "barretenberg/stdlib/primitives/bigfield/bigfield.hpp"
#include "barretenberg/stdlib/primitives/bigfield/goblin_field.hpp"
#include "barretenberg/stdlib/primitives/biggroup/biggroup.hpp"
#include "barretenberg/stdlib/primitives/bool/bool.hpp"
#include "barretenberg/stdlib/primitives/field/field.hpp"
#include "barretenberg/stdlib/primitives/group/cycle_group.hpp"
#include "barretenberg/stdlib/primitives/pairing_points.hpp"
#include "barretenberg/stdlib_circuit_builders/pairing_points_tagging.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace bb {
template <typename Curve> class OpeningPair;
template <typename Curve> class OpeningClaim;
} // namespace bb

namespace bb::stdlib::recursion::honk {
template <typename Builder> struct UltraRecursiveVerifierOutput;
} // namespace bb::stdlib::recursion::honk

namespace bb::stdlib {

/**
 * @brief Visits, in place, every builder pointer, witness index and pairing-point tag index a stdlib value holds.
 * @details Reads and writes members directly and calls nothing that can add a gate, a variable or a tag. A `Visitor`
 * provides `context(Builder*&)`, `witness(uint32_t&)` and `pairing_tag(uint32_t&)`. Covers `field_t`, `bool_t`,
 * `bigfield`, `goblin_field`, both bn254 group element kinds, `cycle_group`, `PairingPoints`, `OpeningPair`,
 * `OpeningClaim`, `UltraRecursiveVerifierOutput` and vectors and arrays of these (`Proof` included).
 */
template <typename Builder> struct BuilderBindingWalk {
    template <typename Visitor> static void visit(field_t<Builder>& value, Visitor& visitor)
    {
        visitor.context(value.context);
        visitor.witness(value.witness_index);
    }

    template <typename Visitor> static void visit(bool_t<Builder>& value, Visitor& visitor)
    {
        visitor.context(value.context);
        visitor.witness(value.witness_index);
    }

    template <typename Params, typename Visitor> static void visit(bigfield<Builder, Params>& value, Visitor& visitor)
    {
        visitor.context(value.context);
        for (auto& limb : value.binary_basis_limbs) {
            visit(limb.element, visitor);
        }
        visit(value.prime_basis_limb, visitor);
    }

    template <typename Visitor> static void visit(goblin_field<Builder>& value, Visitor& visitor)
    {
        for (auto& limb : value.limbs) {
            visit(limb, visitor);
        }
    }

    template <typename Fq, typename Fr, typename NativeGroup, typename Visitor>
    static void visit(element_default::element<Builder, Fq, Fr, NativeGroup>& value, Visitor& visitor)
    {
        visit(value._x, visitor);
        visit(value._y, visitor);
        visit(value._is_infinity, visitor);
    }

    template <typename Fq, typename Fr, typename NativeGroup, typename Visitor>
    static void visit(element_goblin::goblin_element<Builder, Fq, Fr, NativeGroup>& value, Visitor& visitor)
    {
        visit(value.x(), visitor);
        visit(value.y(), visitor);
    }

    template <typename Visitor> static void visit(cycle_group<Builder>& value, Visitor& visitor)
    {
        visitor.context(value.context);
        visit(value._x, visitor);
        visit(value._y, visitor);
        visit(value._is_infinity, visitor);
    }

    template <typename Curve, typename Visitor>
        requires std::is_same_v<typename Curve::Builder, Builder>
    static void visit(recursion::PairingPoints<Curve>& value, Visitor& visitor)
    {
        for (auto& point : value) {
            visit(point, visitor);
        }
        visitor.pairing_tag(value.tag_index);
    }

    template <typename Curve, typename Visitor> static void visit(OpeningPair<Curve>& value, Visitor& visitor)
    {
        visit(value.challenge, visitor);
        visit(value.evaluation, visitor);
    }

    template <typename Curve, typename Visitor> static void visit(OpeningClaim<Curve>& value, Visitor& visitor)
    {
        visit(value.opening_pair, visitor);
        visit(value.commitment, visitor);
    }

    template <typename Visitor>
    static void visit(recursion::honk::UltraRecursiveVerifierOutput<Builder>& value, Visitor& visitor)
    {
        visit(value.points_accumulator, visitor);
        visit(value.ipa_claim, visitor);
        visit(value.ipa_proof, visitor);
        visit(value.kernel_return_data, visitor);
        visit(value.ecc_op_tables, visitor);
        visit(value.transcript_hash, visitor);
    }

    template <typename T, typename Visitor> static void visit(std::vector<T>& values, Visitor& visitor)
    {
        for (auto& value : values) {
            visit(value, visitor);
        }
    }

    template <typename T, size_t N, typename Visitor> static void visit(std::array<T, N>& values, Visitor& visitor)
    {
        for (auto& value : values) {
            visit(value, visitor);
        }
    }
};

/**
 * @brief Make `object`, built on a scratch builder, refer to `builder`, where each scratch variable `i` lives at
 * `real_index_of[i]`.
 * @details Sets every non-null builder pointer, on leaves and on aggregates that keep their own, to `builder`; maps
 * every witness index through `real_index_of`, leaving `IS_CONSTANT` alone; adds `pairing_tag_offset` to every
 * registered pairing-point tag index. Every other member (constants, cached values, limb bounds, infinity flags, origin
 * tags, population flags) keeps its value, and neither builder gains a gate, a variable or a tag.
 * @pre Every witness index in `object` is below `real_index_of.size()` and maps to a variable of `builder`.
 */
template <typename T, typename Builder>
void rebind_to_builder(T& object,
                       Builder* builder,
                       const std::vector<uint32_t>& real_index_of,
                       uint32_t pairing_tag_offset)
{
    struct Rebind {
        Builder* builder;
        const std::vector<uint32_t>& real_index_of;
        uint32_t pairing_tag_offset;

        void context(Builder*& context) const
        {
            if (context != nullptr) {
                context = builder;
            }
        }
        void witness(uint32_t& index) const
        {
            if (index == IS_CONSTANT) {
                return;
            }
            BB_ASSERT_LT(index, real_index_of.size(), "rebind_to_builder: witness index outside the index map");
            index = real_index_of[index];
            BB_ASSERT(index != IS_CONSTANT, "rebind_to_builder: a witness maps to the constant sentinel");
        }
        void pairing_tag(uint32_t& tag) const
        {
            if (tag != PairingPointsTagging::UNREGISTERED) {
                tag += pairing_tag_offset;
            }
        }
    };
    Rebind rebind{ builder, real_index_of, pairing_tag_offset };
    BuilderBindingWalk<Builder>::visit(object, rebind);
}

/**
 * @brief Count the builder pointers in `object`, on leaves and aggregates alike, that equal `builder`.
 * @details Walks the same members as rebind_to_builder() and reads them only; a nonzero count after rebinding means a
 * pointer was left on the scratch builder.
 */
template <typename T, typename Builder> size_t count_contexts_on(const T& object, const Builder* builder)
{
    struct Count {
        const Builder* builder;
        size_t count = 0;

        void context(Builder*& context) { count += (context == builder) ? 1 : 0; }
        void witness(uint32_t& /*index*/) {}
        void pairing_tag(uint32_t& /*tag*/) {}
    };
    Count counter{ builder };
    // The walk takes mutable references; this visitor writes nothing.
    BuilderBindingWalk<Builder>::visit(const_cast<T&>(object), counter);
    return counter.count;
}

} // namespace bb::stdlib
