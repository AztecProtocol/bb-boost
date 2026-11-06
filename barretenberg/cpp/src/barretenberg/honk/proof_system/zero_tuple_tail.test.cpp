#include "barretenberg/ecc/curves/bn254/fr.hpp"
#include "barretenberg/honk/proof_system/logderivative_library.hpp"
#include "barretenberg/polynomials/polynomial.hpp"
#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <utility>
#include <vector>

namespace {
using FF = bb::fr;
struct ZeroTailRow {
    FF selector;
    FF a;
    FF b;
};
struct ZeroTailParams {
    FF beta = FF(7);
    FF gamma = FF(13);
    mutable size_t lookup_calls = 0;
    mutable size_t table_calls = 0;
};
struct ZeroTailPolynomials {
    std::vector<ZeroTailRow> rows;
    bb::Polynomial<FF> inverse;
    ZeroTailPolynomials(size_t n, size_t source_end, size_t destination_end, size_t offset = 0)
        : rows(n + offset)
        , inverse(n, n + offset, offset)
    {
        for (size_t i = 0; i < rows.size(); ++i) {
            rows[i] = { FF(i % 5 == 0 ? 0 : 1),
                        i < source_end ? FF(i + 1) : FF(0),
                        i < destination_end ? FF(3 * i + 2) : FF(0) };
        }
    }
    ZeroTailRow get_row(size_t i) const { return rows[i]; }
};
struct ZeroTailRelation {
    using ValueAccumulator0 = FF;
    static constexpr size_t NUM_LOOKUP_TERMS = 1;
    static constexpr size_t NUM_TABLE_TERMS = 1;
    static auto& get_inverse_polynomial(ZeroTailPolynomials& p) { return p.inverse; }
    static bool operation_exists_at_row(const ZeroTailRow& r) { return r.selector != FF(0); }
    template <typename Accumulator, size_t> static FF compute_lookup_term(const ZeroTailRow& r, const ZeroTailParams& p)
    {
        ++p.lookup_calls;
        return r.a + p.beta; // Deliberately not gamma: reuse the original result, never substitute a formula.
    }
    template <typename Accumulator, size_t> static FF compute_table_term(const ZeroTailRow& r, const ZeroTailParams& p)
    {
        ++p.table_calls;
        return r.b + p.gamma;
    }
};
} // namespace

TEST(AvmZeroTupleTail, PreservesRawInversesAtIdenticalChunkBoundaries)
{
    for (size_t n : std::array<size_t, 5>{ 0, 1, 7, 65, 257 }) {
        for (size_t source_end : std::array<size_t, 3>{ 0, 3, n + 5 }) {
            for (size_t destination_end : std::array<size_t, 3>{ 0, 17, n + 5 }) {
                for (size_t offset : std::array<size_t, 2>{ 0, 1 }) {
                    ZeroTailPolynomials original(n, source_end, destination_end, offset);
                    ZeroTailPolynomials cached(n, source_end, destination_end, offset);
                    ZeroTailParams p, q;
                    for (size_t start = 0; start < n + offset;) {
                        const size_t end = std::min(start + 7, n + offset);
                        bb::compute_logderivative_inverse<FF, ZeroTailRelation>(original, p, start, end);
                        bb::compute_logderivative_inverse<FF, ZeroTailRelation, ZeroTailPolynomials, false, true>(
                            cached, q, start, end, source_end, destination_end);
                        start = end;
                    }
                    for (size_t i = offset; i < n + offset; ++i) {
                        for (size_t limb = 0; limb < 4; ++limb) {
                            EXPECT_EQ(cached.inverse[i].data[limb], original.inverse[i].data[limb]);
                        }
                    }
                    EXPECT_LE(q.lookup_calls, p.lookup_calls);
                    EXPECT_LE(q.table_calls, p.table_calls);
                }
            }
        }
    }
}

TEST(AvmZeroTupleTail, InitializesOnlyOnActiveSuffixRows)
{
    ZeroTailPolynomials p(65, 3, 17);
    ZeroTailParams params;
    bb::compute_logderivative_inverse<FF, ZeroTailRelation, ZeroTailPolynomials, false, true>(p, params, 0, 65, 3, 17);
    EXPECT_EQ(params.lookup_calls, 3U); // Active prefix rows1,2 then the first active zero tuple at3.
    EXPECT_EQ(params.table_calls, 14U); // Thirteen active prefix rows before17, then one zero tuple.

    ZeroTailPolynomials inactive(65, 3, 17);
    for (size_t i = 3; i < 65; ++i) {
        inactive.rows[i].selector = FF(0);
    }
    ZeroTailParams inactive_params;
    bb::compute_logderivative_inverse<FF, ZeroTailRelation, ZeroTailPolynomials, false, true>(
        inactive, inactive_params, 0, 65, 3, 17);
    EXPECT_EQ(inactive_params.lookup_calls, 2U);
    EXPECT_EQ(inactive_params.table_calls, 2U);
}

TEST(AvmZeroTupleTail, DefaultModeIgnoresOptionalBounds)
{
    ZeroTailPolynomials p(31, 31, 31), q(31, 31, 31);
    ZeroTailParams a, b;
    bb::compute_logderivative_inverse<FF, ZeroTailRelation>(p, a);
    bb::compute_logderivative_inverse<FF, ZeroTailRelation>(q, b, 0, 31, 0, 0);
    for (size_t i = 0; i < 31; ++i) {
        for (size_t limb = 0; limb < 4; ++limb) {
            EXPECT_EQ(p.inverse[i].data[limb], q.inverse[i].data[limb]);
        }
    }
    EXPECT_EQ(a.lookup_calls, b.lookup_calls);
    EXPECT_EQ(a.table_calls, b.table_calls);
}

namespace {
struct ShiftedZeroTailPolynomials {
    bb::Polynomial<FF> source;
    bb::Polynomial<FF> shifted_source;
    bb::Polynomial<FF> destination;
    bb::Polynomial<FF> inverse;
    ShiftedZeroTailPolynomials()
        : source(6, 32, 1)
        , destination(3, 32)
        , inverse(16, 32)
    {
        for (size_t i = 1; i < 7; ++i) {
            source.at(i) = FF(i + 2);
        }
        for (size_t i = 0; i < 3; ++i) {
            destination.at(i) = FF(2 * i + 1);
        }
        shifted_source = source.shifted();
    }
    ZeroTailRow get_row(size_t i) const { return { FF(1), shifted_source[i], destination[i] }; }
};
struct ShiftedZeroTailRelation : ZeroTailRelation {
    static auto& get_inverse_polynomial(ShiftedZeroTailPolynomials& p) { return p.inverse; }
};
} // namespace

TEST(AvmZeroTupleTail, UsesActualEndOfShiftedTupleViews)
{
    ShiftedZeroTailPolynomials original, cached;
    ASSERT_EQ(original.source.start_index(), 1U);
    ASSERT_EQ(original.source.end_index(), 7U);
    ASSERT_EQ(original.shifted_source.start_index(), 0U);
    ASSERT_EQ(original.shifted_source.end_index(), 6U);
    ZeroTailParams p, q;
    for (const auto& range : std::array<std::pair<size_t, size_t>, 2>{ { { 0, 5 }, { 5, 16 } } }) {
        bb::compute_logderivative_inverse<FF, ShiftedZeroTailRelation>(original, p, range.first, range.second);
        bb::compute_logderivative_inverse<FF, ShiftedZeroTailRelation, ShiftedZeroTailPolynomials, false, true>(
            cached, q, range.first, range.second, cached.shifted_source.end_index(), cached.destination.end_index());
    }
    for (size_t i = 0; i < 16; ++i) {
        for (size_t limb = 0; limb < 4; ++limb) {
            EXPECT_EQ(cached.inverse[i].data[limb], original.inverse[i].data[limb]);
        }
    }
    EXPECT_LT(q.lookup_calls, p.lookup_calls);
    EXPECT_LT(q.table_calls, p.table_calls);
}

TEST(AvmZeroTupleTail, CachesZeroTermsWithoutChangingZeroDenominators)
{
    ZeroTailPolynomials original(31, 3, 5), cached(31, 3, 5);
    ZeroTailParams p, q;
    p.beta = p.gamma = q.beta = q.gamma = FF(0);
    bb::compute_logderivative_inverse<FF, ZeroTailRelation>(original, p);
    bb::compute_logderivative_inverse<FF, ZeroTailRelation, ZeroTailPolynomials, false, true>(cached, q, 0, 31, 3, 5);
    for (size_t i = 0; i < 31; ++i) {
        for (size_t limb = 0; limb < 4; ++limb) {
            EXPECT_EQ(cached.inverse[i].data[limb], original.inverse[i].data[limb]);
        }
    }
    EXPECT_LT(q.lookup_calls, p.lookup_calls);
    EXPECT_LT(q.table_calls, p.table_calls);
}
