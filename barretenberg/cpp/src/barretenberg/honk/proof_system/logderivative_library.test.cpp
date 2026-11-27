// Test bounded inverse rows against whole-range and individual native inversions.
#include "barretenberg/honk/proof_system/logderivative_library.hpp"
#include "barretenberg/ecc/curves/bn254/fr.hpp"
#include "barretenberg/polynomials/polynomial.hpp"
#include <array>
#include <gtest/gtest.h>
#include <vector>

namespace {
using FF = bb::fr;
struct InverseChunkRow {
    FF selector;
    FF a;
    FF b;
};
struct InverseChunkParameters {
    FF beta = FF(7);
    FF gamma = FF(13);
};
struct InverseChunkPolynomials {
    std::vector<InverseChunkRow> rows;
    bb::Polynomial<FF> inverse;
    explicit InverseChunkPolynomials(size_t n, size_t offset = 0)
        : rows(n + offset)
        , inverse(n, n + offset, offset)
    {
        for (size_t i = 0; i < rows.size(); ++i) {
            rows[i] = { FF(i % 5 == 0 ? 0 : 1), FF(i + 1), FF(3 * i + 2) };
            if (i % 11 == 0) {
                rows[i].a = -FF(7); // Active zero denominators retain the existing zero policy.
            }
        }
    }
    InverseChunkRow get_row(size_t i) const { return rows[i]; }
};
struct InverseChunkRelation {
    using ValueAccumulator0 = FF;
    static constexpr size_t NUM_LOOKUP_TERMS = 1;
    static constexpr size_t NUM_TABLE_TERMS = 1;
    static auto& get_inverse_polynomial(InverseChunkPolynomials& p) { return p.inverse; }
    static bool operation_exists_at_row(const InverseChunkRow& r) { return r.selector != FF(0); }
    template <typename Accumulator, size_t>
    static FF compute_lookup_term(const InverseChunkRow& r, const InverseChunkParameters& p)
    {
        return r.a + p.beta;
    }
    template <typename Accumulator, size_t>
    static FF compute_table_term(const InverseChunkRow& r, const InverseChunkParameters& p)
    {
        return r.b + p.gamma;
    }
};
FF inverse_chunk_oracle(const InverseChunkRow& row, const InverseChunkParameters& params)
{
    if (row.selector == FF(0)) {
        return FF(0);
    }
    const FF denominator = (row.a + params.beta) * (row.b + params.gamma);
    return denominator.is_zero() ? FF(0) : denominator.invert();
}
} // namespace

TEST(AvmInverseChunkPrototype, MatchesWholeRangeAndIndividualInversion)
{
    InverseChunkParameters params;
    for (size_t n : std::array<size_t, 6>{ 0, 1, 2, 7, 31, 257 }) {
        InverseChunkPolynomials whole(n);
        bb::compute_logderivative_inverse<FF, InverseChunkRelation>(whole, params);
        for (size_t chunk_size : std::array<size_t, 4>{ 1, 3, 8, 64 }) {
            InverseChunkPolynomials chunked(n);
            for (size_t start = 0; start < n;) {
                const size_t end = start + std::min(chunk_size, n - start);
                bb::compute_logderivative_inverse<FF, InverseChunkRelation>(chunked, params, start, end);
                start = end;
            }
            for (size_t i = 0; i < n; ++i) {
                EXPECT_EQ(chunked.inverse[i], whole.inverse[i]);
                EXPECT_EQ(chunked.inverse[i], inverse_chunk_oracle(chunked.rows[i], params));
            }
        }
    }
}

TEST(AvmInverseChunkPrototype, ExclusiveBoundsLeaveOtherRowsUntouched)
{
    InverseChunkParameters params;
    InverseChunkPolynomials p(17);
    for (size_t i = 0; i < 17; ++i) {
        p.inverse.at(i) = (i >= 5 && i < 9) ? FF(0) : FF(41);
    }
    bb::compute_logderivative_inverse<FF, InverseChunkRelation>(p, params, 5, 9);
    bb::compute_logderivative_inverse<FF, InverseChunkRelation>(p, params, 13, 12); // Empty range.
    for (size_t i = 0; i < 17; ++i) {
        EXPECT_EQ(p.inverse[i], (i >= 5 && i < 9) ? inverse_chunk_oracle(p.rows[i], params) : FF(41));
    }
}

TEST(AvmInverseChunkPrototype, BoundsRespectShiftedBackingAndPhysicalEnd)
{
    InverseChunkParameters params;
    InverseChunkPolynomials p(23, 1);
    for (size_t i = 1; i < 24; ++i) {
        p.inverse.at(i) = i < 4 || i >= 20 ? FF(0) : FF(41);
    }
    // The helper must clamp the first range to physical start and the last to physical end.
    bb::compute_logderivative_inverse<FF, InverseChunkRelation>(p, params, 0, 4);
    bb::compute_logderivative_inverse<FF, InverseChunkRelation>(p, params, 20, 100);
    for (size_t i = 1; i < 24; ++i) {
        EXPECT_EQ(p.inverse[i], i < 4 || i >= 20 ? inverse_chunk_oracle(p.rows[i], params) : FF(41));
    }
}
