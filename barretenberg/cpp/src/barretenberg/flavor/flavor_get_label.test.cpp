#include "barretenberg/common/assert.hpp"
#include "barretenberg/common/ref_vector.hpp"
#include "barretenberg/flavor/flavor_concepts.hpp"
#include <array>
#include <cstddef>
#include <gtest/gtest.h>
#include <span>
#include <string>
#include <vector>

using namespace bb;

namespace {
// Entities held in one array, labelled in the same order, as the AVM flavor holds them.
struct SpanEntities {
    std::array<int, 4> entities{};
    std::array<std::string, 4> labels{ "a", "b", "c", "d" };
    std::span<int> get_all() { return entities; }
    std::span<const std::string> get_labels() const { return labels; }
};

// A view of the same entities that starts at the second one, as the AVM proving key exposes its unshifted entities.
struct SpanTail {
    SpanEntities& base;
    std::span<int> get_all() { return std::span<int>(base.entities).subspan(1); }
    std::span<const std::string> get_labels() const { return std::span<const std::string>(base.labels).subspan(1); }
};

// Entities gathered by reference, as the other flavors expose them.
struct RefEntities {
    int x = 0;
    int y = 0;
    std::vector<std::string> get_labels() const { return { "x", "y" }; }
    RefVector<int> get_all() { return { x, y }; }
};

// One label short of the entities.
struct MislabelledEntities {
    std::array<int, 2> entities{};
    std::array<std::string, 1> labels{ "a" };
    std::span<int> get_all() { return entities; }
    std::span<const std::string> get_labels() const { return labels; }
};
} // namespace

TEST(FlavorGetLabel, FindsEachEntity)
{
    SpanEntities all;
    for (size_t i = 0; i < all.entities.size(); ++i) {
        EXPECT_EQ(flavor_get_label(all, all.entities[i]), all.labels[i]);
    }
    SpanTail tail{ all };
    EXPECT_EQ(flavor_get_label(tail, all.entities[2]), "c");
    EXPECT_EQ(flavor_get_label(tail, all.entities[0]), "(unknown label)");
    const int outside = 0;
    EXPECT_EQ(flavor_get_label(all, outside), "(unknown label)");

    RefEntities refs;
    EXPECT_EQ(flavor_get_label(refs, refs.y), "y");
    EXPECT_EQ(flavor_get_label(refs, outside), "(unknown label)");
}

TEST(FlavorGetLabel, RequiresOneLabelPerEntity)
{
    MislabelledEntities entities;
    EXPECT_THROW_OR_ABORT(flavor_get_label(entities, entities.entities[0]), ".*Assertion failed.*");
}
