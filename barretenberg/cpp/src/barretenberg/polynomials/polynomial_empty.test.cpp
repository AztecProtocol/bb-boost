#include "barretenberg/common/thread.hpp"
#include "barretenberg/ecc/curves/bn254/fr.hpp"
#include "barretenberg/polynomials/polynomial.hpp"
#include <gtest/gtest.h>
#include <thread>

namespace bb {
namespace {

TEST(PolynomialEmpty, ZeroingConstructorOfAnEmptyPolynomialStartsNoWorkers)
{
    size_t workers_after = 1;
    size_t size = 1;
    size_t virtual_size = 0;
    // A fresh thread has no workers yet, so constructing the polynomial must not be what starts them.
    std::thread caller([&] {
        const Polynomial<fr> empty(0, 1024);
        size = empty.size();
        virtual_size = empty.virtual_size();
        workers_after = parallel_for_num_workers();
    });
    caller.join();

    EXPECT_EQ(size, 0U);
    EXPECT_EQ(virtual_size, 1024U);
#ifdef __wasm__
    // On wasm the pool is process-wide, so an earlier test in this binary may already have started it.
    (void)workers_after;
#else
    EXPECT_EQ(workers_after, 0U);
#endif
}

} // namespace
} // namespace bb
