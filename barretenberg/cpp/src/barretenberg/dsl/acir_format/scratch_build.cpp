#include "barretenberg/dsl/acir_format/scratch_build.hpp"

#if BB_SCRATCH_BUILDS_AVAILABLE

#include "barretenberg/common/throw_or_abort.hpp"

#include <utility>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace acir_format::scratch_build {

namespace {

// 8 MiB, the usual main-thread default; a scratch build needs more stack than some platforms give other threads.
constexpr size_t SCRATCH_STACK_BYTES = size_t{ 8 } << 20;

} // namespace

ScratchCancellation::ScratchCancellation(const std::string_view owner, std::function<void()> before_check)
    : message_(std::string(owner) + ": the scratch build was cancelled")
    , before_check_(std::move(before_check))
{}

void ScratchCancellation::request() noexcept
{
    requested_.store(true, std::memory_order_relaxed);
}

void ScratchCancellation::check()
{
    if (before_check_) {
        before_check_();
    }
    if (requested_.load(std::memory_order_relaxed)) {
        throw_or_abort(message_);
    }
}

bool ScratchThread::start(void* (*body)(void*), void* arg)
{
    pthread_attr_t attributes;
    if (pthread_attr_init(&attributes) != 0) {
        return false;
    }
    int status = pthread_attr_setstacksize(&attributes, SCRATCH_STACK_BYTES);
    if (status == 0) {
        status = pthread_create(&thread_, &attributes, body, arg);
    }
    pthread_attr_destroy(&attributes);
    running_ = status == 0;
    return running_;
}

void ScratchThread::join()
{
    if (running_) {
        pthread_join(thread_, nullptr);
        running_ = false;
    }
}

std::string refusal_reason(const bb::SpliceRefusal refusal)
{
    using bb::SpliceRefusal;
    switch (refusal) {
    case SpliceRefusal::None:
        return "none";
    case SpliceRefusal::BuilderState:
        return "builder state";
    case SpliceRefusal::Failed:
        return "a builder has failed";
    case SpliceRefusal::PublicInput:
        return "the scratch build added a public input";
    case SpliceRefusal::InputValue:
        return "an input value changed";
    case SpliceRefusal::InputCycle:
        return "inputs share a copy cycle";
    case SpliceRefusal::InputTag:
        return "an input carries a range tag";
    case SpliceRefusal::RowForm:
        return "a row of a form the splice cannot copy";
    case SpliceRefusal::UnknownForm:
        return "state of a form the splice does not translate";
    }
    return "unknown";
}

void return_freed_memory()
{
#if defined(__GLIBC__)
    malloc_trim(0);
#endif
}

std::string message_of(const std::exception_ptr& error)
{
    try {
        std::rethrow_exception(error);
    } catch (const std::exception& exception) {
        return exception.what();
    } catch (...) {
        return "an exception that is not a std::exception";
    }
}

int64_t milliseconds_between(const std::chrono::steady_clock::time_point from,
                             const std::chrono::steady_clock::time_point to)
{
    return static_cast<int64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(to - from).count());
}

} // namespace acir_format::scratch_build

#endif
