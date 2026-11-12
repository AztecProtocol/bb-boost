#include "barretenberg/bb/cli.hpp"
#include "barretenberg/common/exit_release.hpp"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <stdio.h>
#endif

int main(int argc, char* argv[])
{
#ifdef _WIN32
    // stdin/stdout carry binary data (`-` paths, the msgpack API, curve constants), but the CRT
    // opens them in text mode, which stops reads at 0x1A and rewrites 0x0A.
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    const int rc = bb::parse_and_run_cli_command(argc, argv);
    if (rc == 0) {
        // On success, skip the single-core teardown of the process's memory. Every output file (proof, VK, public
        // inputs through write_file's open/write/close; --bench_out*, traces, --memory_profile_out through scoped
        // std::ofstream) is closed by now. What a normal exit still writes is written first: the BB_BENCH report of
        // ~GlobalBenchStatsContainer, the one writing destructor (bb has no atexit hook), and the buffered streams.
        // Whenever that cannot be done safely it returns, and the normal exit runs.
        bb::release_memory_and_exit(rc);
    }
    return rc;
}
