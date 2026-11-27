#!/usr/bin/env python3
"""Tests for bench.py: python3 -m unittest discover -s .github/bench"""
import contextlib
import io
import json
import math
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import bench  # noqa: E402


def run(case, point, repeat, ms, mb, exit=0, **kw):
    return {"case": case, "point": point, "repeat": repeat, "exit": exit, "ms": ms, "mb": mb, **kw}


class Aggregate(unittest.TestCase):
    def test_fastest_time_and_highest_memory(self):
        # p0: times 300, 200, 250 -> fastest 200 (mean 250, last 250); memory 10, 30, 20 -> highest 30 (last 20).
        # p1: times 100, 150, 90 -> fastest 90, on another repeat than p0's; memory 50, 40, 45 -> highest 50 (first).
        runs = [run("A", "p0", 0, 300, 10), run("A", "p1", 0, 100, 50),
                run("A", "p0", 1, 200, 30), run("A", "p1", 1, 150, 40),
                run("A", "p0", 2, 250, 20), run("A", "p1", 2, 90, 45)]
        agg = bench.aggregate(runs)
        self.assertEqual(agg["A"]["p0"], {"runs": 3, "ms": 200, "mb": 30})
        self.assertEqual(agg["A"]["p1"], {"runs": 3, "ms": 90, "mb": 50})

    def test_a_failed_run_drops_the_points_numbers(self):
        # One good run, then a kill: no numbers, the error kept; a later good run doesn't bring them back.
        runs = [run("A", "p0", 0, 100, 10), run("A", "p0", 1, 0, 0, exit=-9, error="killed (out of memory?)"),
                run("A", "p0", 2, 50, 5)]
        self.assertEqual(bench.aggregate(runs)["A"]["p0"], {"error": "killed (out of memory?)"})

    def test_failure_of_one_point_leaves_the_others(self):
        runs = [run("A", "p0", 0, 100, 10), run("A", "p1", 0, 0, 0, exit=1)]
        agg = bench.aggregate(runs)
        self.assertEqual(agg["A"]["p0"]["ms"], 100)
        self.assertEqual(agg["A"]["p1"], {"error": "exit 1"})

    def test_swapping_is_flagged(self):
        runs = [run("A", "p0", 0, 100, 10, swapped=False), run("A", "p0", 1, 120, 10, swapped=True)]
        self.assertTrue(bench.aggregate(runs)["A"]["p0"]["swapped"])


class Rotation(unittest.TestCase):
    def test_rotates_by_repeat_and_input(self):
        self.assertEqual(bench.rotation(4, 0, 0), [0, 1, 2, 3])
        self.assertEqual(bench.rotation(4, 1, 0), [1, 2, 3, 0])
        self.assertEqual(bench.rotation(4, 0, 2), [2, 3, 0, 1])
        self.assertEqual(bench.rotation(4, 2, 3), [1, 2, 3, 0])  # (2 + 3) % 4 = 1

    def test_every_point_goes_first_once_over_n_inputs(self):
        firsts = [bench.rotation(4, 1, i)[0] for i in range(4)]
        self.assertEqual(sorted(firsts), [0, 1, 2, 3])

    def test_each_order_holds_every_point_once(self):
        for r in range(3):
            for i in range(5):
                self.assertEqual(sorted(bench.rotation(4, r, i)), [0, 1, 2, 3])


class Summary(unittest.TestCase):
    CASES = [
        {"id": "A", "prefix": "x/A", "values": {"p0": {"ms": 200, "mb": 100}, "p1": {"ms": 100, "mb": 100}}},
        {"id": "B", "prefix": "x/B", "values": {"p0": {"ms": 400, "mb": 200}, "p1": {"ms": 400, "mb": 50}}},
        {"id": "C", "prefix": "x/C", "values": {"p0": {"ms": 1000, "mb": 10}, "p1": {"error": "exit 1"}}},
        {"id": "D", "prefix": "x/D", "values": {"p0": {"error": "exit 1"}, "p1": {"ms": 5, "mb": 5}}},
    ]

    def test_geometric_mean_of_ratios_over_inputs_both_have(self):
        s = bench.column_summary(self.CASES, ["p0", "p1"])
        # p1 time: A 100/200 = 0.5, B 400/400 = 1.0; C (p1 failed) and D (base failed) don't count.
        # geomean = sqrt(0.5 * 1.0) = 0.70710678 (the arithmetic mean would be 0.75).
        self.assertAlmostEqual(s["p1"]["time_ratio"], math.sqrt(0.5), places=9)
        # p1 memory: A 100/100 = 1.0, B 50/200 = 0.25 -> sqrt(0.25) = 0.5.
        self.assertAlmostEqual(s["p1"]["memory_ratio"], 0.5, places=9)
        self.assertEqual(s["p1"]["n"], 2)
        # The base against itself: A, B and C, all 1.
        self.assertEqual(s["p0"], {"time_ratio": 1.0, "memory_ratio": 1.0, "n": 3})

    def test_no_common_inputs(self):
        s = bench.column_summary(self.CASES[2:3], ["p0", "p1"])
        self.assertEqual(s["p1"], {"time_ratio": None, "memory_ratio": None, "n": 0})

    def test_pct(self):
        self.assertEqual(bench.pct(math.sqrt(0.5)), "-29.3%")  # (0.7071 - 1) * 100 = -29.29
        self.assertEqual(bench.pct(1.0951), "+9.5%")
        self.assertEqual(bench.pct(None), "–")

    def test_geomean(self):
        self.assertAlmostEqual(bench.geomean([2, 8]), 4.0)  # sqrt(16)
        self.assertIsNone(bench.geomean([]))
        self.assertIsNone(bench.geomean([0.0, -1.0]))


class Output(unittest.TestCase):
    def test_bench_entries_use_aztec_packages_names_and_units(self):
        case = {"prefix": "golf-inputs/rollup/Root", "values": {"p0": {"ms": 1234.56789, "mb": 15700.04}}}
        self.assertEqual(bench.bench_entries(case, "p0"), [
            {"name": "golf-inputs/rollup/Root/seconds", "value": 1234.568, "unit": "ms"},
            {"name": "golf-inputs/rollup/Root/memory", "value": 15700.0, "unit": "MB"}])
        self.assertEqual(bench.bench_entries(case, "p1"), [])

    def test_gbench_time_units(self):
        self.assertAlmostEqual(bench.gbench_ms({"benchmarks": [{"real_time": 1234.5, "time_unit": "us"}]}), 1.2345)
        self.assertAlmostEqual(bench.gbench_ms({"benchmarks": [{"real_time": 4847.5, "time_unit": "ms"}]}), 4847.5)
        self.assertAlmostEqual(bench.gbench_ms({"benchmarks": [{"real_time": 2.0, "time_unit": "s"}]}), 2000.0)

    def test_build_data_orders_columns_and_checks_points(self):
        pts = [{"id": "p0", "label": "base"}, {"id": "p1", "label": "+1"}]
        res = lambda col, ids=("p0", "p1"): {"column": col, "points": list(ids), "machine": {}, "threads": 4,
                                             "repeats": 3, "skipped": [], "stopped_early": False,
                                             "cases": Summary.CASES[:2]}
        data = bench.build_data(pts, [res("avm"), res("client")], None, "now")
        self.assertEqual([c["column"] for c in data["columns"]], ["client", "avm"])
        with self.assertRaises(ValueError):
            bench.build_data(pts, [res("avm", ("p1", "p0"))], None, "now")
        md = bench.summary_markdown(data)
        self.assertIn("| **Time vs base** (geomean) | +0.0% (n=2) | -29.3% (n=2) |", md)
        self.assertIn("| A | 0.20 s, 0.10 GB | 0.10 s, 0.10 GB (-50.0%) |", md)


class Discover(unittest.TestCase):
    def test_names(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            for circuit, req in (("RootRollupArtifact", "bb"), ("PublicTxBaseRollupArtifact", "bb-avm"),
                                 ("InboxParity4Artifact", "bb"), ("PublicChonkVerifier", "bb")):
                d = t / "in" / "rollup" / circuit / "abc123"
                d.mkdir(parents=True)
                (d / "flags").write_text("--scheme ultra_honk --disable_zk\n")
                (d / "requires").write_text(req + "\n")
            a = t / "in" / "avm" / "megabulk-b3775dcaea13"
            a.mkdir(parents=True)
            (a / "avm_inputs.bin").write_bytes(b"x")
            for flow in ("f1", "f2"):
                (t / "flows" / flow).mkdir(parents=True)
                (t / "flows" / flow / "ivc-inputs.msgpack").write_bytes(b"x")
            (t / "in" / "client" / "f1").mkdir(parents=True)
            (t / "in" / "client" / "f1" / "vk").write_bytes(b"k")

            rollup = {c.id: c for c in bench.discover("rollup", t / "in", None)}
            self.assertEqual(sorted(rollup), ["rollup/InboxParity4", "rollup/PublicChonkVerifier",
                                              "rollup/PublicTxBase", "rollup/Root"])
            self.assertEqual(rollup["rollup/PublicTxBase"].binary, "bb-avm")
            self.assertEqual(rollup["rollup/Root"].prefix, "golf-inputs/rollup/Root")
            self.assertEqual(rollup["rollup/Root"].args[:4], ["prove", "--scheme", "ultra_honk", "--disable_zk"])
            avm = bench.discover("avm", t / "in", None)
            self.assertEqual([(c.id, c.prefix, c.avm_public_inputs) for c in avm],
                             [("avm/megabulk", "golf-inputs/avm/megabulk", None)])
            client = bench.discover("client", t / "in", t / "flows")
            self.assertEqual([c.prefix for c in client], ["aztec-packages/app-proving/f1/native",
                                                          "aztec-packages/app-proving/f2/native"])
            self.assertEqual([c.vk is not None for c in client], [True, False])
            self.assertEqual([c.id for c in bench.discover("micro", None, None)],
                             ["micro/ultra_honk", "micro/ultra_honk_zk"])


FAKE_BB = """#!/usr/bin/env python3
# A fake bb: `prove ... -o OUT` sleeps SLEEP s, writes PROOF to OUT/proof; `verify ... -p P` checks P says good
# (or, if LENIENT, accepts anything, as a broken bb might).
import sys, time
args = sys.argv[1:]
if args[0] == "verify":
    sys.exit(0 if {lenient} or open(args[args.index("-p") + 1]).read() == "good" else 1)
if {fail}:
    print("boom"); sys.exit(3)
time.sleep({sleep})
out = args[args.index("-o") + 1]
open(out + "/proof", "w").write("{proof}")
open(out + "/public_inputs", "w").write("")
"""


class RunEndToEnd(unittest.TestCase):
    def test_run_times_every_point_and_checks_proofs_with_the_first(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            d = t / "in" / "rollup" / "TxMergeRollupArtifact" / "abc"
            d.mkdir(parents=True)
            (d / "flags").write_text("--scheme ultra_honk\n")
            for name in ("vk", "bytecode.gz", "witness.gz"):
                (d / name).write_bytes(b"x")
            # p0: the original; p1: faster; p2: fast but its proof is wrong, and its own verifier accepts it; p3: fails.
            fakes = {"p0": (0.2, "good", False, False), "p1": (0.05, "good", False, False),
                     "p2": (0.01, "bad", False, True), "p3": (0, "good", True, False)}
            for pid, (sleep, proof, fail, lenient) in fakes.items():
                (t / "bins" / pid).mkdir(parents=True)
                exe = t / "bins" / pid / "bb"
                exe.write_text(FAKE_BB.format(sleep=sleep, proof=proof, fail=fail, lenient=lenient))
                exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
            (t / "points.json").write_text(json.dumps([{"id": p} for p in fakes]))
            out = t / "results-rollup.json"
            with contextlib.redirect_stdout(io.StringIO()) as printed:  # keep its ::error:: line out of CI logs
                code = bench.main(["run", "--points", str(t / "points.json"), "--bins", str(t / "bins"),
                                   "--column", "rollup", "--inputs", str(t / "in"), "--repeats", "2",
                                   "--threads", "2", "--work", str(t / "work"), "--out", str(out)])
            self.assertIn("::error::Proofs that don't verify: rollup/TxMerge @ p2", printed.getvalue())
            self.assertEqual(code, 1)  # p2's proof doesn't verify
            res = json.loads(out.read_text())
            self.assertEqual(len(res["runs"]), 8)  # 1 input x 4 points x 2 repeats
            # Repeat 0 starts with p0, repeat 1 with p1.
            self.assertEqual([r["point"] for r in res["runs"]], ["p0", "p1", "p2", "p3", "p1", "p2", "p3", "p0"])
            (case,) = res["cases"]
            self.assertEqual(case["id"], "rollup/TxMerge")
            v = case["values"]
            self.assertGreaterEqual(v["p0"]["ms"], 200)
            self.assertLess(v["p1"]["ms"], v["p0"]["ms"])
            self.assertEqual(v["p0"]["runs"], 2)
            self.assertTrue(v["p3"]["error"].startswith("exit 3\n"))
            self.assertIn("boom", v["p3"]["error"])
            self.assertEqual(case["verified"], {"p0": True, "p1": True, "p2": False})

    def test_measure_reports_the_childs_peak_memory(self):
        with tempfile.TemporaryDirectory() as t:
            code, ms, mb, _ = bench.measure(
                [sys.executable, "-c", "b = bytearray(200 * 1000 * 1000); b[::4096] = b'x' * len(b[::4096])"],
                dict(os.environ), Path(t) / "log")
            self.assertEqual(code, 0)
            self.assertGreater(mb, 200)  # at least the 200 MB it touched
            self.assertLess(mb, 400)
            code, *_ = bench.measure([sys.executable, "-c", "raise SystemExit(7)"], dict(os.environ), Path(t) / "l2")
            self.assertEqual(code, 7)


if __name__ == "__main__":
    unittest.main()
