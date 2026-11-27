#!/usr/bin/env python3
"""bb-boost's benchmarks: time several bb builds ("points") on the same machine, taking turns.

  bench.py run --points points.json --bins DIR --column COLUMN --inputs DIR [--flows DIR] --out results.json
  bench.py publish --points points.json --results DIR --out DIR [--run-url URL]

`run` times one column's inputs with every point's binaries. Each repeat runs every input once per point, in a
rotating order, so no point always goes first. A point's time on an input is its fastest run, its memory the highest
peak RSS of its runs (aztec-golf's method). Each point's last proof is then checked with the first point's (the
original's) binary.

`publish` merges the columns' results into data.json (read by index.html), one aztec-packages style bench JSON per
point (`[{name, value, unit}]`, as in its `bench-out/*.bench.json`), and a Markdown summary.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

COLUMNS = ("client", "micro", "rollup", "avm")
COLUMN_TITLES = {
    "client": "Client: aztec-packages' pinned Chonk flows (native)",
    "micro": "Micro-benchmark: aztec-packages' UltraHonk proof of 2^20 gates",
    "rollup": "Rollup: aztec-golf's season 2 inputs",
    "avm": "AVM: aztec-golf's season 2 inputs",
}
MICRO = {
    "ultra_honk": "construct_proof_ultrahonk_power_of_2/20$",
    "ultra_honk_zk": "construct_proof_ultrahonk_zk_power_of_2/20$",
}
RUN_TIMEOUT_S = 40 * 60


@dataclass
class Case:
    """One input of a column, as `<column>/<short name>`; `prefix` names its metrics."""

    id: str
    prefix: str
    binary: str  # bb, bb-avm or ultra_honk_bench
    path: Path | None = None
    args: list[str] = field(default_factory=list)  # bb arguments before -o
    vk: Path | None = None  # client: the flow's VK, to verify
    flags: list[str] = field(default_factory=list)  # rollup: bb flags, to verify
    avm_public_inputs: Path | None = None


def discover(column: str, inputs: Path | None, flows: Path | None) -> list[Case]:
    cases: list[Case] = []
    if column == "client":
        for d in sorted(p for p in flows.iterdir() if (p / "ivc-inputs.msgpack").is_file()):
            vk = inputs / "client" / d.name / "vk" if inputs else None
            cases.append(Case(f"client/{d.name}", f"aztec-packages/app-proving/{d.name}/native", "bb", d,
                              ["prove", "--scheme", "chonk", "--ivc_inputs_path", str(d / "ivc-inputs.msgpack")],
                              vk=vk if vk and vk.is_file() else None))
    elif column == "micro":
        for name, filt in MICRO.items():
            cases.append(Case(f"micro/{name}", f"aztec-packages/bb-micro-bench/native/{name}", "ultra_honk_bench",
                              args=[f"--benchmark_filter={filt}"]))
    elif column == "rollup":
        for d in sorted(p for p in (inputs / "rollup").glob("*/*") if (p / "flags").is_file()):
            circuit = re.sub(r"(RollupArtifact|Artifact)$", "", d.parent.name)
            flags = (d / "flags").read_text().split()
            requires = (d / "requires").read_text().strip() if (d / "requires").is_file() else "bb"
            cases.append(Case(f"rollup/{circuit}", f"golf-inputs/rollup/{circuit}", requires, d,
                              ["prove", *flags, "-b", str(d / "bytecode.gz"), "-w", str(d / "witness.gz"),
                               "-k", str(d / "vk")], vk=d / "vk", flags=flags))
    elif column == "avm":
        for d in sorted(p for p in (inputs / "avm").iterdir() if (p / "avm_inputs.bin").is_file()):
            name = re.sub(r"-[0-9a-f]+$", "", d.name)
            pub = d / "avm_public_inputs.bin"
            cases.append(Case(f"avm/{name}", f"golf-inputs/avm/{name}", "bb-avm", d,
                              ["avm_prove", "--avm-inputs", str(d / "avm_inputs.bin")],
                              avm_public_inputs=pub if pub.is_file() else None))
    else:
        raise ValueError(f"unknown column {column}")
    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError(f"two inputs share a name in {column}: {ids}")
    return cases


def rotation(n_points: int, repeat: int, case_index: int) -> list[int]:
    """The order the points take on one input in one repeat: rotated by repeat + input, so the first slot moves."""
    k = (repeat + case_index) % n_points
    return [(k + j) % n_points for j in range(n_points)]


def aggregate(runs: list[dict]) -> dict:
    """{case: {point: {ms, mb, runs} or {error}}}: fastest time and highest peak RSS over a point's runs.

    A point with any failed run on an input has no numbers for it, only the first error."""
    out: dict[str, dict[str, dict]] = {}
    for r in runs:
        cell = out.setdefault(r["case"], {}).setdefault(r["point"], {"runs": 0})
        if "error" in cell:
            continue
        if r["exit"] != 0:
            out[r["case"]][r["point"]] = {"error": r.get("error") or f"exit {r['exit']}"}
            continue
        cell["runs"] += 1
        cell["ms"] = r["ms"] if "ms" not in cell else min(cell["ms"], r["ms"])
        cell["mb"] = r["mb"] if "mb" not in cell else max(cell["mb"], r["mb"])
        if r.get("swapped"):
            cell["swapped"] = True
    return out


def geomean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x > 0 and math.isfinite(x)]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def column_summary(cases: list[dict], point_ids: list[str]) -> dict:
    """Per point: the geometric mean over inputs of its time and memory as a ratio of the first point's.

    Only inputs where both the point and the first point have numbers count; `n` says how many."""
    base = point_ids[0]
    out = {}
    for p in point_ids:
        t, m = [], []
        for c in cases:
            b, v = c["values"].get(base, {}), c["values"].get(p, {})
            if "ms" in b and "ms" in v and b["ms"] > 0 and b["mb"] > 0:
                t.append(v["ms"] / b["ms"])
                m.append(v["mb"] / b["mb"])
        out[p] = {"time_ratio": geomean(t), "memory_ratio": geomean(m), "n": len(t)}
    return out


def bench_entries(case: dict, point: str) -> list[dict]:
    """aztec-packages' bench JSON entries for one input at one point (its `seconds` metric is in ms)."""
    v = case["values"].get(point, {})
    if "ms" not in v:
        return []
    return [{"name": f"{case['prefix']}/seconds", "value": round(v["ms"], 3), "unit": "ms"},
            {"name": f"{case['prefix']}/memory", "value": round(v["mb"], 1), "unit": "MB"}]


def pct(ratio: float | None) -> str:
    return "–" if ratio is None else f"{(ratio - 1) * 100:+.1f}%"


# ---------------------------------------------------------------- running


def swap_out_pages() -> int | None:
    try:
        for line in Path("/proc/vmstat").read_text().splitlines():
            if line.startswith("pswpout "):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def measure(cmd: list[str], env: dict, log: Path) -> tuple[int, float, float, bool]:
    """Runs cmd; returns (exit code, wall ms, peak RSS MB of that process, whether the machine swapped)."""
    swap0 = swap_out_pages()
    with open(log, "wb") as f:
        start = time.perf_counter()
        proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        deadline = start + RUN_TIMEOUT_S
        while True:
            pid, status, rusage = os.wait4(proc.pid, os.WNOHANG)
            if pid:
                break
            if time.perf_counter() > deadline:
                proc.kill()
                pid, status, rusage = os.wait4(proc.pid, 0)
                break
            time.sleep(0.05)
        wall_ms = (time.perf_counter() - start) * 1000
    code = os.waitstatus_to_exitcode(status)
    proc.returncode = code  # reaped by wait4 above
    swap1 = swap_out_pages()
    # ru_maxrss is kilobytes on Linux, bytes on macOS.
    rss_mb = rusage.ru_maxrss / (1e6 if sys.platform == "darwin" else 1e3)
    return code, wall_ms, rss_mb, swap0 is not None and swap1 is not None and swap1 > swap0


def tail(path: Path, n: int = 15) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def prove_cmd(case: Case, bins: Path, out: Path) -> list[str]:
    exe = str(bins / case.binary)
    if case.binary == "ultra_honk_bench":
        return [exe, *case.args, f"--benchmark_out={out / 'gbench.json'}", "--benchmark_out_format=json"]
    return [exe, *case.args, "-o", str(out)]


def verify_cmd(case: Case, base_bins: Path, out: Path) -> list[str] | None:
    if case.id.startswith("client/"):
        if not case.vk:
            return None
        return [str(base_bins / "bb"), "verify", "--scheme", "chonk", "-p", str(out / "proof"), "-k", str(case.vk)]
    if case.id.startswith("rollup/"):
        return [str(base_bins / case.binary), "verify", *case.flags, "-k", str(case.vk), "-p", str(out / "proof"),
                "-i", str(out / "public_inputs")]
    if case.id.startswith("avm/") and case.avm_public_inputs:
        return [str(base_bins / "bb-avm"), "avm_verify", "-p", str(out / "proof"),
                "--avm-public-inputs", str(case.avm_public_inputs)]
    return None  # the micro-benchmark writes no proof; avm_prove verifies its own proof


def cmd_run(a) -> int:
    points = json.loads(Path(a.points).read_text())
    pids = [p["id"] for p in points]
    bins = Path(a.bins)
    cases = discover(a.column, Path(a.inputs) if a.inputs else None, Path(a.flows) if a.flows else None)
    skipped = [c.id for c in cases if c.id in a.skip]
    cases = [c for c in cases if c.id not in a.skip]
    work = Path(a.work)
    env = dict(os.environ, HARDWARE_CONCURRENCY=str(a.threads))
    deadline = time.time() + a.deadline_minutes * 60
    results = {"column": a.column, "repeats": a.repeats, "threads": a.threads, "points": pids,
               "machine": machine_info(), "skipped": skipped, "cases": [], "runs": [], "stopped_early": False}
    runs: list[dict] = results["runs"]

    def save():
        agg = aggregate(runs)
        results["cases"] = [{"id": c.id, "prefix": c.prefix, "values": agg.get(c.id, {}),
                             "verified": verified.get(c.id, {})} for c in cases]
        Path(a.out).write_text(json.dumps(results, indent=1))

    verified: dict[str, dict] = {}
    for r in range(a.repeats):
        for i, case in enumerate(cases):
            if time.time() > deadline:
                results["stopped_early"] = True
                print(f"::warning::Deadline reached: stopped before repeat {r + 1} of {case.id}", flush=True)
                break
            for j in rotation(len(pids), r, i):
                pid = pids[j]
                out = work / f"{i:03d}" / pid
                shutil.rmtree(out, ignore_errors=True)
                out.mkdir(parents=True)
                log = work / f"{i:03d}-{pid}.log"
                code, ms, mb, swapped = measure(prove_cmd(case, bins / pid, out), env, log)
                run = {"case": case.id, "point": pid, "repeat": r, "exit": code, "ms": ms, "mb": mb,
                       "swapped": swapped}
                if code == 0 and case.binary == "ultra_honk_bench":
                    try:
                        run["ms"] = gbench_ms(json.loads((out / "gbench.json").read_text()))
                    except (OSError, ValueError, KeyError, IndexError) as e:
                        run["exit"], run["error"] = 1, f"no benchmark result: {e}"
                if code != 0:
                    why = "killed (out of memory?)" if code == -9 else f"exit {code}"
                    run["error"] = f"{why}\n{tail(log)}"
                runs.append(run)
                status = "ok" if run["exit"] == 0 else run["error"].splitlines()[0]
                print(f"[{r + 1}/{a.repeats}] {case.id} @ {pid}: {run['ms'] / 1000:.2f} s, {mb:.0f} MB"
                      f"{' (swapped)' if swapped else ''} {status}", flush=True)
                save()
        else:
            continue
        break

    # Each point's last proof, checked by the first point's binary (the original bb).
    for i, case in enumerate(cases):
        for pid in pids:
            out = work / f"{i:03d}" / pid
            cmd = verify_cmd(case, bins / pids[0], out)
            if cmd is None or not (out / "proof").exists():
                continue
            log = work / f"{i:03d}-{pid}-verify.log"
            code, *_ = measure(cmd, env, log)
            verified.setdefault(case.id, {})[pid] = code == 0
            print(f"verify {case.id} @ {pid}: {'ok' if code == 0 else 'FAILED'}", flush=True)
            if code != 0:
                print(tail(log), flush=True)
    save()
    bad = [f"{c} @ {p}" for c, v in verified.items() for p, ok in v.items() if not ok]
    if bad:
        print(f"::error::Proofs that don't verify: {', '.join(bad)}")
    return 1 if bad else 0


def gbench_ms(doc: dict) -> float:
    b = doc["benchmarks"][0]
    scale = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}[b["time_unit"]]
    return b["real_time"] * scale


def machine_info() -> dict:
    info = {"cpus": os.cpu_count()}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                info["memory_gb"] = round(int(line.split()[1]) / 1024 / 1024, 1)
    except OSError:
        pass
    runner = os.environ.get("RUNNER_NAME")
    if runner:
        info["runner"] = runner
    return info


# ---------------------------------------------------------------- publishing


def build_data(points: list[dict], results: list[dict], run_url: str | None, now: str) -> dict:
    pids = [p["id"] for p in points]
    order = {c: i for i, c in enumerate(COLUMNS)}
    columns = []
    for res in sorted(results, key=lambda r: order.get(r["column"], 99)):
        if res["points"] != pids:
            raise ValueError(f"{res['column']} was timed for {res['points']}, not {pids}")
        columns.append({
            "column": res["column"], "title": COLUMN_TITLES.get(res["column"], res["column"]),
            "machine": res["machine"], "threads": res["threads"], "repeats": res["repeats"],
            "skipped": res["skipped"], "stopped_early": res["stopped_early"], "cases": res["cases"],
            "summary": column_summary(res["cases"], pids),
        })
    return {"generated": now, "run_url": run_url, "points": points, "columns": columns}


def summary_markdown(data: dict) -> str:
    pts = data["points"]
    lines = ["# bb-boost benchmarks", ""]
    if data.get("run_url"):
        lines += [f"Run: {data['run_url']}", ""]
    for col in data["columns"]:
        m = col["machine"]
        lines += [f"## {col['title']}", "",
                  f"{m.get('cpus')} CPUs ({m.get('cpu', '?')}), {m.get('memory_gb', '?')} GB; "
                  f"`HARDWARE_CONCURRENCY={col['threads']}`; fastest of {col['repeats']} runs, highest peak RSS."
                  + (" **Stopped early (time limit)**: some inputs have fewer runs." if col["stopped_early"] else "")
                  + (f" Skipped: {', '.join(col['skipped'])}." if col["skipped"] else ""), ""]
        lines.append("| | " + " | ".join(p["label"] for p in pts) + " |")
        lines.append("|---|" + "---:|" * len(pts))
        s = col["summary"]
        lines.append("| **Time vs " + pts[0]["label"] + "** (geomean) | "
                     + " | ".join(f"{pct(s[p['id']]['time_ratio'])} (n={s[p['id']]['n']})" for p in pts) + " |")
        lines.append("| **Memory vs " + pts[0]["label"] + "** | "
                     + " | ".join(pct(s[p["id"]]["memory_ratio"]) for p in pts) + " |")
        for c in col["cases"]:
            cells = []
            base = c["values"].get(pts[0]["id"], {})
            for p in pts:
                v = c["values"].get(p["id"], {})
                if "ms" not in v:
                    cells.append("failed" if "error" in v else "–")
                    continue
                cell = f"{v['ms'] / 1000:.2f} s, {v['mb'] / 1000:.2f} GB"
                if p is not pts[0] and "ms" in base:
                    cell += f" ({pct(v['ms'] / base['ms'])})"
                if c.get("verified", {}).get(p["id"]) is False:
                    cell += " **proof doesn't verify**"
                if v.get("swapped"):
                    cell += " (swapped)"
                cells.append(cell)
            lines.append(f"| {c['id']} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def cmd_publish(a) -> int:
    points = json.loads(Path(a.points).read_text())
    results = [json.loads(p.read_text()) for p in sorted(Path(a.results).rglob("results-*.json"))]
    if not results:
        print("::error::No results to publish")
        return 1
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    data = build_data(points, results, a.run_url, now)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.json").write_text(json.dumps(data, indent=1))
    for p in points:
        entries = [e for col in data["columns"] for c in col["cases"] for e in bench_entries(c, p["id"])]
        (out / f"{p['id']}.bench.json").write_text(json.dumps(entries, indent=1))
    (out / "summary.md").write_text(summary_markdown(data))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--points", required=True)
    r.add_argument("--bins", required=True, help="one folder per point id, holding its binaries")
    r.add_argument("--column", required=True, choices=COLUMNS)
    r.add_argument("--inputs", help="the bench inputs (rollup/, avm/, client/<flow>/vk)")
    r.add_argument("--flows", help="the pinned Chonk flows")
    r.add_argument("--skip", nargs="*", default=[], help="input ids to leave out, e.g. rollup/Root")
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--threads", type=int, default=os.cpu_count())
    r.add_argument("--deadline-minutes", type=float, default=290)
    r.add_argument("--work", default="bench-work")
    r.add_argument("--out", required=True)
    p = sub.add_parser("publish")
    p.add_argument("--points", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--run-url")
    a = ap.parse_args(argv)
    return cmd_run(a) if a.cmd == "run" else cmd_publish(a)


if __name__ == "__main__":
    sys.exit(main())
