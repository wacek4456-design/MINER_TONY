#!/usr/bin/env python3
"""Sweep lane/VRAM settings on ONE card and report measured H/s.

Tony's hint decoded: do not chase one huge batch. Split the VRAM budget across
lanes so a short kernel is always in flight instead of the card idling between
batches. Which split wins is hardware-specific, so measure it.

The knobs, all already in miner.ini:
  [cuda] vram_reference_difficulty  lanes = reference // network_difficulty
  [cuda] max_lanes                  hard cap on that
  [cuda] max_batch_size             per-lane batch cap (0 = fill the budget)
  [efficiency] target_vram_pct      share of the card the miner may use
  [efficiency] desktop_headroom_pct VRAM kept free (pointless when headless)

Runs in its own directory, so the miners on the other cards keep earning.

  python3 xnm-tune.py --gpu 0 --seconds 180
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

BASE = Path(os.environ.get("XNM_BASE", "/root/xnminer-base"))
PLAN_RE = re.compile(r"lanes=(\d+)[x×]([\d,]+)|batch=([\d,]+)")
DIFF_RE = re.compile(r"[Dd]ifficulty[= ](\d+)")
POOL_DOWN_RE = re.compile(r"port 80 unreachable|Server unreachable")


STUB_PORT = 8899


def start_stub_pool(difficulty: int) -> None:
    """Local stand-in for xenblocks.io.

    Pointing the miner at a dead address does NOT work: it retries /difficulty
    for 180s before falling back to the configured memory_cost, which eats most
    of a run. This answers instantly, always with the same difficulty, and
    swallows submissions so nothing reaches the real pool.
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    payload = json.dumps({"difficulty": str(difficulty)}).encode()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _reply(self, body: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            self._reply(payload)

        def do_POST(self) -> None:
            n = int(self.headers.get("Content-Length") or 0)
            if n:
                self.rfile.read(n)
            self._reply(b'{"status":"accepted"}')

        def log_message(self, *args) -> None:
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", STUB_PORT), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()


def pool_up(base_url: str, timeout_s: float = 8.0) -> int | None:
    """Live difficulty, or None when the pool is not answering."""
    try:
        with urllib.request.urlopen(
            base_url.rstrip("/") + "/difficulty", timeout=timeout_s
        ) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        return int(data.get("difficulty", data.get("diff", 0))) or None
    except Exception:
        return None


def wait_for_pool(base_url: str, max_wait_s: int = 900) -> int | None:
    """xenblocks.io drops port 80 for minutes at a time; a run started during an
    outage measures nothing but the wait loop. Hold until it answers."""
    deadline = time.time() + max_wait_s
    warned = False
    while time.time() < deadline:
        diff = pool_up(base_url)
        if diff:
            if warned:
                print("    pool back up, resuming", flush=True)
            return diff
        if not warned:
            print("    pool unreachable - waiting (this is an outage, not a config)",
                  flush=True)
            warned = True
        time.sleep(15)
    return None


def set_ini(path: Path, section: str, key: str, value: str) -> None:
    """Set key inside section, appending the key if the section lacks it."""
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    in_section = False
    written = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if in_section and not written:
                out.append(f"{key} = {value}")
                written = True
            in_section = stripped == f"[{section}]"
        elif in_section and re.match(rf"^\s*{re.escape(key)}\s*=", line):
            out.append(f"{key} = {value}")
            written = True
            continue
        out.append(line)
    if in_section and not written:
        out.append(f"{key} = {value}")
        written = True
    if not written:
        out.append(f"[{section}]")
        out.append(f"{key} = {value}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def measure(run_dir: Path, gpu: int, seconds: int, offline: bool = True) -> dict:
    """One timed run; returns median steady-state H/s plus the chosen plan."""
    data = run_dir / "data"
    for name in ("session_timelapse.jsonl", "session.log", "miner.lock"):
        (data / name).unlink(missing_ok=True)
    data.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["XNM_GPU"] = str(gpu)

    # The timelapse samples (our H/s source) are written from the dashboard
    # refresh loop, so --no-dashboard produces no data at all. Run the miner
    # exactly as production does: dashboard on, attached to a pty. Output is
    # drained and thrown away.
    import pty

    master, slave = pty.openpty()
    proc = subprocess.Popen(
        [sys.executable, "main.py", "--max-seconds", str(seconds)],
        cwd=str(run_dir), env=env,
        stdout=slave, stderr=slave, stdin=slave, start_new_session=True,
    )
    os.close(slave)

    def drain() -> None:
        try:
            while os.read(master, 65536):
                pass
        except OSError:
            pass

    threading.Thread(target=drain, daemon=True).start()
    try:
        proc.wait(timeout=seconds + 180)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
    try:
        os.close(master)
    except OSError:
        pass

    samples: list[float] = []
    tl = data / "session_timelapse.jsonl"
    if tl.is_file():
        for line in tl.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "sample" and rec.get("hps"):
                samples.append(float(rec["hps"]))

    log = (data / "session.log").read_text(encoding="utf-8", errors="replace") \
        if (data / "session.log").is_file() else ""
    plan = ""
    max_lanes_seen = 1
    for m in PLAN_RE.finditer(log):
        if m.group(1):
            plan = f"{m.group(1)}x{m.group(2)}"
            max_lanes_seen = max(max_lanes_seen, int(m.group(1)))
        else:
            plan = f"1x{m.group(3)}"
    diffs = DIFF_RE.findall(log)

    # Drop the warm-up samples; the engine replans batch size on first difficulty read.
    steady = samples[2:] or samples
    hps = statistics.median(steady) if steady else 0.0

    # In offline mode the pool is deliberately unreachable, so "Server
    # unreachable" in the log is expected and says nothing about the run.
    if not samples:
        status = "NO DATA"
    elif offline:
        status = "ok"
    elif POOL_DOWN_RE.search(log):
        status = "partial"          # mined, but lost the pool for part of the run
    else:
        status = "ok"

    return {
        "hps": hps,
        "samples": len(samples),
        "plan": plan or "?",
        "lanes": max_lanes_seen,
        "difficulty": int(diffs[-1]) if diffs else 0,
        "rc": proc.returncode,
        "status": status,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0, help="physical card to test on")
    # Samples land every 30 s and the engine ramps lanes gradually, so short runs
    # measure the warm-up rather than the steady state.
    ap.add_argument("--seconds", type=int, default=420, help="per configuration")
    ap.add_argument("--dir", default="/root/xnminer-tune")
    ap.add_argument("--wallet", default="")
    ap.add_argument("--lanes", default="1,2,4,8", help="lane counts to try")
    ap.add_argument("--vram", default="69.09,92", help="target_vram_pct values")
    ap.add_argument("--batches", default="0", help="max_batch_size values, 0 = auto")
    ap.add_argument("--difficulty", type=int, default=1100,
                    help="Argon2 memory_cost to benchmark at; pinned for every run")
    args = ap.parse_args()

    run_dir = Path(args.dir)
    if not (run_dir / "main.py").is_file():
        if not (BASE / "main.py").is_file():
            sys.exit(f"ERROR: no miner source at {BASE} - run the bootstrap first")
        print(f"Preparing isolated copy at {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(BASE, run_dir, dirs_exist_ok=True)
        shutil.rmtree(run_dir / "native" / "XenblocksMiner-main", ignore_errors=True)
    ini = run_dir / "miner.ini"
    if not ini.is_file():
        shutil.copy(run_dir / "miner.ini.example", ini)

    wallet = args.wallet
    if not wallet:
        for line in (BASE.parent / "xnminer-gpu0" / "miner.ini").read_text().splitlines():
            if line.strip().startswith("address"):
                wallet = line.split("=", 1)[1].strip()
    if not wallet.startswith("0x"):
        sys.exit("ERROR: pass --wallet 0x...")
    set_ini(ini, "account", "address", wallet)
    set_ini(ini, "account", "worker", "xnminer-tune")
    # Dashboard ON: it drives the refresh loop that writes the timelapse samples
    # this benchmark measures. Its output goes to a pty and is discarded.
    set_ini(ini, "monitoring", "dashboard_enabled", "true")
    set_ini(ini, "monitoring", "woodyminer_enabled", "false")
    set_ini(ini, "efficiency", "desktop_headroom_pct", "5")
    lanes = [int(x) for x in args.lanes.split(",") if x.strip()]
    # Never let max_lanes silently cap a configuration we asked to measure -
    # that would report a lower lane count as if it were the tested one.
    set_ini(ini, "cuda", "max_lanes", str(max(lanes)))

    vrams = [float(x) for x in args.vram.split(",") if x.strip()]
    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    total = len(lanes) * len(vrams) * len(batches)
    mins = total * args.seconds / 60

    # Network difficulty swings (2100 -> 5100 -> 1100 within one sweep) and the
    # pool drops out for minutes at a time. Both make runs incomparable: memory
    # per hash and lane count are derived from the difficulty. So point the
    # benchmark at a dead address - the miner then mines at [mining] memory_cost
    # forever, identical for every configuration. It submits nothing, which is
    # exactly right for a throughput measurement.
    net_diff = args.difficulty
    start_stub_pool(net_diff)
    set_ini(ini, "server", "base_url", f"http://127.0.0.1:{STUB_PORT}")
    set_ini(ini, "mining", "memory_cost", str(net_diff))

    print(f"\nGPU {args.gpu} · {total} configurations × {args.seconds}s ≈ {mins:.0f} min")
    print(f"Offline benchmark, difficulty pinned at {net_diff} - nothing is submitted.")
    print("Other cards keep mining untouched.\n")

    probe = measure(run_dir, args.gpu, 120)
    print(f"Baseline: {probe['hps']:,.0f} H/s  (plan {probe['plan']}, "
          f"lanes {probe['lanes']}, {probe['status']})\n")

    results = []
    for want_lanes in lanes:
        for vram in vrams:
            for batch in batches:
                set_ini(ini, "cuda", "vram_reference_difficulty",
                        str(net_diff * want_lanes))
                set_ini(ini, "efficiency", "target_vram_pct", str(vram))
                set_ini(ini, "cuda", "max_batch_size", str(batch))
                label = f"lanes≈{want_lanes} vram={vram}% batch={batch or 'auto'}"
                print(f"  running {label} ...", flush=True)

                for attempt in (1, 2):
                    r = measure(run_dir, args.gpu, args.seconds)
                    if r["status"] == "ok":
                        break
                    if attempt == 1:
                        print(f"    {r['status']} - retrying once", flush=True)

                r.update(label=label, want_lanes=want_lanes, vram=vram, batch=batch)
                results.append(r)
                print(f"    {r['hps']:,.0f} H/s   plan {r['plan']}   "
                      f"lanes {r['lanes']}   [{r['status']}]")

    valid = [r for r in results if r["status"] == "ok"]
    invalid = [r for r in results if r["status"] != "ok"]
    valid.sort(key=lambda r: r["hps"], reverse=True)
    best = valid[0]["hps"] if valid else 0

    print("\n" + "=" * 84)
    print(f"{'H/s':>12}  {'vs best':>8}  {'lanes':>5}  {'plan':>14}  configuration")
    print("-" * 84)
    for r in valid:
        rel = f"{r['hps'] / best * 100:.0f}%" if best else "—"
        print(f"{r['hps']:>12,.0f}  {rel:>8}  {r['lanes']:>5}  {r['plan']:>14}  {r['label']}")
    for r in invalid:
        print(f"{'—':>12}  {'—':>8}  {'—':>5}  {r['status']:>14}  {r['label']}")
    print("=" * 84)
    if invalid:
        print(f"{len(invalid)} run(s) discarded - the pool was down, not the settings.")
    if valid:
        b = valid[0]
        print(f"\nBest: {b['label']}  ->  {b['hps']:,.0f} H/s")
        print("Apply to every card by editing /root/xnminer-gpu*/miner.ini:")
        print(f"  [cuda] vram_reference_difficulty = {net_diff * b['want_lanes']}")
        print(f"  [cuda] max_lanes = {max(lanes)}")
        print(f"  [cuda] max_batch_size = {b['batch']}")
        print(f"  [efficiency] target_vram_pct = {b['vram']}")
        print("  [efficiency] desktop_headroom_pct = 5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
