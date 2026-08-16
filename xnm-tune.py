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
import time
from pathlib import Path

BASE = Path(os.environ.get("XNM_BASE", "/root/xnminer-base"))
PLAN_RE = re.compile(r"lanes=(\d+)×([\d,]+)|batch=([\d,]+)")
DIFF_RE = re.compile(r"Difficulty (\d+)")


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


def measure(run_dir: Path, gpu: int, seconds: int) -> dict:
    """One timed run; returns median steady-state H/s plus the chosen plan."""
    data = run_dir / "data"
    for name in ("session_timelapse.jsonl", "session.log", "miner.lock"):
        (data / name).unlink(missing_ok=True)
    data.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["XNM_GPU"] = str(gpu)
    proc = subprocess.run(
        [sys.executable, "main.py", "--no-dashboard", "--max-seconds", str(seconds)],
        cwd=run_dir, env=env, capture_output=True, text=True, timeout=seconds + 180,
    )

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
    for m in PLAN_RE.finditer(log):
        plan = f"{m.group(1)}x{m.group(2)}" if m.group(1) else f"1x{m.group(3)}"
    diffs = DIFF_RE.findall(log)

    # Drop the warm-up samples; the engine replans batch size on first difficulty read.
    steady = samples[2:] or samples
    return {
        "hps": statistics.median(steady) if steady else 0.0,
        "samples": len(samples),
        "plan": plan or "?",
        "difficulty": int(diffs[-1]) if diffs else 0,
        "rc": proc.returncode,
        "tail": (proc.stdout or proc.stderr or "").strip().splitlines()[-1:] or [""],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0, help="physical card to test on")
    ap.add_argument("--seconds", type=int, default=180, help="per configuration")
    ap.add_argument("--dir", default="/root/xnminer-tune")
    ap.add_argument("--wallet", default="")
    ap.add_argument("--lanes", default="1,2,4,8", help="lane counts to try")
    ap.add_argument("--vram", default="69.09,92", help="target_vram_pct values")
    ap.add_argument("--batches", default="0", help="max_batch_size values, 0 = auto")
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
    set_ini(ini, "monitoring", "dashboard_enabled", "false")
    set_ini(ini, "monitoring", "woodyminer_enabled", "false")
    set_ini(ini, "efficiency", "desktop_headroom_pct", "5")
    set_ini(ini, "cuda", "max_lanes", "16")

    lanes = [int(x) for x in args.lanes.split(",") if x.strip()]
    vrams = [float(x) for x in args.vram.split(",") if x.strip()]
    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    total = len(lanes) * len(vrams) * len(batches)
    mins = total * args.seconds / 60

    print(f"\nGPU {args.gpu} · {total} configurations × {args.seconds}s ≈ {mins:.0f} min")
    print("Other cards keep mining untouched.\n")

    # Lanes come from reference//difficulty, so read the live difficulty once.
    probe = measure(run_dir, args.gpu, 60)
    net_diff = probe["difficulty"] or 1100
    print(f"Network difficulty: {net_diff}   baseline {probe['hps']:,.0f} H/s "
          f"(plan {probe['plan']})\n")

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
                r = measure(run_dir, args.gpu, args.seconds)
                r.update(label=label, want_lanes=want_lanes, vram=vram, batch=batch)
                results.append(r)
                print(f"    {r['hps']:,.0f} H/s   plan {r['plan']}   "
                      f"diff {r['difficulty']}   rc={r['rc']}")
                if r["difficulty"] and r["difficulty"] != net_diff:
                    print(f"    ! network difficulty moved {net_diff} -> "
                          f"{r['difficulty']} - results not comparable")

    results.sort(key=lambda r: r["hps"], reverse=True)
    best = results[0]["hps"] if results else 0
    print("\n" + "=" * 78)
    print(f"{'H/s':>12}  {'vs best':>8}  {'plan':>14}  configuration")
    print("-" * 78)
    for r in results:
        rel = f"{r['hps'] / best * 100:.0f}%" if best else "—"
        print(f"{r['hps']:>12,.0f}  {rel:>8}  {r['plan']:>14}  {r['label']}")
    print("=" * 78)
    if results:
        b = results[0]
        print(f"\nBest: {b['label']}  ->  {b['hps']:,.0f} H/s")
        print("Apply to every card by editing /root/xnminer-gpu*/miner.ini:")
        print(f"  [cuda] vram_reference_difficulty = {net_diff * b['want_lanes']}")
        print(f"  [cuda] max_lanes = 16")
        print(f"  [cuda] max_batch_size = {b['batch']}")
        print(f"  [efficiency] target_vram_pct = {b['vram']}")
        print("  [efficiency] desktop_headroom_pct = 5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
