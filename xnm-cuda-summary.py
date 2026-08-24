#!/usr/bin/env python3
"""One screen for every per-GPU xnminer instance.

Reads what each miner already writes and pairs it with live nvidia-smi:
  data/session_timelapse.jsonl    hashrate, queue depth, network state
  data/mining_stats_history.json  accepted blocks per type, per mining day
Network difficulty is polled separately, in a background thread.

Standard library only: the C++ miner pulls in no Python, so nothing here may.

  python3 xnm-cuda-summary.py [glob]     default: /root/xnm-gpu*
"""
from __future__ import annotations

import glob as globmod
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timedelta

PATTERN = sys.argv[1] if len(sys.argv) > 1 else "/root/xnm-gpu*"
REFRESH_S = 3.0
STALE_AFTER_S = 90.0          # samples land every 30s
RESTART_GAP_S = 300.0        # a bigger hole between samples means a restart
QUEUE_SCAN_S = 30.0          # blocks.db parse is slow on a big queue
DIFF_POLL_S = 20.0

POOL_DIFF = "http://xenblocks.io/difficulty"
POOL_HTTPS = "https://xenblocks.io/v1/leaderboard"

C = {
    "off": "\033[0m", "dim": "\033[38;2;170;170;170m", "b": "\033[1m",
    "cyan": "\033[96m", "green": "\033[92m", "yellow": "\033[93m",
    "red": "\033[91m", "white": "\033[97m", "celadon": "\033[38;2;172;225;175m",
}

NET = {"difficulty": None, "source": "", "at": 0.0}
QUEUE_MIX = {"at": 0.0, "counts": {"XNM": 0, "XBLK": 0, "XUNI": 0}}
_lock = threading.Lock()


def paint(text: str, *styles: str) -> str:
    return "".join(C[s] for s in styles) + text + C["off"]


def plain(text: str) -> str:
    return re.sub(r"\033\[[0-9;]*m", "", text)


# --- network difficulty -----------------------------------------------------
# Port 80 is the mining API and can be slow or flaky depending on the host; the
# HTTPS leaderboard carries the same number and stays up. Never block the UI on
# either - a background thread refreshes, the screen reads the last value.
def _fetch_difficulty() -> tuple[int | None, str]:
    for url, key, label in ((POOL_DIFF, "difficulty", "pool"),
                            (POOL_HTTPS, "difficulty", "https")):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "xnm-summary"})
            with urllib.request.urlopen(req, timeout=6) as r:
                data = json.loads(r.read().decode("utf-8", "replace"))
            value = int(data.get(key, data.get("diff", 0)))
            if value:
                return value, label
        except Exception:
            continue
    return None, ""


def _difficulty_worker() -> None:
    while True:
        value, source = _fetch_difficulty()
        if value:
            with _lock:
                NET.update(difficulty=value, source=source, at=time.time())
        time.sleep(DIFF_POLL_S)


# --- per-miner files --------------------------------------------------------
def gpu_index_of(name: str) -> int:
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else 0


def last_sample(path: str) -> dict:
    """Newest timelapse sample.

    The file interleaves two shapes: samples (hps, accepted, queued, ...) and
    one-line events ({"event": "...", "ts": ...}). The final line is usually an
    event, so skip anything without "hps" or every row reads as zeros.
    """
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(max(0, size - 32768))
            chunk = f.read().decode("utf-8", "replace")
    except OSError:
        return {}
    for line in reversed(chunk.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "hps" in rec:
            return rec
    return {}


def run_start_ts(path: str) -> float:
    """Start of the CURRENT run, not the age of the file.

    timelapse.cpp opens the file with ios::app and never truncates it, so the
    first line survives every restart - reading it reported "Up 8h36m" for a
    miner that had been restarted minutes earlier. Walk the samples instead and
    take the last point preceded by a restart-sized hole. Only the tail is read,
    so a run older than that window reads as starting at the window edge.
    """
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(max(0, size - 4_000_000))
            chunk = f.read().decode("utf-8", "replace")
    except OSError:
        return 0.0
    start = prev = 0.0
    for line in chunk.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue          # first line of the window is usually a partial
        if "hps" not in rec:
            continue
        ts = parse_ts(rec.get("ts"))
        if not ts:
            continue
        if not start or (prev and ts - prev > RESTART_GAP_S):
            start = ts
        prev = ts
    return start


def queue_breakdown(dirs) -> dict:
    """What is still WAITING, by type, across every card.

    The XNM/XBLK columns count blocks the pool already accepted today, so a
    fresh instance shows zeros there while the queue fills up - which reads as
    if the queued blocks had no type. They do: blocks.db stores block_type per
    entry, so nothing has to be re-classified here.

    Parsing that file costs real time once the queue is tens of thousands of
    blocks, so refresh it on a slow timer rather than on every 3s repaint. The
    mix barely moves between frames; the trade is that this line can lag the
    Queue column slightly, since the db is written less often than samples.
    """
    now = time.time()
    if now - QUEUE_MIX["at"] < QUEUE_SCAN_S:
        return QUEUE_MIX["counts"]
    counts = {"XNM": 0, "XBLK": 0, "XUNI": 0}
    for d in dirs:
        try:
            with open(os.path.join(d, "data", "blocks.db"), "r",
                      encoding="utf-8", errors="replace") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        for item in data.get("pending") or []:
            kind = item.get("block_type") or "XNM"
            if kind not in ("XUNI", "XBLK"):
                kind = "XNM"          # same bucketing as BlockStore::pending_by_type
            counts[kind] += 1
    QUEUE_MIX["at"] = now
    QUEUE_MIX["counts"] = counts
    return counts


def mining_day() -> str:
    """The miner rolls its day at 01:00 local, so before 1am we are still in
    yesterday's bucket."""
    now = datetime.now()
    if now.hour < 1:
        now -= timedelta(days=1)
    return now.strftime("%Y-%m-%d")


def today_counts(path: str) -> dict:
    """{"XNM": n, "XUNI": n, "XBLK": n} accepted today, per miner."""
    empty = {"XNM": 0, "XUNI": 0, "XBLK": 0}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return empty
    day = data.get(mining_day())
    if not isinstance(day, dict):
        return empty
    return {k: int(day.get(k, 0) or 0) for k in empty}


def parse_ts(value) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except ValueError:
        return 0.0


def nvidia_smi() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,utilization.gpu,temperature.gpu,power.draw,"
             "memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    cards = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            cards[int(parts[0])] = {
                "name": parts[1].replace("NVIDIA ", "").replace("GeForce ", ""),
                "util": int(float(parts[2])),
                "temp": int(float(parts[3])),
                "power": float(parts[4]),
                "vram_used": int(float(parts[5])),
                "vram_total": int(float(parts[6])),
            }
        except ValueError:
            continue
    return cards


# --- rendering --------------------------------------------------------------
def hps_fmt(v: float) -> str:
    if v >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if v >= 1000:
        return f"{v / 1000:.1f}k"
    return f"{v:.0f}"


def dur(seconds: float) -> str:
    s = int(max(0, seconds))
    d, rem = divmod(s, 86400)
    h, rem = divmod(rem, 3600)
    m, _ = divmod(rem, 60)
    if d:
        return f"{d}d{h:02d}h"
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m"


COLS = [("GPU", 4), ("Card", 12), ("Up", 7), ("H/s", 8), ("Util", 5),
        ("Temp", 5), ("Mem", 5), ("Power", 6), ("VRAM", 10),
        ("XNM", 6), ("XBLK", 6), ("Queue", 6), ("Net", 8)]
RIGHT = {"GPU", "Up", "H/s", "Util", "Temp", "Mem", "Power", "VRAM",
         "XNM", "XBLK", "Queue"}


def row(cells) -> str:
    out = []
    for (label, width), cell in zip(COLS, cells):
        text = str(cell)
        pad = " " * max(0, width - len(plain(text)))
        out.append((pad + text) if label in RIGHT else (text + pad))
    return "  ".join(out)


def render(dirs, cards) -> str:
    now = time.time()
    tot_hps = tot_power = 0.0
    tot = {"XNM": 0, "XUNI": 0, "XBLK": 0}
    tot_queue = live = 0
    longest_up = 0.0
    body = []

    for d in dirs:
        idx = gpu_index_of(os.path.basename(d))
        data_dir = os.path.join(d, "data")
        s = last_sample(os.path.join(data_dir, "session_timelapse.jsonl"))
        blocks = today_counts(os.path.join(data_dir, "mining_stats_history.json"))
        nv = cards.get(idx, {})

        age = now - parse_ts(s.get("ts"))
        fresh = bool(s) and age < STALE_AFTER_S
        hps = float(s.get("hps") or 0.0) if fresh else 0.0
        if fresh:
            live += 1
            tot_hps += hps
        for k in tot:
            tot[k] += blocks[k]
        queued = int(s.get("queued") or 0)
        tot_queue += queued
        tot_power += nv.get("power", 0.0)

        up = (now - run_start_ts(os.path.join(data_dir, "session_timelapse.jsonl"))) if s else 0.0
        longest_up = max(longest_up, up)

        if not s:
            net = paint("-", "dim")
        elif not fresh:
            net = paint("stale", "yellow", "b")
        elif s.get("network_ok"):
            net = paint("online", "green")
        else:
            net = paint("offline", "red", "b")

        temp = nv.get("temp", s.get("temp_c", 0))
        mem_j = s.get("memory_junction_c")
        vram = (f"{nv['vram_used'] / 1024:.1f}/{nv['vram_total'] / 1024:.0f}G"
                if "vram_total" in nv else "-")

        body.append(row([
            paint(str(idx), "b"),
            nv.get("name", "?")[:12],
            dur(up) if s else paint("-", "dim"),
            paint(hps_fmt(hps), "cyan") if fresh else paint("-", "dim"),
            f"{nv['util']}%" if "util" in nv else "-",
            paint(f"{temp}C", "red", "b") if temp and temp >= 84 else f"{temp}C",
            f"{mem_j}C" if mem_j else paint("-", "dim"),
            f"{nv['power']:.0f}W" if "power" in nv else "-",
            vram,
            paint(str(blocks["XNM"]), "green") if blocks["XNM"] else "0",
            paint(str(blocks["XBLK"]), "cyan") if blocks["XBLK"] else "0",
            paint(str(queued), "yellow") if queued else "0",
            net,
        ]))

    head = "  ".join((label.rjust(w) if label in RIGHT else label.ljust(w))
                     for label, w in COLS)
    rule = paint("-" * len(plain(head)), "dim")

    with _lock:
        diff, source, at = NET["difficulty"], NET["source"], NET["at"]
    if diff:
        stamp = "" if source == "pool" else paint(f" ({source})", "dim")
        diff_txt = paint(f"siec m={diff}", "b", "celadon") + stamp
    elif at:
        diff_txt = paint("siec: brak odpowiedzi", "red", "b")
    else:
        diff_txt = paint("siec ...", "dim")

    lines = [
        (paint("XenBlocks", "b", "white") + "  " +
         paint("wszystkie karty", "dim") + "   " +
         paint(f"{live}/{len(dirs)} kopie", "b", "celadon") + "   " +
         diff_txt + "   " +
         paint(f"praca {dur(longest_up)}", "white") + "   " +
         paint(time.strftime("%H:%M:%S"), "dim")),
        "",
        paint(head, "b", "cyan"),
        rule,
    ]
    lines.extend(body)
    lines.append(rule)
    lines.append(row([
        paint("ALL", "b"), "", dur(longest_up),
        paint(hps_fmt(tot_hps), "b", "cyan"), "", "", "",
        paint(f"{tot_power:.0f}W", "b") if tot_power else "",
        "",
        paint(str(tot["XNM"]), "b", "green"),
        paint(str(tot["XBLK"]), "b", "cyan"),
        paint(str(tot_queue), "b", "yellow") if tot_queue else "0",
        "",
    ]))
    lines.append("")
    lines.append(paint(
        f"Dzis przyjete: XNM {tot['XNM']}  XBLK {tot['XBLK']}  XUNI {tot['XUNI']}"
        f"   (doba liczona od 01:00)", "dim"))
    if tot_queue:
        q = queue_breakdown(dirs)
        lines.append(paint(
            f"W kolejce:     XNM {q['XNM']}  XBLK {q['XBLK']}  XUNI {q['XUNI']}"
            f"   (odswiezane co {int(QUEUE_SCAN_S)} s)", "dim"))
        lines.append(paint(
            f"Kolejka {tot_queue} blok(ow) czeka na okno m=100 - nie kasuj instancji, "
            "dopoki nie zejdzie do zera.", "yellow"))
    lines.append(paint(
        "Ctrl+B potem cyfra = karta   |   Ctrl+C zamyka TYLKO ten podglad, "
        "nie minery", "dim"))
    return "\n".join(lines)


def main() -> int:
    threading.Thread(target=_difficulty_worker, daemon=True).start()
    try:
        while True:
            dirs = sorted(
                (d for d in globmod.glob(PATTERN) if os.path.isdir(d)),
                key=lambda d: gpu_index_of(os.path.basename(d)),
            )
            screen = (render(dirs, nvidia_smi()) if dirs
                      else paint(f"Brak katalogow pasujacych do {PATTERN}", "red", "b"))
            sys.stdout.write("\033[H\033[J" + screen + "\n")
            sys.stdout.flush()
            time.sleep(REFRESH_S)
    except KeyboardInterrupt:
        sys.stdout.write("\033[?25h\n")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
