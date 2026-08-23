#!/usr/bin/env bash
# badnob/xnminer-linux-test (pure C++/CUDA) - multi-GPU bootstrap for Vast.ai
#
#   ./vast-xnminer-cuda.sh 0xYourWallet
#
# Measured 2026-08-22 on an RTX 3090 at difficulty 100: 3.12 MH/s, 100% GPU
# util - against ~1.09 MH/s for the Python miner on the same card and ~265 kH/s
# in 4-card production. The Python loop between batches was the bottleneck;
# this build has none.
#
# The miner mines at m=100 permanently (force_mine_memory_cost) and queues hits
# until the network matches, so the difficulty proxy we used before is no
# longer needed.
#
# Env overrides:
#   XNM_MAX_GPUS=8      cap on cards used
#   XNM_VRAM_PCT=79.0   VRAM cap per card
#   XNM_LANES=8         CUDA lanes per card
#   XNM_REPO=<url>      source repository
set -euo pipefail

WALLET="${1:-${MINER_ADDR:-0x9d79B1921b75AC7C199314406f5398E15f2fb47C}}"
REPO="${XNM_REPO:-https://github.com/badnob/xnminer-linux-test.git}"
BASE="${XNM_BASE:-/root/xnminer-cuda}"
SESSION="xnm"
MAX_GPUS="${XNM_MAX_GPUS:-8}"
VRAM_PCT="${XNM_VRAM_PCT:-79.0}"
LANES="${XNM_LANES:-8}"

log() { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

[[ "$WALLET" =~ ^0x[0-9a-fA-F]{40}$ ]] || die "not a valid EVM address: $WALLET"

export DEBIAN_FRONTEND=noninteractive
export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# --- 1. build dependencies --------------------------------------------------
# install-deps.sh in the repo calls sudo, which bare nvidia/cuda images do not
# have. Install the same packages directly as root instead.
if ! command -v cmake >/dev/null 2>&1 || ! command -v ninja >/dev/null 2>&1 \
   || ! command -v git >/dev/null 2>&1 || ! command -v tmux >/dev/null 2>&1; then
  log "Installing build dependencies"
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends \
    git build-essential cmake ninja-build pkg-config \
    libcurl4-openssl-dev ca-certificates tmux >/dev/null
fi
command -v nvcc >/dev/null 2>&1 \
  || die "nvcc missing - use an image tagged '-devel', not '-base' or '-runtime'"

# --- 2. how many cards ------------------------------------------------------
DETECTED="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
[ "${DETECTED:-0}" -ge 1 ] || die "nvidia-smi reports no GPUs"
GPUS="$DETECTED"
if [ "$GPUS" -gt "$MAX_GPUS" ]; then
  GPUS="$MAX_GPUS"
  log "Detected ${DETECTED} GPUs - mining on the first ${GPUS} (raise with XNM_MAX_GPUS)"
else
  log "Detected ${GPUS} GPU(s) - mining on all of them"
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# --- 3. source + build once -------------------------------------------------
if [ -d "$BASE/.git" ]; then
  log "Source present - updating"
  git -C "$BASE" pull --ff-only >/dev/null 2>&1 || log "pull skipped (local changes?)"
else
  log "Cloning ${REPO}"
  git clone --depth 1 "$REPO" "$BASE"
fi
cd "$BASE"
chmod +x ./*.sh

# --- 3b. local patches ------------------------------------------------------
# 1. XBLK before XNM. Upstream flushes XNM > XBLK > XUNI, and list_flush_batch
#    reads the queue head-first and stops at the wave cap, so a few XBLK behind
#    thousands of XNM never enter a wave. Flips the priority AND the selection.
# 2. One dropped connection no longer aborts the whole wave. Measured on a Vast
#    host losing ~40% of connections to port 80: waves of 1024 delivered 61-127
#    blocks because the FIRST drop killed the rest. Now only a fully lost round
#    counts as "pool down"; partial loss just re-queues those blocks.
# 3. Submit timeout capped at 6s, separate from the difficulty poll - a round is
#    only as fast as its slowest member, and a blackholed socket held it for 12s.
# Set XNM_XBLK_FIRST=0 to build stock upstream instead.
if [ "${XNM_XBLK_FIRST:-1}" = "1" ]; then
  cat > /root/xblk-first.py <<'XBLKEOF'
import pathlib, sys
root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")

def swap(rel, marker, old, new):
    p = root / rel
    s = p.read_text(encoding="utf-8")
    if marker in s:
        print("already patched:", rel, marker); return
    if s.count(old) != 1:
        raise SystemExit("no unique match for %s in %s (found %d)" % (marker, rel, s.count(old)))
    p.write_text(s.replace(old, new, 1), encoding="utf-8", newline="\n")
    print("patched:", rel, marker)

# --- 1. XBLK before XNM ------------------------------------------------------
swap("src/queue/policy.cpp", "XBLK_FIRST_PATCH", """    // Priority: XNM > XBLK > XUNI (lower = flush sooner).
    if (kind == "XNM" || kind == "XEN11" || kind == "NORMAL") return 0;
    if (kind == "XBLK") return 1;
    if (kind == "XUNI") return 2;""", """    // XBLK_FIRST_PATCH - priority: XBLK > XNM > XUNI (lower = flush sooner).
    if (kind == "XBLK") return 0;
    if (kind == "XNM" || kind == "XEN11" || kind == "NORMAL") return 1;
    if (kind == "XUNI") return 2;""")

swap("src/queue/store.cpp", "XBLK_FIRST_DECL", """    std::vector<PendingBlock> preferred;
    std::vector<PendingBlock> wrapped;
    preferred.reserve(static_cast<size_t>(std::max(cap, 0)));
    wrapped.reserve(static_cast<size_t>(std::max(cap, 0)));""", """    // XBLK_FIRST_DECL
    std::vector<PendingBlock> preferred;
    preferred.reserve(static_cast<size_t>(std::max(cap, 0)));""")

swap("src/queue/store.cpp", "XBLK_FIRST_PATCH", """    for (const auto& pb : pending_) {
        if (skip_before_id > 0 && pb.id < skip_before_id) take(wrapped, pb);
        else take(preferred, pb);
        if (static_cast<int>(preferred.size()) >= cap) break;
    }
    if (static_cast<int>(preferred.size()) < cap) {
        for (auto& pb : wrapped) {
            preferred.push_back(std::move(pb));
            if (static_cast<int>(preferred.size()) >= cap) break;
        }
    }""", """    // XBLK_FIRST_PATCH - bucket the whole queue by flush_priority instead of
    // reading pending_ head-first. Upstream stopped at cap, so a few XBLK behind
    // thousands of XNM never entered a wave at all. Within a class the not-yet-
    // tried still go before the rotated-past (skip_before_id) ones.
    constexpr int kClasses = 4;
    std::vector<PendingBlock> fresh[kClasses];
    std::vector<PendingBlock> rotated[kClasses];
    for (const auto& pb : pending_) {
        int cls = flush_priority(pb.hit.block_type);
        if (cls < 0 || cls >= kClasses) cls = kClasses - 1;
        if (skip_before_id > 0 && pb.id < skip_before_id) take(rotated[cls], pb);
        else take(fresh[cls], pb);
    }
    for (int cls = 0; cls < kClasses && static_cast<int>(preferred.size()) < cap; ++cls) {
        for (auto& pb : fresh[cls]) {
            if (static_cast<int>(preferred.size()) >= cap) break;
            preferred.push_back(std::move(pb));
        }
        for (auto& pb : rotated[cls]) {
            if (static_cast<int>(preferred.size()) >= cap) break;
            preferred.push_back(std::move(pb));
        }
    }""")

# --- 2. one dropped connection must not abort the whole wave -----------------
swap("src/app/supervisor.cpp", "WAVE_TOLERANCE_DECL", """    int flushed = 0;
    int holds = 0;
    int last_hold_status = 0;
    std::string last_hold_hint;
    bool transport_fail = false;""", """    int flushed = 0;
    int holds = 0;
    int last_hold_status = 0;
    std::string last_hold_hint;
    bool transport_fail = false;
    // WAVE_TOLERANCE_DECL - a lossy link drops individual connections; upstream
    // aborted the whole 1024-block wave on the FIRST one, so a ~40% drop rate
    // delivered 61-127 blocks per wave instead of 1024. Count drops instead and
    // only give up when a full round is lost, which is what a dead pool looks like.
    int transport_fails = 0;
    int last_transport_status = 0;
    size_t round_fails = 0;""")

swap("src/app/supervisor.cpp", "WAVE_TOLERANCE_ROUND", """        const size_t n = std::min(static_cast<size_t>(parallel), work.size() - base);
        std::vector<std::future<SubmitResult>> futs;""", """        const size_t n = std::min(static_cast<size_t>(parallel), work.size() - base);
        round_fails = 0;                      // WAVE_TOLERANCE_ROUND
        const int flushed_before_round = flushed;
        std::vector<std::future<SubmitResult>> futs;""")

swap("src/app/supervisor.cpp", "WAVE_TOLERANCE_COUNT", """                if (!on_shutdown) {
                    note_submit_transport_failure("Queue flush", result.status);
                    transport_fail = true;
                }""", """                if (!on_shutdown) {       // WAVE_TOLERANCE_COUNT
                    ++transport_fails;
                    ++round_fails;
                    last_transport_status = result.status;
                }""")

swap("src/app/supervisor.cpp", "WAVE_TOLERANCE_ABORT", """        if (!accepted_ids.empty()) store_->mark_submitted_many(accepted_ids);
    }""", """        if (!accepted_ids.empty()) store_->mark_submitted_many(accepted_ids);
        // WAVE_TOLERANCE_ABORT - whole round lost and nothing accepted: treat as
        // pool down (backoff, keep the bag). Partial loss only costs those blocks
        // a later retry - they were never marked submitted, so they stay queued.
        if (!on_shutdown && round_fails == n && flushed == flushed_before_round) {
            note_submit_transport_failure("Queue flush", last_transport_status);
            transport_fail = true;
        }
    }""")

swap("src/app/supervisor.cpp", "WAVE_TOLERANCE_LOG", """        log("info", "Submitted " + std::to_string(flushed) + " queued block(s) (" + context +
                        ", CPU submit x" + std::to_string(parallel) + ")");""", """        log("info", "Submitted " + std::to_string(flushed) + " queued block(s) (" + context +
                        ", CPU submit x" + std::to_string(parallel) +
                        (transport_fails ? ", " + std::to_string(transport_fails) + " dropped"
                                         : std::string()) +   // WAVE_TOLERANCE_LOG
                        ")");""")

# --- 3. submit timeout separate from the difficulty poll ---------------------
swap("src/app/supervisor.cpp", "SUBMIT_TIMEOUT_PATCH", """    // CPU submit path: allow full configured timeout (does not block CUDA).
    return settings_.connection_timeout_s > 0 ? settings_.connection_timeout_s : 8;""", """    // SUBMIT_TIMEOUT_PATCH - a round of `parallel` submits is only as fast as its
    // slowest member, so one blackholed connection used to hold a whole round for
    // the full 12s. Cap the submit path; the difficulty poll keeps the long one.
    const int cfg = settings_.connection_timeout_s > 0 ? settings_.connection_timeout_s : 8;
    return std::min(cfg, 6);""")
XBLKEOF
  # Fail loud: if upstream ever renames these blocks the patch must not silently
  # no-op into a stock build that still sends XNM first.
  XBLK_OUT="$(python3 /root/xblk-first.py "$BASE" 2>&1)"     || die "local patch failed (upstream source changed?): ${XBLK_OUT} - re-run with XNM_XBLK_FIRST=0 to build stock"
  if printf '%s' "$XBLK_OUT" | grep -q "^patched:"; then
    log "Local patches applied - forcing rebuild"
    rm -f "$BASE/build/bin/xnminer"
  else
    log "Local patches already applied"
  fi
fi

# build.sh reads the real compute_cap from nvidia-smi itself, so no override.
if [ ! -x "$BASE/build/bin/xnminer" ]; then
  log "Building (one-off, a few minutes)"
  ./build.sh
fi
BIN="$(find "$BASE" -maxdepth 3 -type f -name xnminer -perm -u+x 2>/dev/null | head -1)"
[ -n "$BIN" ] || die "build produced no xnminer binary - check the output above"
log "Binary: ${BIN}"

# --- 4. one copy per card ---------------------------------------------------
# device_id lives in miner.ini, so each card needs its own directory and its
# own data/ (queue, lock, logs).
log "Preparing ${GPUS} instance(s)"
for i in $(seq 0 $((GPUS - 1))); do
  DIR="/root/xnm-gpu${i}"
  mkdir -p "$DIR/data"
  cp -f "$BIN" "$DIR/xnminer"
  [ -f "$DIR/miner.ini" ] || cp "$BASE/miner.ini.example" "$DIR/miner.ini"
  cd "$DIR"
  sed -i "s|^address =.*|address = ${WALLET}|"                 miner.ini
  sed -i "s|^worker =.*|worker = xnm-gpu${i}|"                 miner.ini
  sed -i "s|^device_id =.*|device_id = ${i}|"                  miner.ini
  sed -i "s|^target_vram_pct =.*|target_vram_pct = ${VRAM_PCT}|" miner.ini
  sed -i "s|^max_lanes =.*|max_lanes = ${LANES}|"              miner.ini
  sed -i "s|^woodyminer_custom_name =.*|woodyminer_custom_name = xnm-gpu${i}|" miner.ini
  echo "  GPU${i} -> ${DIR}"
  cd "$BASE"
done

# --- 4b. summary view -------------------------------------------------------
# Reads each miner's own data/session_timelapse.jsonl plus live nvidia-smi.
# Standard library only - the C++ miner pulls in no Python packages.
log "Writing summary view"
cat > "${BASE}/xnm-cuda-summary.py" <<'PYSUMEOF'
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
DIFF_POLL_S = 20.0

POOL_DIFF = "http://xenblocks.io/difficulty"
POOL_HTTPS = "https://xenblocks.io/v1/leaderboard"

C = {
    "off": "\033[0m", "dim": "\033[38;2;170;170;170m", "b": "\033[1m",
    "cyan": "\033[96m", "green": "\033[92m", "yellow": "\033[93m",
    "red": "\033[91m", "white": "\033[97m", "celadon": "\033[38;2;172;225;175m",
}

NET = {"difficulty": None, "source": "", "at": 0.0}
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
PYSUMEOF
chmod +x "${BASE}/xnm-cuda-summary.py"
command -v python3 >/dev/null 2>&1 || apt-get install -y -qq python3 >/dev/null 2>&1 || true

# --- 5. launch --------------------------------------------------------------
if [ -n "${TMUX:-}" ] && [ "$(tmux display-message -p '#S' 2>/dev/null)" = "$SESSION" ]; then
  die "You are inside tmux session '$SESSION'. Detach (Ctrl+B then D) and re-run."
fi
tmux kill-session -t "$SESSION" 2>/dev/null || true

for i in $(seq 0 $((GPUS - 1))); do
  CMD="cd /root/xnm-gpu${i} && ./xnminer; exec bash"
  if [ "$i" = "0" ]; then
    tmux new-session -d -s "$SESSION" -n "gpu0" "$CMD"
  else
    tmux new-window -t "$SESSION" -n "gpu${i}" "$CMD"
  fi
done

# Summary window last, so windows 0..N-1 keep matching the GPU numbers.
tmux new-window -t "$SESSION" -n "all"   "cd ${BASE} && python3 xnm-cuda-summary.py '/root/xnm-gpu*'; exec bash"
tmux select-window -t "${SESSION}:all"

log "${GPUS} miner(s) + summary running in tmux session '${SESSION}'"
echo "  summary:      Ctrl+B then ${GPUS}"
echo "  switch card:  Ctrl+B then 0 .. $((GPUS - 1))"
echo "  detach:       Ctrl+B then D"
echo "  re-attach:    tmux attach -t ${SESSION}"
echo

sleep 2
if [ -n "${TMUX:-}" ]; then
  exec tmux switch-client -t "$SESSION"
elif [ -t 1 ]; then
  exec tmux attach -t "$SESSION"
else
  echo "Started headless."
fi
