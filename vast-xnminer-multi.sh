#!/usr/bin/env bash
# XenBlocks miner (badnob/xnminer-linux) - MULTI-GPU bootstrap for Vast.ai
# Image: nvidia/cuda:*-devel-ubuntu*
#
#   ./vast-xnminer-multi.sh 0xYourWallet
#
# Upstream is hardwired to GPU 0. This builds the engine once, then runs one
# miner per card: CUDA_VISIBLE_DEVICES pins the compute, XNM_GPU (added by the
# patch below) points NVML monitoring at the same physical card.
#
# Each miner gets its own directory, miner.ini, worker name and tmux window.
set -euo pipefail

WALLET="${1:-${MINER_ADDR:-0x9d79B1921b75AC7C199314406f5398E15f2fb47C}}"
BASE="${XNM_BASE:-/root/xnminer-base}"
SRC_TARBALL="https://github.com/badnob/xnminer-linux/archive/refs/heads/main.tar.gz"
SESSION="xnm"
WARN_TEMP="${XNM_WARN_TEMP:-78}"
MAX_TEMP="${XNM_MAX_TEMP:-82}"
# Measured on an RTX 3090, 2026-08-17. Lanes = vram_reference_difficulty //
# difficulty, capped by max_lanes. Reference 800 gives 1 lane at difficulty 1100
# and 8 lanes at difficulty 100 - the measured optimum at both, and the network
# only ever sits at one of those two values.
#   difficulty 100 : 4 lanes 912k | 8 lanes 1,087k | 16 lanes 1,068k | auto 1,010k
#   difficulty 1100: every lane/VRAM combination lands within 9% of stock, and
#                    stock wins - so nothing else here is worth overriding.
# Set explicitly: the built-in default is memory_cost, so editing memory_cost
# would silently collapse the lane count to 1.
MAX_LANES="${XNM_MAX_LANES:-8}"
LANE_REF="${XNM_LANE_REF:-800}"
# Stock 69.09 measured best at difficulty 1100 (92% was 4% slower). At
# difficulty 100 the bottleneck moves to the single Python thread feeding the
# card, where a bigger batch means fewer round trips - untested, so it stays an
# override rather than a default:  XNM_VRAM_PCT=92 ./vast-xnminer-multi.sh 0x...
VRAM_PCT="${XNM_VRAM_PCT:-69.09}"
# Local difficulty proxy. xenblocks.io drops port 80 for long stretches; the
# miner then falls back to memory_cost and mines at the wrong difficulty - every
# block it finds is held "until difficulty matches", and at m=1100 while the
# network is at m=100 it also runs ~6x slower. The same domain still serves the
# live difficulty over HTTPS at /v1/leaderboard, so proxy that.
DIFF_PORT="${XNM_DIFF_PORT:-8899}"

log() { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

[[ "$WALLET" =~ ^0x[0-9a-fA-F]{40}$ ]] || die "not a valid EVM address: $WALLET"

export DEBIAN_FRONTEND=noninteractive
export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# --- 1. packages -----------------------------------------------------------
if ! command -v python3 >/dev/null 2>&1 || ! command -v cmake >/dev/null 2>&1 \
   || ! command -v tmux >/dev/null 2>&1; then
  log "Installing python3 / pip / cmake / tmux"
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends \
    python3 python3-pip cmake build-essential ca-certificates wget tmux >/dev/null
fi

# --- 2. how many cards -----------------------------------------------------
# One miner per card, however many are present. Capped so a huge host cannot
# spawn more windows than tmux digit shortcuts can reach.
MAX_GPUS="${XNM_MAX_GPUS:-8}"
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

# --- 3. source -------------------------------------------------------------
if [ -f "$BASE/main.py" ]; then
  log "Base source already present - reusing"
else
  log "Fetching miner source"
  mkdir -p "$BASE"
  wget -qO /tmp/xnminer-src.tar.gz "$SRC_TARBALL"
  tar -xzf /tmp/xnminer-src.tar.gz -C "$BASE" --strip-components=1
fi
cd "$BASE"

# --- 4. patches ------------------------------------------------------------
log "Patching upstream (6 fixes)"
python3 - <<'PYEOF'
import pathlib, sys

def edit(path, pairs, marker):
    # Explicit utf-8: some sources carry non-ASCII characters and a container
    # with a non-UTF-8 locale would otherwise fail to read them.
    p = pathlib.Path(path)
    s = p.read_text(encoding="utf-8")
    if marker in s:
        print(f"  {path}: already patched")
        return
    for old, new in pairs:
        if old not in s:
            sys.exit(f"ERROR: pattern not found in {path}:\n  {old}")
        s = s.replace(old, new, 1)
    p.write_text(s, encoding="utf-8")
    print(f"  {path}: patched")

# 1) verify_known_block() hashes the reference log0.txt key with OUR wallet as
#    the Argon2 salt, but that block was mined with its author's salt. Different
#    salt -> different digest, so "XEN11" never appears and startup aborts for
#    everyone except the original miner.
edit("core/supervisor.py", [(
    'if not diag["calibration_m100"]:',
    'if False:  # self-test disabled - known block uses a foreign salt',
)], "self-test disabled")

# 2) LaneSlot holds a std::mutex - neither copyable nor movable - yet the lanes
#    live in a std::vector that push_back()s and resize()s them. No conforming
#    compiler accepts that.
edit("native/engine/xen_cuda_api.cpp", [
    ("    std::mutex mutex;",
     "    std::unique_ptr<std::mutex> mutex = std::make_unique<std::mutex>();"),
    ("lane_lock(slot->mutex)", "lane_lock(*slot->mutex)"),
    ("lane_lock(slot->mutex)", "lane_lock(*slot->mutex)"),
], "unique_ptr<std::mutex> mutex")

# 3) NVML device index is hardwired to 0 everywhere, so every copy would report
#    card 0's VRAM and temperature. Take it from XNM_GPU instead.
#    NVML ignores CUDA_VISIBLE_DEVICES, so this needs the real physical index.
edit("monitoring/nvidia.py", [
    ("from __future__ import annotations",
     "from __future__ import annotations\n\nimport os"),
    ("        self.device_index = device_index",
     '        self.device_index = int(os.environ.get("XNM_GPU", device_index))'),
    ("nvmlDeviceGetHandleByIndex(device_index)",
     "nvmlDeviceGetHandleByIndex(self.device_index)"),
    ('f"NVML ready: GPU{device_index} {name}"',
     'f"NVML ready: GPU{self.device_index} {name}"'),
], 'os.environ.get("XNM_GPU"')

# 4) same for nvidia-smi power control
edit("efficiency/gpu_power.py", [
    ("from __future__ import annotations",
     "from __future__ import annotations\n\nimport os"),
    ("        self._device_index = device_index",
     '        self._device_index = int(os.environ.get("XNM_GPU", device_index))'),
], 'os.environ.get("XNM_GPU"')

# 5) woodyminer leaderboard id must differ per card, or the four uploads
#    overwrite each other as one machine.
edit("monitoring/woodyminer_stats.py", [
    ('    device_info = f"{device_index},"',
     '    import os as _os\n'
     '    device_info = f"{_os.environ.get(\'XNM_GPU\', device_index)},"'),
], "_os.environ.get")

# 6) fills_budget() allows 2 MiB of slack no matter how many lanes there are,
#    but every lane rounds its own batch down, so a 2-lane plan misses the
#    budget by ~3 MiB out of 20 GB and _plan_from_device raises
#    "CUDA harvest plan under-filled VRAM cap". That single line is why every
#    lanes>1 configuration died at startup. Needed for max_lanes to mean
#    anything at high difficulty; the win itself shows up at difficulty 100.
edit("mining/vram_batch.py", [
    ("return abs(self.batch_vram_mib - self.budget_mib) <= tolerance_mib",
     "return abs(self.batch_vram_mib - self.budget_mib) <= max(tolerance_mib, self.lanes * 4)"),
], "self.lanes * 4")
PYEOF

# --- 5. python deps --------------------------------------------------------
if ! python3 -c "import argon2, pynvml, psutil, rich" 2>/dev/null; then
  log "Python packages"
  # Ubuntu 24.04 marks system python as externally managed (PEP 668)
  python3 -m pip install --no-cache-dir -q -r requirements.txt 2>/dev/null \
    || python3 -m pip install --no-cache-dir -q --break-system-packages -r requirements.txt
fi

# --- 6. build the engine once ----------------------------------------------
SO="native/build/bin/libxen_cuda.so"
if [ -f "$SO" ]; then
  log "CUDA engine already built - skipping"
else
  # native/build.sh defaults to sm_90;120 only, which most rentable cards cannot
  # load. Build for every architecture present on this machine.
  ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
          | tr -d ' .' | sort -u | paste -sd';' -)"
  [ -n "$ARCH" ] || ARCH="${XNM_ARCH:-75;80;86;89;90}"
  log "Building engine for sm_{${ARCH//;/, }} - one-off, ~1-2 min"
  command -v nvcc >/dev/null 2>&1 \
    || die "nvcc missing - use an image tagged '-devel', not '-base' or '-runtime'"
  chmod +x native/build.sh
  CMAKE_CUDA_ARCHITECTURES="$ARCH" ./native/build.sh
fi
[ -f "$SO" ] || die "libxen_cuda.so missing - check the build output above"

# --- 7. one copy per card --------------------------------------------------
# Separate directories because each miner holds its own data/miner.lock.
log "Preparing ${GPUS} miner instance(s)"
for i in $(seq 0 $((GPUS - 1))); do
  DIR="/root/xnminer-gpu${i}"
  if [ ! -f "$DIR/main.py" ]; then
    mkdir -p "$DIR"
    cp -r "$BASE"/. "$DIR"/
    rm -rf "$DIR/data" "$DIR/native/XenblocksMiner-main"
  fi
  cd "$DIR"
  # A killed tmux server (or a dead instance) leaves data/miner.lock behind and
  # the next miner refuses to start with "Another miner instance is already
  # running". Nothing else holds it at this point, so clear it every time.
  rm -f "$DIR/data/miner.lock"
  [ -f miner.ini ] || cp miner.ini.example miner.ini
  sed -i "s|^address =.*|address = ${WALLET}|" miner.ini
  sed -i "s|^worker =.*|worker = xnminer-gpu${i}|" miner.ini
  sed -i "s|^woodyminer_custom_name =.*|woodyminer_custom_name = xnminer-gpu${i}|" miner.ini
  # Upstream defaults (72/75) are set for a quiet desktop. Rented 3090s sit at
  # 70-75C under load, so they spend their time derating batches and cooling
  # down instead of hashing. These are still well under the card's own limits.
  sed -i "s|^warn_gpu_temp_c.*|warn_gpu_temp_c = ${WARN_TEMP}|" miner.ini
  sed -i "s|^max_gpu_temp_c.*|max_gpu_temp_c = ${MAX_TEMP}|" miner.ini
  sed -i "s|^max_lanes.*|max_lanes = ${MAX_LANES}|" miner.ini
  sed -i "s|^target_vram_pct.*|target_vram_pct = ${VRAM_PCT}|" miner.ini
  sed -i "s|^base_url =.*|base_url = http://127.0.0.1:${DIFF_PORT}|" miner.ini
  if grep -q "^vram_reference_difficulty" miner.ini; then
    sed -i "s|^vram_reference_difficulty.*|vram_reference_difficulty = ${LANE_REF}|" miner.ini
  else
    sed -i "s|^\[cuda\]|[cuda]\nvram_reference_difficulty = ${LANE_REF}|" miner.ini
  fi
  echo "  GPU${i} -> ${DIR}"
  cd "$BASE"
done

# --- 7a. difficulty proxy ---------------------------------------------------
log "Starting difficulty proxy on 127.0.0.1:${DIFF_PORT}"
cat > "${BASE}/xnm-diffproxy.py" <<'PYDIFFEOF'
#!/usr/bin/env python3
"""Keep the miner on the network's real difficulty during pool outages.

xenblocks.io drops plain HTTP (port 80) for long stretches. When that happens
the miner cannot read /difficulty, falls back to [mining] memory_cost, and mines
at the wrong difficulty - every block it finds is then held "until difficulty
matches", and at m=1100 while the network is at m=100 it also runs ~6x slower.

The same domain still answers over HTTPS: /v1/leaderboard carries the live
difficulty. This proxy sits on 127.0.0.1 and serves the miner:

  GET  /difficulty -> the pool if it answers, otherwise the HTTPS leaderboard
  POST /verify     -> forwarded to the pool; on failure returns 503 so the
                      miner queues the block exactly as it normally would

Point the miner at it:  [server] base_url = http://127.0.0.1:8899

  python3 xnm-diffproxy.py            # foreground
  python3 xnm-diffproxy.py --port N   # different port
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

POOL = "http://xenblocks.io"
FALLBACK = "https://xenblocks.io/v1/leaderboard"
CACHE_S = 10.0

_state = {"difficulty": None, "source": "none", "at": 0.0}
_lock = threading.Lock()


def _fetch(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "xnm-diffproxy"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def refresh() -> tuple[int | None, str]:
    """Fetch the difficulty: the pool first, HTTPS leaderboard as fallback."""
    diff, source = None, "none"
    try:                                    # 1. the real pool, port 80
        data = json.loads(_fetch(f"{POOL}/difficulty", 4).decode("utf-8", "replace"))
        diff = int(data.get("difficulty", data.get("diff", 0))) or None
        source = "pool"
    except Exception:
        try:                                # 2. HTTPS leaderboard, same domain
            data = json.loads(_fetch(FALLBACK, 10).decode("utf-8", "replace"))
            diff = int(data.get("difficulty", 0)) or None
            source = "https-leaderboard"
        except Exception:
            pass

    if diff:
        with _lock:
            if _state["source"] != source:
                print(f"[diffproxy] difficulty {diff} via {source}", flush=True)
            _state.update(difficulty=diff, source=source, at=time.time())
        return diff, source

    with _lock:                             # 3. last known value beats nothing
        return _state["difficulty"], "stale"


def current_difficulty() -> tuple[int | None, str]:
    """Cached value, never blocking.

    The miner gives /difficulty only network_poll_timeout_s (3s by default)
    before declaring the network down. Probing the dead pool takes longer than
    that on its own, so the HTTP handler must never wait on the network: a
    background thread keeps the value fresh and requests are served from it.
    """
    with _lock:
        if _state["difficulty"]:
            return _state["difficulty"], _state["source"]
    return refresh()                        # first call only


def _refresher() -> None:
    while True:
        try:
            refresh()
        except Exception:
            pass
        time.sleep(CACHE_S)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def do_GET(self) -> None:
        if self.path.rstrip("/").endswith("difficulty"):
            diff, _ = current_difficulty()
            if diff:
                self._send(200, json.dumps({"difficulty": str(diff)}).encode())
            else:
                self._send(503, b'{"error":"difficulty unavailable"}')
            return
        try:                                 # anything else: pass through
            self._send(200, _fetch(POOL + self.path, 8))
        except Exception as exc:
            self._send(503, json.dumps({"error": str(exc)}).encode())

    def do_POST(self) -> None:
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        req = urllib.request.Request(
            POOL + self.path, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                self._send(resp.status, resp.read())
        except urllib.error.HTTPError as exc:
            # A real rejection from the pool - pass it through verbatim so the
            # miner can tell "duplicate" from "difficulty mismatch".
            self._send(exc.code, exc.read())
        except Exception as exc:
            # Pool unreachable: 503 makes the miner queue the block, its normal
            # path for a network outage.
            self._send(503, json.dumps({"error": str(exc)}).encode())

    def log_message(self, *args) -> None:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--once", action="store_true", help="print difficulty and exit")
    args = ap.parse_args()

    if args.once:
        diff, source = current_difficulty()
        print(f"difficulty={diff} source={source}")
        return 0 if diff else 1

    diff, source = current_difficulty()
    threading.Thread(target=_refresher, daemon=True).start()
    print(f"[diffproxy] listening on 127.0.0.1:{args.port} "
          f"(difficulty {diff} via {source})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PYDIFFEOF
pkill -f xnm-diffproxy.py 2>/dev/null || true
# Restart loop: if the miners lose this, they lose the real difficulty.
nohup bash -c "while true; do python3 '${BASE}/xnm-diffproxy.py' --port ${DIFF_PORT}; sleep 5; done" \n  >> /root/xnm-diffproxy.log 2>&1 &
sleep 2
python3 "${BASE}/xnm-diffproxy.py" --once || log "WARNING: no difficulty source reachable yet"

# --- 7b. aggregate view -----------------------------------------------------
# Reads the same files the per-GPU dashboards write, so the numbers agree.
log "Writing summary view"
cat > "${BASE}/xnm-summary.py" <<'PYSUMEOF'
#!/usr/bin/env python3
"""Aggregate view across every per-GPU xnminer instance.

Reads exactly what the individual dashboards write, so the numbers agree:
  data/session_timelapse.jsonl   -> last {"type":"sample"} = hps / vram / temp
  data/mining_stats_history.json -> accepted BLOCK counts for the mining day
Live GPU utilisation and power come straight from NVML.

Usage:  python3 xnm-summary.py [glob]      default glob: /root/xnminer-gpu*
"""
from __future__ import annotations

import glob as globmod
import json
import os
import re
import sys
import threading
import time
import urllib.request
from datetime import date
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

# Reward maths straight from the miner so halvings stay in sync.
try:
    from monitoring.periods import mining_day
    from monitoring.rewards import blocks_to_tokens, reward_era_label
except ImportError:  # run from outside the miner tree
    def mining_day(now=None):  # type: ignore
        return date.today()

    def blocks_to_tokens(kind, blocks, on=None):  # type: ignore
        return float(blocks) * (2.5 if kind == "XNM" else 1.0)

    def reward_era_label(on=None):  # type: ignore
        return "reward table unavailable"

try:
    import pynvml
except ImportError:
    pynvml = None

KINDS = ("XNM", "XUNI", "XBLK")
STALE_AFTER_S = 90.0          # samples land every 30 s
REFRESH_S = 3.0


def last_sample(path: Path) -> dict:
    """Last {"type":"sample"} record, read from the tail so the file can grow."""
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            f.seek(max(0, size - 65536))
            chunk = f.read().decode("utf-8", errors="replace")
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
        if rec.get("type") == "sample":
            return rec
    return {}


def today_blocks(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {k: 0 for k in KINDS}
    day = raw.get("days", {}).get(mining_day().isoformat(), {})
    return {k: int(day.get(k, 0)) for k in KINDS}


def gpu_index_of(dirname: str) -> int:
    m = re.search(r"(\d+)$", dirname)
    return int(m.group(1)) if m else 0


def nvml_live(index: int) -> dict:
    if pynvml is None:
        return {}
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(index)
        name = pynvml.nvmlDeviceGetName(h)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        return {
            "name": name.replace("NVIDIA ", ""),
            "util": pynvml.nvmlDeviceGetUtilizationRates(h).gpu,
            "temp": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
            "power": pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0,
            "vram_used": mem.used // 1048576,
            "vram_total": mem.total // 1048576,
        }
    except Exception:
        return {}


# --- pool side ------------------------------------------------------------
# The pool exposes /difficulty and /verify and nothing else - there is no
# global-hashrate endpoint, and XenBlocks "difficulty" is the Argon2
# memory_cost, so a network H/s cannot be derived from it either. We show the
# live network difficulty instead, which is a real chain-wide number.
NET = {"difficulty": None, "latency_ms": None, "ok": False, "checked": 0.0}
NET_POLL_S = 30.0


def poll_network(base_url: str) -> None:
    url = base_url.rstrip("/") + "/difficulty"
    while True:
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(url, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
            NET.update(
                difficulty=int(data.get("difficulty", data.get("diff", 0))) or None,
                latency_ms=(time.perf_counter() - t0) * 1000,
                ok=True,
            )
        except Exception:
            NET.update(ok=False, latency_ms=None)
        NET["checked"] = time.time()
        time.sleep(NET_POLL_S)


def ini_value(path: Path, key: str, default: str = "") -> str:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith(key) and "=" in stripped:
                return stripped.split("=", 1)[1].strip()
    except OSError:
        pass
    return default


def wallet_of(path: Path) -> str:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("address"):
                return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return ""


def hps_fmt(v: float) -> str:
    """Compact - the column header already says H/s."""
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


def build(dirs: list[str]) -> Group:
    now = time.time()
    table = Table(expand=True, header_style="bold cyan", border_style="grey37")
    table.add_column("GPU", justify="right", no_wrap=True)
    table.add_column("Card", no_wrap=True)
    table.add_column("Up", justify="right", no_wrap=True)
    table.add_column("H/s", justify="right")
    table.add_column("Util", justify="right")
    table.add_column("Temp", justify="right")
    table.add_column("Power", justify="right")
    table.add_column("VRAM", justify="right")
    table.add_column("XNM", justify="right", style="green")
    table.add_column("XUNI", justify="right", style="yellow")
    table.add_column("XBLK", justify="right", style="red")
    table.add_column("Net", justify="center")

    tot_hps = 0.0
    tot_power = 0.0
    tot = {k: 0 for k in KINDS}
    live_count = 0
    longest_up = 0.0
    wallet = ""

    for d in dirs:
        p = Path(d)
        idx = gpu_index_of(p.name)
        sample = last_sample(p / "data" / "session_timelapse.jsonl")
        blocks = today_blocks(p / "data" / "mining_stats_history.json")
        nv = nvml_live(idx)
        wallet = wallet or wallet_of(p / "miner.ini")

        age = now - float(sample.get("wall_ts") or 0)
        fresh = bool(sample) and age < STALE_AFTER_S
        hps = float(sample.get("hps") or 0.0) if fresh else 0.0
        if fresh:
            live_count += 1
            tot_hps += hps
        for k in KINDS:
            tot[k] += blocks[k]
        tot_power += nv.get("power", 0.0)

        if not sample:
            net = Text("—", style="grey50")
        elif not fresh:
            net = Text("stale", style="bold yellow")
        elif sample.get("network_ok"):
            net = Text("online", style="green")
        else:
            net = Text("offline", style="bold red")

        temp = nv.get("temp", sample.get("temp_c", 0))
        # elapsed_s counts from that miner's own session start; add the age of
        # the last sample so the clock keeps ticking between 30 s writes.
        up = float(sample.get("elapsed_s") or 0) + (age if fresh else 0)
        longest_up = max(longest_up, up if sample else 0)

        table.add_row(
            str(idx),
            nv.get("name", "?"),
            dur(up) if sample else Text("—", style="grey50"),
            hps_fmt(hps) if fresh else Text("—", style="grey50"),
            f"{nv['util']}%" if "util" in nv else "—",
            Text(f"{temp}°C", style="bold red" if temp and temp >= 80 else ""),
            f"{nv['power']:.0f}W" if "power" in nv else "—",
            (f"{nv['vram_used'] / 1024:.1f}/{nv['vram_total'] / 1024:.0f}G"
             if "vram_total" in nv else "—"),
            str(blocks["XNM"]),
            str(blocks["XUNI"]),
            str(blocks["XBLK"]),
            net,
        )

    table.add_section()
    table.add_row(
        Text("ALL", style="bold"),
        Text(f"{live_count}/{len(dirs)} mining", style="bold"),
        Text(dur(longest_up), style="bold"),
        Text(hps_fmt(tot_hps), style="bold cyan"),
        "", "",
        Text(f"{tot_power:.0f}W", style="bold") if tot_power else "",
        "",
        Text(str(tot["XNM"]), style="bold green"),
        Text(str(tot["XUNI"]), style="bold yellow"),
        Text(str(tot["XBLK"]), style="bold red"),
        "",
    )

    tokens = " · ".join(
        f"{blocks_to_tokens(k, tot[k]):g} {k}" for k in KINDS if tot[k]
    ) or "no accepted blocks yet today"

    if NET["ok"] and NET["difficulty"]:
        # Celadon - readable on black without colliding with the greens used for
        # XNM counts or the "online" state.
        net_bit = (f"network diff {NET['difficulty']:,}"
                   f" ({NET['latency_ms']:.0f}ms)", "bold #ACE1AF")
    elif NET["checked"]:
        net_bit = ("pool unreachable", "bold red")
    else:
        net_bit = ("network …", "grey50")

    head = Text.assemble(
        ("XenBlocks · all GPUs", "bold white"),
        ("   ", ""),
        (reward_era_label(), "cyan"),
        ("   ", ""),
        net_bit,
        ("   ", ""),
        (f"running {dur(longest_up)}" if longest_up else "starting up", "bold white"),
        ("   ", ""),
        (time.strftime("%H:%M:%S"), "grey50"),
    )
    foot = Text.assemble(
        ("Today (accepted): ", "bold"), (tokens, "bold white"),
        ("\nwallet ", "grey50"), (wallet or "?", "grey50"),
        ("   ·   blocks per mining day (1am boundary) · H/s sampled every 30s", "grey50"),
        ("\nPool publishes difficulty only — it exposes no network hashrate, and Argon2 "
         "memory-cost difficulty cannot be converted into one.", "grey50"),
        ("\nCtrl+B then 0-9 switches card · Ctrl+C closes this view only", "grey50"),
    )
    return Group(head, table, foot)


def main() -> int:
    pattern = sys.argv[1] if len(sys.argv) > 1 else "/root/xnminer-gpu*"

    found = sorted(d for d in globmod.glob(pattern) if os.path.isdir(d))
    base_url = "http://xenblocks.io"
    if found:
        base_url = ini_value(Path(found[0]) / "miner.ini", "base_url", base_url)
    threading.Thread(target=poll_network, args=(base_url,), daemon=True).start()

    if pynvml is not None:
        try:
            pynvml.nvmlInit()
        except Exception:
            pass
    console = Console()
    try:
        with Live(console=console, screen=True, refresh_per_second=4) as live:
            while True:
                dirs = sorted(
                    (d for d in globmod.glob(pattern) if os.path.isdir(d)),
                    key=lambda d: gpu_index_of(os.path.basename(d)),
                )
                if not dirs:
                    live.update(Text(f"No miner directories match {pattern}", style="bold red"))
                else:
                    live.update(build(dirs))
                time.sleep(REFRESH_S)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
PYSUMEOF
chmod +x "${BASE}/xnm-summary.py"

# --- 8. launch one miner per card, each in its own tmux window -------------
# Refuse to kill the session we are sitting in - that would kill this script.
if [ -n "${TMUX:-}" ] && [ "$(tmux display-message -p '#S' 2>/dev/null)" = "$SESSION" ]; then
  die "You are inside tmux session '$SESSION'. Detach (Ctrl+B then D) and re-run."
fi
tmux kill-session -t "$SESSION" 2>/dev/null || true

for i in $(seq 0 $((GPUS - 1))); do
  CMD="cd /root/xnminer-gpu${i} && CUDA_VISIBLE_DEVICES=${i} XNM_GPU=${i} python3 main.py; exec bash"
  if [ "$i" = "0" ]; then
    tmux new-session -d -s "$SESSION" -n "gpu0" "$CMD"
  else
    tmux new-window -t "$SESSION" -n "gpu${i}" "$CMD"
  fi
done

# Summary window last, so windows 0..N-1 keep matching the GPU numbers.
tmux new-window -t "$SESSION" -n "all" \
  "cd ${BASE} && python3 xnm-summary.py; exec bash"
tmux select-window -t "${SESSION}:all"

log "${GPUS} miner(s) + summary running in tmux session '${SESSION}'"
echo "  summary:      Ctrl+B then ${GPUS}"
echo "  single card:  Ctrl+B then 0 .. $((GPUS - 1))"
echo "  detach:       Ctrl+B then D"
echo "  re-attach:    tmux attach -t ${SESSION}"
echo

sleep 2
if [ -n "${TMUX:-}" ]; then
  # Already in another tmux session - hop across instead of nesting.
  exec tmux switch-client -t "${SESSION}:all"
elif [ -t 1 ]; then
  exec tmux attach -t "${SESSION}:all"
else
  echo "Started headless. Attach later with: tmux attach -t ${SESSION}"
fi
