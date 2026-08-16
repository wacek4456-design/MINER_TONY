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
GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
[ "${GPUS:-0}" -ge 1 ] || die "nvidia-smi reports no GPUs"
log "Detected ${GPUS} GPU(s)"
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
log "Patching upstream (3 fixes)"
python3 - <<'PYEOF'
import pathlib, sys

def edit(path, pairs, marker):
    p = pathlib.Path(path)
    s = p.read_text()
    if marker in s:
        print(f"  {path}: already patched")
        return
    for old, new in pairs:
        if old not in s:
            sys.exit(f"ERROR: pattern not found in {path}:\n  {old}")
        s = s.replace(old, new, 1)
    p.write_text(s)
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
  [ -f miner.ini ] || cp miner.ini.example miner.ini
  sed -i "s|^address =.*|address = ${WALLET}|" miner.ini
  sed -i "s|^worker =.*|worker = xnminer-gpu${i}|" miner.ini
  sed -i "s|^woodyminer_custom_name =.*|woodyminer_custom_name = xnminer-gpu${i}|" miner.ini
  echo "  GPU${i} -> ${DIR}"
  cd "$BASE"
done

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
import time
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


def wallet_of(path: Path) -> str:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("address"):
                return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return ""


def hps_fmt(v: float) -> str:
    if v >= 1_000_000:
        return f"{v / 1_000_000:.2f} MH/s"
    if v >= 1000:
        return f"{v / 1000:.1f} kH/s"
    return f"{v:.0f} H/s"


def build(dirs: list[str]) -> Group:
    now = time.time()
    table = Table(expand=True, header_style="bold cyan", border_style="grey37")
    table.add_column("GPU", justify="right", no_wrap=True)
    table.add_column("Card", no_wrap=True)
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
        table.add_row(
            str(idx),
            nv.get("name", "?"),
            hps_fmt(hps) if fresh else Text("—", style="grey50"),
            f"{nv['util']}%" if "util" in nv else "—",
            Text(f"{temp}°C", style="bold red" if temp and temp >= 80 else ""),
            f"{nv['power']:.0f} W" if "power" in nv else "—",
            f"{nv['vram_used']}/{nv['vram_total']}" if "vram_total" in nv else "—",
            str(blocks["XNM"]),
            str(blocks["XUNI"]),
            str(blocks["XBLK"]),
            net,
        )

    table.add_section()
    table.add_row(
        Text("ALL", style="bold"),
        Text(f"{live_count}/{len(dirs)} mining", style="bold"),
        Text(hps_fmt(tot_hps), style="bold cyan"),
        "", "",
        Text(f"{tot_power:.0f} W", style="bold") if tot_power else "",
        "",
        Text(str(tot["XNM"]), style="bold green"),
        Text(str(tot["XUNI"]), style="bold yellow"),
        Text(str(tot["XBLK"]), style="bold red"),
        "",
    )

    tokens = " · ".join(
        f"{blocks_to_tokens(k, tot[k]):g} {k}" for k in KINDS if tot[k]
    ) or "no accepted blocks yet today"

    head = Text.assemble(
        ("XenBlocks · all GPUs", "bold white"),
        ("   ", ""),
        (reward_era_label(), "cyan"),
    )
    foot = Text.assemble(
        ("Today (accepted): ", "bold"), (tokens, "bold white"),
        ("\nwallet ", "grey50"), (wallet or "?", "grey50"),
        ("   ·   blocks counted per mining day (1am boundary) · H/s sampled every 30s", "grey50"),
        ("\nCtrl+B then 0-9 switches card · Ctrl+C closes this view only", "grey50"),
    )
    return Group(head, table, foot)


def main() -> int:
    pattern = sys.argv[1] if len(sys.argv) > 1 else "/root/xnminer-gpu*"
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

if [ -n "${TMUX:-}" ]; then
  echo "Already inside tmux - jump across with:  tmux switch-client -t ${SESSION}"
elif [ -t 1 ]; then
  sleep 2
  exec tmux attach -t "$SESSION"
else
  echo "Started headless. Attach later with: tmux attach -t ${SESSION}"
fi
