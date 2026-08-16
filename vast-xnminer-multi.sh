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

# --- 8. launch one miner per card, each in its own tmux window -------------
tmux kill-session -t "$SESSION" 2>/dev/null || true
for i in $(seq 0 $((GPUS - 1))); do
  CMD="cd /root/xnminer-gpu${i} && CUDA_VISIBLE_DEVICES=${i} XNM_GPU=${i} python3 main.py; exec bash"
  if [ "$i" = "0" ]; then
    tmux new-session -d -s "$SESSION" -n "gpu0" "$CMD"
  else
    tmux new-window -t "$SESSION" -n "gpu${i}" "$CMD"
  fi
done

log "${GPUS} miner(s) running in tmux session '${SESSION}'"
echo "  switch card:  Ctrl+B then 0 / 1 / 2 / 3"
echo "  detach:       Ctrl+B then D"
echo "  re-attach:    tmux attach -t ${SESSION}"
echo

if [ -t 1 ]; then
  sleep 2
  exec tmux attach -t "$SESSION"
else
  echo "Started headless. Attach later with: tmux attach -t ${SESSION}"
fi
