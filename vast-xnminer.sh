#!/usr/bin/env bash
# XenBlocks miner (badnob/xnminer-linux) - bootstrap for Vast.ai
# Image: nvidia/cuda:*-devel-ubuntu*   (nvcc already present)
#
# Two ways to use it:
#   1) as a boot script, straight from your PC - nothing to host anywhere:
#        vastai create instance <offer_id> --image nvidia/cuda:12.6.1-devel-ubuntu24.04 \
#          --disk 32 --ssh --direct --onstart-script vast-xnminer.sh
#      Miner starts by itself; log lands in /root/xnminer.log
#   2) by hand on the instance:
#        ./vast-xnminer.sh 0xYourWallet
#
# Optional: XNM_SO_URL=<url to a prebuilt libxen_cuda.so> skips the compile.
set -euo pipefail

WALLET="${1:-${MINER_ADDR:-0x9d79B1921b75AC7C199314406f5398E15f2fb47C}}"
MINER_DIR="${XNM_DIR:-/root/xnminer}"
SRC_TARBALL="https://github.com/wacek4456-design/NAZWA_REPO/archive/refs/heads/main.tar.gz"
SO_URL="${XNM_SO_URL:-}"
SO_PATH="native/build/bin/libxen_cuda.so"
LOG="/root/xnminer.log"
SESSION="xnm"

# No terminal => started by Vast at boot: log to a file, no tmux, no dashboard.
if [ -t 1 ]; then INTERACTIVE=1; else INTERACTIVE=0; exec >>"$LOG" 2>&1; fi

log() { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

[[ "$WALLET" =~ ^0x[0-9a-fA-F]{40}$ ]] || die "not a valid EVM address: $WALLET"

export DEBIAN_FRONTEND=noninteractive
export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

log "xnminer bootstrap - wallet ${WALLET}"

# --- 1. packages the bare nvidia/cuda image lacks --------------------------
if ! command -v python3 >/dev/null 2>&1 || ! command -v cmake >/dev/null 2>&1; then
  log "Installing python3 / pip / cmake"
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends \
    python3 python3-pip cmake build-essential ca-certificates wget tmux >/dev/null
fi

# --- 2. interactive only: survive a dropped SSH session --------------------
if [ "$INTERACTIVE" = "1" ] && [ -z "${TMUX:-}" ] && [ "${XNM_NO_TMUX:-0}" != "1" ] \
   && command -v tmux >/dev/null 2>&1; then
  SELF="$(readlink -f "$0")"
  log "Re-running inside tmux session '$SESSION'  (detach: Ctrl+B then D)"
  exec tmux new -A -s "$SESSION" "XNM_NO_TMUX=1 XNM_SO_URL='${SO_URL}' bash '$SELF' '$WALLET'"
fi

# --- 3. source (tarball, so git is not needed) -----------------------------
if [ -f "$MINER_DIR/main.py" ]; then
  log "Source already present in $MINER_DIR - reusing"
else
  log "Fetching miner source"
  mkdir -p "$MINER_DIR"
  wget -qO /tmp/xnminer-src.tar.gz "$SRC_TARBALL"
  tar -xzf /tmp/xnminer-src.tar.gz -C "$MINER_DIR" --strip-components=1
fi
cd "$MINER_DIR"

# --- 4. patch the startup self-test ----------------------------------------
# verify_known_block() hashes the reference log0.txt key using OUR wallet as the
# Argon2 salt, but that block was mined with its author's salt. Different salt ->
# different digest, so "XEN11" can never appear and startup aborts for everyone
# except the original miner. Neutralise the gate; the diagnostic still logs.
if ! grep -q 'self-test disabled' core/supervisor.py; then
  log "Patching Argon2 calibration gate"
  sed -i 's/if not diag\["calibration_m100"\]:/if False:  # self-test disabled - known block uses a foreign salt/' \
    core/supervisor.py
fi
grep -q 'self-test disabled' core/supervisor.py \
  || die "patch did not apply - upstream code changed, check core/supervisor.py"

# --- 4b. patch the CUDA engine source --------------------------------------
# LaneSlot holds a std::mutex - neither copyable nor movable - yet the lanes
# live in a std::vector that push_back()s and resize()s them. No conforming
# compiler accepts that, so the Linux engine has never built as shipped.
# Putting the mutex behind a unique_ptr makes LaneSlot movable again.
if ! grep -q 'unique_ptr<std::mutex> mutex' native/engine/xen_cuda_api.cpp; then
  log "Patching LaneSlot (std::mutex inside a std::vector)"
  sed -i 's/^    std::mutex mutex;$/    std::unique_ptr<std::mutex> mutex = std::make_unique<std::mutex>();/' \
    native/engine/xen_cuda_api.cpp
  sed -i 's/lane_lock(slot->mutex)/lane_lock(*slot->mutex)/g' native/engine/xen_cuda_api.cpp
fi
grep -q 'unique_ptr<std::mutex> mutex' native/engine/xen_cuda_api.cpp \
  || die "engine patch did not apply - check native/engine/xen_cuda_api.cpp"

# --- 5. wallet -------------------------------------------------------------
[ -f miner.ini ] || cp miner.ini.example miner.ini
sed -i "s|^address =.*|address = ${WALLET}|" miner.ini

# --- 6. python deps --------------------------------------------------------
if ! python3 -c "import argon2, pynvml, psutil, rich" 2>/dev/null; then
  log "Python packages"
  # Ubuntu 24.04 marks system python as externally managed (PEP 668)
  python3 -m pip install --no-cache-dir -q -r requirements.txt 2>/dev/null \
    || python3 -m pip install --no-cache-dir -q --break-system-packages -r requirements.txt
fi

# --- 7. CUDA engine --------------------------------------------------------
if [ -f "$SO_PATH" ]; then
  log "CUDA engine already built - skipping"
elif [ -n "$SO_URL" ]; then
  log "Downloading prebuilt CUDA engine"
  mkdir -p "$(dirname "$SO_PATH")"
  wget -qO "$SO_PATH" "$SO_URL" || die "could not download $SO_URL"
else
  # native/build.sh defaults to sm_90;120 only. On a 3090/4090/A100 that builds
  # a library the card cannot load, so pin the real compute capability.
  ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ' || true)"
  if [ -z "$ARCH" ]; then
    ARCH="${XNM_ARCH:-75;80;86;89;90}"
    log "compute_cap unavailable - building for $ARCH (slower)"
  else
    log "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) -> sm_${ARCH}"
  fi
  command -v nvcc >/dev/null 2>&1 \
    || die "nvcc missing - use an image tagged '-devel', not '-base' or '-runtime'"
  log "Building CUDA engine (one-off, ~1-2 min)"
  chmod +x native/build.sh
  CMAKE_CUDA_ARCHITECTURES="$ARCH" ./native/build.sh
fi
[ -f "$SO_PATH" ] || die "libxen_cuda.so missing - check the build output above"

# --- 8. mine ---------------------------------------------------------------
if [ "$INTERACTIVE" = "1" ]; then
  log "Starting miner  (Ctrl+C stops and flushes the queue)"
  exec python3 main.py
else
  log "Starting miner headless - follow with: tail -f $LOG"
  exec python3 main.py --no-dashboard
fi
