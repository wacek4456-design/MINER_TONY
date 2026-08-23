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

log "${GPUS} miner(s) running in tmux session '${SESSION}'"
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
