#!/usr/bin/env bash
# XenBlocks miner (badnob/xnminer-linux) - one-command bootstrap for Vast.ai
#
#   wget -N <url>/vast-xnminer.sh && chmod +x vast-xnminer.sh && ./vast-xnminer.sh 0xYourWallet
#
# Re-runnable: clone, patch, build and venv are all skipped if already present.
set -euo pipefail

WALLET="${1:-${MINER_ADDR:-0x9d79B1921b75AC7C199314406f5398E15f2fb47C}}"
MINER_DIR="${XNM_DIR:-$HOME/xnminer}"
REPO="https://github.com/badnob/xnminer-linux.git"
SESSION="xnm"

log() { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

[[ "$WALLET" =~ ^0x[0-9a-fA-F]{40}$ ]] || die "not a valid EVM address: $WALLET"

SUDO=""
[ "$(id -u)" -eq 0 ] || SUDO="sudo"

# --- keep mining alive when the SSH session drops --------------------------
# Vast kills foreground processes with the shell, so re-exec inside tmux.
if [ -z "${TMUX:-}" ] && [ "${XNM_NO_TMUX:-0}" != "1" ]; then
  if ! command -v tmux >/dev/null 2>&1; then
    $SUDO apt-get update -qq >/dev/null 2>&1 || true
    $SUDO apt-get install -y -qq tmux >/dev/null 2>&1 || true
  fi
  if command -v tmux >/dev/null 2>&1; then
    SELF="$(readlink -f "$0")"
    log "Running inside tmux session '$SESSION'  (detach: Ctrl+B then D)"
    exec tmux new -A -s "$SESSION" "XNM_NO_TMUX=1 bash '$SELF' '$WALLET'"
  fi
  echo "tmux unavailable - running in foreground; do not close this window"
fi

# --- 1. repo ---------------------------------------------------------------
log "Repository -> $MINER_DIR"
if ! command -v git >/dev/null 2>&1; then
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq git >/dev/null
fi
if [ -d "$MINER_DIR/.git" ]; then
  echo "already cloned - reusing (delete $MINER_DIR for a clean start)"
else
  git clone --depth 1 "$REPO" "$MINER_DIR"
fi
cd "$MINER_DIR"

# --- 2. patch the startup self-test ----------------------------------------
# verify_known_block() hashes the reference log0.txt key using OUR wallet as the
# Argon2 salt, but that block was mined with its author's salt. Different salt ->
# different digest, so "XEN11" can never appear and startup aborts for everyone
# except the original miner. Neutralise the gate; the diagnostic still logs.
log "Patching Argon2 calibration gate"
if ! grep -q 'self-test disabled' core/supervisor.py; then
  sed -i 's/if not diag\["calibration_m100"\]:/if False:  # self-test disabled - known block uses a foreign salt/' \
    core/supervisor.py
fi
grep -q 'self-test disabled' core/supervisor.py \
  || die "patch did not apply - upstream code changed, check core/supervisor.py"

# --- 3. wallet -------------------------------------------------------------
log "Config"
[ -f miner.ini ] || cp miner.ini.example miner.ini
sed -i "s|^address =.*|address = ${WALLET}|" miner.ini
echo "wallet: ${WALLET}"
chmod +x install.sh start-miner.sh native/build.sh

# --- 4. GPU architecture ---------------------------------------------------
# native/build.sh defaults to sm_90;120 only. On a 3090/4090/A100 that produces
# a library the card cannot load, so pin the real compute capability.
log "GPU"
ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ' || true)"
if [ -z "$ARCH" ]; then
  ARCH="75;80;86;89;90"
  echo "compute_cap unavailable - building fat binary for $ARCH (slower)"
else
  echo "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) -> sm_${ARCH}"
fi
export CMAKE_CUDA_ARCHITECTURES="$ARCH"

# --- 5. deps + engine ------------------------------------------------------
if [ ! -f native/build/bin/libxen_cuda.so ] || [ ! -d .venv ]; then
  log "Installing deps and building CUDA engine (slowest step)"
  # --no-driver: the container already has the host NVIDIA driver
  ./install.sh --no-driver
else
  echo "engine and venv already present - skipping install"
fi
[ -f native/build/bin/libxen_cuda.so ] \
  || die "libxen_cuda.so was not built - check the build output above"

# --- 6. mine ---------------------------------------------------------------
log "Starting miner"
exec ./start-miner.sh
