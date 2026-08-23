#!/usr/bin/env bash
# woodysoil/XenblocksMiner (prebuilt C++ binary) - bootstrap for Vast.ai
#
#   ./vast-woody.sh 0xYourWallet
#
# Why this exists: the Python xnminer tops out at ~3.5% of the card's memory-
# bandwidth ceiling at difficulty 100 (measured 2026-08-18: 265 kH/s on an
# A5000 while rigs on weaker 3060 Ti cards report 1.6 MH/s). The C++ miner has
# no Python loop between batches. One process drives every GPU.
#
# Env overrides:
#   WOODY_DEVICES=0        mine on selected card(s) only, e.g. "0" or "0,2"
#   WOODY_DEVFEE=0         dev fee in permille (default 0)
#   WOODY_NAME=...         custom name on woodyminer.com stats
#   WOODY_RPC=direct       talk straight to xenblocks.io, skip the local proxy
#   WOODY_VERSION=1.4.0    release to download
set -euo pipefail

WALLET="${1:-${MINER_ADDR:-0x9d79B1921b75AC7C199314406f5398E15f2fb47C}}"
VER="${WOODY_VERSION:-1.4.0}"
URL="https://github.com/woodysoil/XenblocksMiner/releases/download/v${VER}/xenblocksMiner-${VER}-linux.tar.gz"
DIR="${WOODY_DIR:-/root/woody}"
SESSION="woody"
DIFF_PORT="${XNM_DIFF_PORT:-8899}"
DEVFEE="${WOODY_DEVFEE:-0}"
DEVICES="${WOODY_DEVICES:-}"
NAME="${WOODY_NAME:-}"
RPC_MODE="${WOODY_RPC:-proxy}"
LOG="/root/woody.log"

log() { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

[[ "$WALLET" =~ ^0x[0-9a-fA-F]{40}$ ]] || die "not a valid EVM address: $WALLET"

export DEBIAN_FRONTEND=noninteractive

# --- 1. packages (binary needs nothing built; python3 only for the proxy) ---
NEED=""
command -v wget  >/dev/null 2>&1 || NEED="$NEED wget"
command -v tmux  >/dev/null 2>&1 || NEED="$NEED tmux"
if [ "$RPC_MODE" = "proxy" ]; then
  command -v python3 >/dev/null 2>&1 || NEED="$NEED python3"
fi
if [ -n "$NEED" ]; then
  log "Installing:$NEED"
  apt-get update -qq
  # shellcheck disable=SC2086
  apt-get install -y -qq --no-install-recommends $NEED ca-certificates >/dev/null
fi

# --- 2. the miner binary ----------------------------------------------------
log "Fetching XenblocksMiner v${VER}"
mkdir -p "$DIR"
cd "$DIR"
wget -N -q "$URL" || die "download failed: $URL"
tar -zxf "xenblocksMiner-${VER}-linux.tar.gz" --overwrite
chmod +x xenblocksMiner

# Smoke test - a missing shared library on this Ubuntu would die instantly and
# silently inside tmux, so surface it here instead.
if ! ./xenblocksMiner --help >/tmp/woody-help.txt 2>&1; then
  cat /tmp/woody-help.txt
  die "binary does not run on this image (see output above - likely a missing library)"
fi

# --- 3. difficulty proxy ----------------------------------------------------
# xenblocks.io drops port 80 for long stretches; the same domain serves the
# live difficulty over HTTPS. --rpcLink points the miner at this proxy: GETs
# are answered from cache in milliseconds, POSTs are forwarded to the real
# pool and fail fast (503) during an outage.
RPC_ARGS=""
if [ "$RPC_MODE" = "proxy" ]; then
  if wget -qO- -T 2 "http://127.0.0.1:${DIFF_PORT}/difficulty" >/dev/null 2>&1; then
    log "Difficulty proxy already answering on :${DIFF_PORT} - reusing it"
  else
    log "Starting difficulty proxy on 127.0.0.1:${DIFF_PORT}"
    cat > "${DIR}/xnm-diffproxy.py" <<'PYDIFFEOF'
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

        # Fail fast while port 80 is down. Otherwise every queued block costs the
        # miner a full connect timeout, and because /difficulty still answers it
        # believes the network is up and keeps retrying instead of hashing -
        # which starves mining far worse than the outage itself.
        with _lock:
            pool_alive = _state["source"] == "pool"
        if not pool_alive:
            self._send(503, b'{"error":"pool offline (port 80) - queue it"}')
            return

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
    nohup bash -c "while true; do python3 '${DIR}/xnm-diffproxy.py' --port ${DIFF_PORT}; sleep 5; done" \
      >> /root/xnm-diffproxy.log 2>&1 &
    sleep 2
    python3 "${DIR}/xnm-diffproxy.py" --once || log "WARNING: no difficulty source reachable yet"
  fi
  RPC_ARGS="--rpcLink http://127.0.0.1:${DIFF_PORT}"
fi

# --- 4. run it in tmux ------------------------------------------------------
CMD="./xenblocksMiner --minerAddr ${WALLET} --totalDevFee ${DEVFEE}"
[ -n "$RPC_ARGS" ] && CMD="${CMD} ${RPC_ARGS}"
[ -n "$DEVICES" ]  && CMD="${CMD} --device=${DEVICES}"
[ -n "$NAME" ]     && CMD="${CMD} --customName ${NAME}"

if [ -n "${TMUX:-}" ] && [ "$(tmux display-message -p '#S' 2>/dev/null)" = "$SESSION" ]; then
  die "You are inside tmux session '$SESSION'. Detach (Ctrl+B then D) and re-run."
fi
tmux kill-session -t "$SESSION" 2>/dev/null || true

log "Launching: ${CMD}"
tmux new-session -d -s "$SESSION" -n miner \
  "cd '${DIR}' && ${CMD} 2>&1 | tee -a '${LOG}'; exec bash"

log "Miner running in tmux session '${SESSION}'  (log: ${LOG})"
echo "  attach:  tmux attach -t ${SESSION}     detach: Ctrl+B then D"
echo "  stats:   https://woodyminer.com (find your address)"
echo

sleep 2
if [ -n "${TMUX:-}" ]; then
  exec tmux switch-client -t "$SESSION"
elif [ -t 1 ]; then
  exec tmux attach -t "$SESSION"
else
  echo "Started headless."
fi
