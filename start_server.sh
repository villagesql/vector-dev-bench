#!/bin/bash
# Copyright (c) 2026 VillageSQL Contributors
# SPDX-License-Identifier: Apache-2.0
# start_server.sh — boot the local debug VillageSQL server on a scratch datadir
# for the vsql-vector recall harness. Idempotent-ish: wipes+reinits the scratch
# datadir each run so tests start clean. Prints the socket path on success.
#
# Server build dir resolution:
#   1. $SRV_BUILD if set (the normal path for this standalone repo), else
#   2. VillageSQL_BUILD_DIR from a sibling ../build/CMakeCache.txt — a
#      convenience only if this repo is dropped next to a vsql-vector checkout;
#      normally unset, so set SRV_BUILD.
#
# Env overrides:
#   SRV_BUILD    server build dir (REQUIRED unless the ../build fallback applies)
#   WORKDIR      scratch dir for datadir/socket/logs (default bench/.run)
#   EXTENSIONS   space-separated extension names to install (default vsql_vector)
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$BENCH_DIR/.." && pwd)"

detect_srv_build() {
  local cache="$REPO_DIR/build/CMakeCache.txt"
  [ -f "$cache" ] || return 1
  # CMakeCache line: VillageSQL_BUILD_DIR:PATH=/abs/path/to/server/build
  sed -n 's/^VillageSQL_BUILD_DIR:[^=]*=//p' "$cache" | head -1
}

SRV_BUILD="${SRV_BUILD:-$(detect_srv_build || true)}"
if [ -z "${SRV_BUILD:-}" ]; then
  echo "ERROR: server build dir unknown. Set SRV_BUILD=/path/to/villagesql/build" >&2
  echo "(the dir with runtime_output_directory/mysqld and veb_output_directory/)." >&2
  exit 1
fi
WORKDIR="${WORKDIR:-$BENCH_DIR/.run}"
MYSQLD="$SRV_BUILD/runtime_output_directory/mysqld"
DATADIR="$WORKDIR/data"
SOCKET="$WORKDIR/mysqld.sock"
ERRLOG="$WORKDIR/mysqld.err"
PIDFILE="$WORKDIR/mysqld.pid"

[ -x "$MYSQLD" ] || { echo "ERROR: mysqld not found/executable: $MYSQLD" >&2; exit 1; }

# Stop any prior instance on this socket.
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  kill "$(cat "$PIDFILE")" 2>/dev/null || true
  sleep 2
fi

rm -rf "$WORKDIR"
mkdir -p "$DATADIR"

echo "Initializing scratch datadir..." >&2
"$MYSQLD" --no-defaults --initialize-insecure \
  --basedir="$SRV_BUILD" --datadir="$DATADIR" >"$ERRLOG" 2>&1

echo "Starting mysqld (socket=$SOCKET)..." >&2
# skip-networking: harness talks over the unix socket only.
# MYSQLD_EXTRA: optional extra mysqld args (e.g. --innodb-buffer-pool-size=4G)
# for perf experiments; space-separated, word-split intentionally.
"$MYSQLD" --no-defaults \
  --basedir="$SRV_BUILD" --datadir="$DATADIR" \
  --socket="$SOCKET" --pid-file="$PIDFILE" \
  --skip-networking \
  --log-error="$ERRLOG" \
  --secure-file-priv="" \
  ${MYSQLD_EXTRA:-} \
  &

# Wait for the socket to accept connections.
MYSQL="$SRV_BUILD/runtime_output_directory/mysql"
for i in $(seq 1 60); do
  if "$MYSQL" --no-defaults -uroot --socket="$SOCKET" -e "SELECT 1" >/dev/null 2>&1; then
    # Bake the custom-KNN gates in as SERVER-WIDE DEFAULTS so every connection
    # (incl. the harness's) works without per-session SET. All three are
    # required: preview (to install/use the extension), hypergraph optimizer
    # (classic optimizer never selects the custom KNN scan → filesort → crash),
    # and the custom-index debug gate (POC path is debug-gated). Extensions to
    # install are passed space-separated via EXTENSIONS (default: vsql_vector,
    # the SVECTOR+HNSW extension). Datadir is fresh each run, so (re)install here.
    echo "Applying gate defaults + installing extensions..." >&2
    "$MYSQL" --no-defaults -uroot --socket="$SOCKET" 2>>"$ERRLOG" <<SQL || true
SET PERSIST vsql_allow_preview_extensions = ON;
SET GLOBAL optimizer_switch='hypergraph_optimizer=on';
SET GLOBAL debug='+d,villagesql_custom_index_proceed';
SQL
    # NOTE: install only ONE vector extension that registers the SVECTOR type.
    # Installing two extensions that both register the same type name makes type
    # resolution ambiguous and the server asserts (resolve_type_descriptor:
    # results.size() > 1). Default is vsql_vector; override via EXTENSIONS.
    for ext in ${EXTENSIONS:-vsql_vector}; do
      "$MYSQL" --no-defaults -uroot --socket="$SOCKET" \
        -e "INSTALL EXTENSION $ext;" 2>>"$ERRLOG" || \
        echo "  (note: INSTALL $ext skipped/failed — may already be installed)" >&2
    done
    echo "READY socket=$SOCKET" >&2
    echo "$SOCKET"
    exit 0
  fi
  sleep 1
done

echo "ERROR: server did not become ready; tail of $ERRLOG:" >&2
tail -30 "$ERRLOG" >&2
exit 1
