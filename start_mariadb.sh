#!/bin/bash
# Copyright (c) 2026 VillageSQL Contributors
# SPDX-License-Identifier: Apache-2.0
# start_mariadb.sh — boot a MariaDB server on a scratch datadir for driving its
# vector (MHNSW) index with the harness. Prints the socket path on success.
# Mirrors start_server.sh's scratch-dir model.
set -euo pipefail

# MB: MariaDB build tree OR installed package basedir. Point at your own, e.g.
#   MB=/opt/homebrew/opt/mariadb@11.8            (Homebrew stock package)
#   MB=/path/to/mariadb-server/build-release     (source build tree)
# MARIADB_SRC: source dir (only needed for a build tree, for install-db scripts).
MB="${MB:?set MB to your MariaDB basedir (build tree or installed package)}"
SRC="${MARIADB_SRC:-$MB}"
WORKDIR="${WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.run-maria}"
DATADIR="$WORKDIR/data"
SOCKET="$WORKDIR/mariadb.sock"
ERRLOG="$WORKDIR/mariadb.err"
PIDFILE="$WORKDIR/mariadb.pid"

# Support BOTH a build tree ($MB/sql/mariadbd, $MB/client/mariadb) and an
# installed layout ($MB/bin/... -- e.g. Homebrew's mariadb@11.8 stock package).
# The installed layout ships mariadb-install-db in bin/ and needs no srcdir.
if [ -x "$MB/sql/mariadbd" ]; then
  MARIADBD="$MB/sql/mariadbd"; MARIADB="$MB/client/mariadb"; LAYOUT="build"
  INSTALL_DB="$MB/scripts/mariadb-install-db"
elif [ -x "$MB/bin/mariadbd" ]; then
  MARIADBD="$MB/bin/mariadbd"; MARIADB="$MB/bin/mariadb"; LAYOUT="install"
  INSTALL_DB="$MB/bin/mariadb-install-db"
else
  echo "ERROR: mariadbd not found under $MB (looked in sql/ and bin/)" >&2; exit 1
fi

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  kill "$(cat "$PIDFILE")" 2>/dev/null || true; sleep 2
fi
rm -rf "$WORKDIR"; mkdir -p "$DATADIR"

echo "Initializing scratch datadir ($LAYOUT layout)..." >&2
if [ "$LAYOUT" = "build" ]; then
  # build tree: mariadb-install-db needs the source dir for scripts/charsets.
  "$INSTALL_DB" --no-defaults --srcdir="$SRC" --builddir="$MB" \
    --datadir="$DATADIR" --auth-root-authentication-method=normal >"$ERRLOG" 2>&1 \
    || "$MB/scripts/mysql_install_db" --no-defaults --srcdir="$SRC" --builddir="$MB" \
         --datadir="$DATADIR" >"$ERRLOG" 2>&1
else
  # installed package: basedir has share/ + support-files; no srcdir needed.
  "$INSTALL_DB" --no-defaults --basedir="$MB" \
    --datadir="$DATADIR" --auth-root-authentication-method=normal >"$ERRLOG" 2>&1
fi

echo "Starting mariadbd (socket=$SOCKET)..." >&2
# MHNSW_CACHE_BYTES: size for MariaDB's HNSW graph cache. The 16MB default is
# smaller than a real dataset's working graph; on overflow MariaDB reset()s the
# graph context and loses the learned stat that sizes its bloom-filter visited-
# set, which affects both build speed and recall. To size the cache to the graph,
# pass ~max(512MB, 2 x N x dim x 4) (the harness computes this and warns if it is
# too small). Unset = leave MariaDB at its 16MB default.
MHNSW_ARG=""
[ -n "${MHNSW_CACHE_BYTES:-}" ] && MHNSW_ARG="--mhnsw-max-cache-size=${MHNSW_CACHE_BYTES}"
# MARIADBD_EXTRA: optional extra args (e.g. --innodb-buffer-pool-size=4G) for
# perf experiments; space-separated, word-split intentionally.
"$MARIADBD" --no-defaults --basedir="$MB" --datadir="$DATADIR" \
  --socket="$SOCKET" --pid-file="$PIDFILE" --skip-networking \
  --log-error="$ERRLOG" --secure-file-priv="" \
  ${MHNSW_ARG:+$MHNSW_ARG} ${MARIADBD_EXTRA:-} &

for i in $(seq 1 60); do
  if "$MARIADB" --no-defaults -uroot --socket="$SOCKET" -e "SELECT 1" >/dev/null 2>&1; then
    echo "READY socket=$SOCKET" >&2
    echo "$SOCKET"
    exit 0
  fi
  sleep 1
done
echo "ERROR: mariadb did not become ready; tail $ERRLOG:" >&2
tail -30 "$ERRLOG" >&2
exit 1
