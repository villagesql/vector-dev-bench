#!/bin/bash
# Copyright (c) 2026 VillageSQL Contributors
# SPDX-License-Identifier: Apache-2.0
# start_postgres.sh -- boot a scratch PostgreSQL cluster with pgvector for the
# recall harness, as a neutral third HNSW reference (pgvector). Mirrors
# start_server.sh / start_mariadb.sh: fresh datadir each run, unix socket only,
# prints the socket DIR on success (psql connects via -h <dir>).
set -euo pipefail

PGBIN="${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}"
WORKDIR="${WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.run-pg}"
DATADIR="$WORKDIR/data"
SOCKDIR="$WORKDIR"          # psql -h "$SOCKDIR"
ERRLOG="$WORKDIR/pg.log"
DB="bench"

INITDB="$PGBIN/initdb"; PGCTL="$PGBIN/pg_ctl"; PSQL="$PGBIN/psql"; CREATEDB="$PGBIN/createdb"
[ -x "$INITDB" ] || { echo "ERROR: initdb not found: $INITDB" >&2; exit 1; }

# stop any prior cluster on this datadir
if [ -f "$DATADIR/postmaster.pid" ]; then
  "$PGCTL" -D "$DATADIR" stop -m immediate >/dev/null 2>&1 || true
  sleep 1
fi
rm -rf "$WORKDIR"; mkdir -p "$DATADIR"

echo "Initializing scratch pg cluster..." >&2
"$INITDB" -D "$DATADIR" -E UTF8 --locale=C >"$ERRLOG" 2>&1

# unix socket in WORKDIR, no TCP. Buffer/build memory overridable for perf runs:
#   SHARED_BUFFERS (default 128MB) -- pg's buffer pool (analog of innodb_buffer_pool_size)
#   MAINT_WORK_MEM (default 64MB)  -- CRITICAL for HNSW CREATE INDEX: if the graph
#     doesn't fit here pg falls back to a slow on-disk build, so size it to the
#     working set for a fair build comparison.
SHARED_BUFFERS="${SHARED_BUFFERS:-128MB}"
MAINT_WORK_MEM="${MAINT_WORK_MEM:-64MB}"
echo "Starting postgres (socketdir=$SOCKDIR, shared_buffers=$SHARED_BUFFERS maint_work_mem=$MAINT_WORK_MEM)..." >&2
"$PGCTL" -D "$DATADIR" -l "$ERRLOG" \
  -o "-k $SOCKDIR -c listen_addresses='' -c shared_buffers=$SHARED_BUFFERS -c maintenance_work_mem=$MAINT_WORK_MEM" \
  start >/dev/null 2>&1

for i in $(seq 1 60); do
  if "$PSQL" -h "$SOCKDIR" -d postgres -tAc "SELECT 1" >/dev/null 2>&1; then
    "$CREATEDB" -h "$SOCKDIR" "$DB" 2>>"$ERRLOG" || true
    "$PSQL" -h "$SOCKDIR" -d "$DB" -c "CREATE EXTENSION IF NOT EXISTS vector;" >>"$ERRLOG" 2>&1
    echo "READY sockdir=$SOCKDIR db=$DB" >&2
    echo "$SOCKDIR"
    exit 0
  fi
  sleep 1
done
echo "ERROR: postgres did not become ready; tail $ERRLOG:" >&2
tail -30 "$ERRLOG" >&2
exit 1
