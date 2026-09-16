#!/bin/bash
# Copyright (c) 2026 VillageSQL Contributors
# SPDX-License-Identifier: Apache-2.0
# start_gv.sh — boot a scratch Google MySQL 9.x server (native VECTOR type +
# ScaNN approximate index) for the recall harness's `google` profile. Wipes +
# reinits the scratch datadir each run so tests start clean. Prints the socket.
#
# Google-specific vs the other engines:
#   * the ScaNN runtime (libscann.so) must be on LD_LIBRARY_PATH, or CREATE
#     VECTOR INDEX fails to load the storage engine.
#   * --cloudsql-vector=ON is required at startup (the VECTOR feature master
#     switch; it is a read-only variable, so it cannot be set at runtime).
#   * MySQL 9.x has no --mysql-native-password option; the bench account uses
#     the default auth plugin (caching_sha2_password), which PyMySQL supports.
#
# Env overrides:
#   GV_BUILD    server build/install dir with bin/mysqld + bin/mysql (REQUIRED)
#   SCANN_LIB   dir containing libscann.so (REQUIRED)
#   WORKDIR     scratch dir for datadir/socket/logs (default <repo>/.run-gv)
#   GV_PORT     TCP port (default 3307, to avoid a stock 3306)
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${GV_BUILD:?set GV_BUILD=/path/to/google-mysql/build (with bin/mysqld)}"
: "${SCANN_LIB:?set SCANN_LIB=/path/to/dir/containing/libscann.so}"

MYSQLD="$GV_BUILD/bin/mysqld"
MYSQL="$GV_BUILD/bin/mysql"
WORKDIR="${WORKDIR:-$BENCH_DIR/.run-gv}"
DATA="$WORKDIR/data"
SOCK="$WORKDIR/mysqld.sock"
ERR="$WORKDIR/mysqld.err"
PIDFILE="$WORKDIR/mysqld.pid"
GV_PORT="${GV_PORT:-3307}"

export LD_LIBRARY_PATH="$SCANN_LIB:${LD_LIBRARY_PATH:-}"

# Stop a previous scratch server, then reinit a clean datadir.
[ -f "$PIDFILE" ] && kill "$(cat "$PIDFILE")" 2>/dev/null || true
sleep 1
rm -rf "$WORKDIR"; mkdir -p "$DATA"
"$MYSQLD" --no-defaults --initialize-insecure --basedir="$GV_BUILD" \
  --datadir="$DATA" --log-error="$ERR"

"$MYSQLD" --no-defaults --basedir="$GV_BUILD" --datadir="$DATA" \
  --socket="$SOCK" --pid-file="$PIDFILE" --port="$GV_PORT" \
  --secure-file-priv= --cloudsql-vector=ON --log-error="$ERR" &

for i in $(seq 1 60); do
  "$MYSQL" --no-defaults -uroot --socket="$SOCK" -e "SELECT 1" >/dev/null 2>&1 && break
  sleep 1
done

# Bench account (default auth plugin; PyMySQL handles caching_sha2_password).
"$MYSQL" --no-defaults -uroot --socket="$SOCK" -e "
  CREATE USER IF NOT EXISTS 'bench'@'%'         IDENTIFIED BY 'bench';
  GRANT ALL PRIVILEGES ON *.* TO 'bench'@'%'         WITH GRANT OPTION;
  CREATE USER IF NOT EXISTS 'bench'@'localhost' IDENTIFIED BY 'bench';
  GRANT ALL PRIVILEGES ON *.* TO 'bench'@'localhost' WITH GRANT OPTION;
  FLUSH PRIVILEGES;"

# Sanity: server up and the VECTOR functions resolve.
"$MYSQL" --no-defaults -uroot --socket="$SOCK" \
  -e "SELECT VERSION(); SELECT string_to_vector('[1,2,3]') IS NOT NULL AS vec_ok;" \
  2>&1 | grep -viE "insecure|password" || true

echo "READY socket=$SOCK"
