#!/bin/bash
# Copyright (c) 2026 VillageSQL Contributors
# SPDX-License-Identifier: Apache-2.0
# start_myvector.sh -- boot stock MySQL in Docker with the MyVector *component*
# installed, and expose it over TCP for the harness (profile=myvector).
#
# MyVector (https://github.com/askdba/myvector) is not a server fork: it is a
# component .so loaded into an unmodified mysqld, plus a set of UDFs and stored
# procedures. This script therefore can't just pull one image -- it mounts the
# component you built from the MyVector repo into a stock `mysql:<tag>` image,
# installs it, registers the UDFs/procedures, creates the bench account, and
# prints the TCP spec (as start_server_docker.sh does -- a container's unix
# socket is not reachable from the host).
#
# Drive the harness with the printed spec, e.g.:
#   recall_bench.py --profile myvector --socket tcp:127.0.0.1:$PORT \
#       --dataset fashion-mnist-784-euclidean --ef-search-sweep 10,20,50,100,200
#
# Env (paths come from a MyVector checkout/build -- see that repo's build docs):
#   MYVECTOR_SO     REQUIRED. Path to libmyvector_component.so (the built component).
#   MYVECTOR_JSON   Path to myvector.json manifest (default: next to MYVECTOR_SO).
#   MYVECTOR_PROCS  REQUIRED. Path to the SQL that creates the MyVector stored
#                   procedures (mysql.MYVECTOR_INDEX_BUILD, _STATUS, _DROP, _LOAD)
#                   -- from the MyVector repo (e.g. its component install SQL).
#   MYSQL_TAG       stock MySQL image tag (default 8.4).
#   PORT            host port to publish on 127.0.0.1 (default 3308, to avoid a
#                   local mysqld on 3306 and start_gv.sh's 3307). The harness
#                   --port / tcp:.
#   NAME            container name (default vdb-myvector).
#   READY_TIMEOUT   seconds to wait for readiness (default 120).
#
# Prints on success: "READY host=127.0.0.1 port=$PORT" to stderr and the bare
# "tcp:127.0.0.1:$PORT" to stdout (ready to pass as --socket).
set -euo pipefail

MYVECTOR_SO="${MYVECTOR_SO:?set MYVECTOR_SO to the built libmyvector_component.so}"
MYVECTOR_JSON="${MYVECTOR_JSON:-$(dirname "$MYVECTOR_SO")/myvector.json}"
MYVECTOR_PROCS="${MYVECTOR_PROCS:?set MYVECTOR_PROCS to the MyVector stored-procedure install SQL}"
MYSQL_TAG="${MYSQL_TAG:-8.4}"
PORT="${PORT:-3308}"
NAME="${NAME:-vdb-myvector}"
READY_TIMEOUT="${READY_TIMEOUT:-120}"
HOST="127.0.0.1"
ROOT_PW="benchroot"

for f in "$MYVECTOR_SO" "$MYVECTOR_JSON" "$MYVECTOR_PROCS"; do
  [ -r "$f" ] || { echo "ERROR: not readable: $f" >&2; exit 1; }
done

DOCKER="docker"
docker info >/dev/null 2>&1 || DOCKER="sudo docker"

# Fresh container each run (datadir goes with it -> component re-installs cleanly).
$DOCKER rm -f "$NAME" >/dev/null 2>&1 || true

# The component loader resolves 'file://myvector' to <plugin_dir>/myvector.so +
# myvector.json, so mount them there. Ask the image for its plugin_dir rather
# than hardcoding it: the official Oracle-Linux-based images use
# /usr/lib64/mysql/plugin, Debian-based ones /usr/lib/mysql/plugin.
PLUGIN_DIR=$($DOCKER run --rm --entrypoint mysqld "mysql:$MYSQL_TAG" --verbose --help 2>/dev/null \
  | awk '$1 == "plugin-dir" {print $2}')
PLUGIN_DIR="${PLUGIN_DIR%/}"
[ -n "$PLUGIN_DIR" ] || { echo "ERROR: could not read plugin-dir from mysql:$MYSQL_TAG" >&2; exit 1; }

# myvector.cnf: MYVECTOR_INDEX_BUILD (and online index updates) open their own
# connection back to the server, configured from myvector.cnf in mysqld's working
# directory (the datadir). Without it the build fails ("Can't connect to local
# MySQL server through socket ''"), the index stays empty and every ANN query
# returns nothing (recall 0). A raw mysql:<tag> image has no such file, so write
# it from an initdb hook that sorts BEFORE zz-myvector-procs.sql (which may
# INSTALL the component) -- the same recipe as MyVector's own demo container.
# Root credentials are fine here: this is a throwaway bench container.
CNF_HOOK=$(mktemp "${TMPDIR:-/tmp}/myvector-cnf-hook.XXXXXX")
# The hook only runs on first boot (initdb), so it's removed when this script
# exits. The container is throwaway (re-run this script for a fresh one);
# `docker restart` on it would fail on the now-missing bind-mount source.
trap 'rm -f "$CNF_HOOK"' EXIT
cat >"$CNF_HOOK" <<'HOOK'
cat >/var/lib/mysql/myvector.cnf <<CNF
myvector_host=127.0.0.1
myvector_port=3306
myvector_user_id=root
myvector_user_password=${MYSQL_ROOT_PASSWORD}
CNF
chmod 600 /var/lib/mysql/myvector.cnf
HOOK
chmod 644 "$CNF_HOOK"

echo "Starting mysql:$MYSQL_TAG as '$NAME' with the MyVector component..." >&2
$DOCKER run -d --name "$NAME" -p "$HOST:${PORT}:3306" \
  -e MYSQL_ROOT_PASSWORD="$ROOT_PW" \
  -v "$MYVECTOR_SO":"$PLUGIN_DIR"/myvector.so:ro \
  -v "$MYVECTOR_JSON":"$PLUGIN_DIR"/myvector.json:ro \
  -v "$CNF_HOOK":/docker-entrypoint-initdb.d/00-myvector-cnf.sh:ro \
  -v "$MYVECTOR_PROCS":/docker-entrypoint-initdb.d/zz-myvector-procs.sql:ro \
  "mysql:$MYSQL_TAG" >/dev/null

mysql_root() { $DOCKER exec -i -e MYSQL_PWD="$ROOT_PW" "$NAME" mysql -uroot -h127.0.0.1 "$@"; }

echo "Waiting for mysqld (timeout ${READY_TIMEOUT}s)..." >&2
ready=""
for _ in $(seq 1 "$READY_TIMEOUT"); do
  if mysql_root -N -e "SELECT 1" >/dev/null 2>&1; then ready=1; break; fi
  sleep 1
done
[ -n "$ready" ] || { echo "ERROR: mysqld did not become ready" >&2; $DOCKER logs --tail 40 "$NAME" >&2; exit 1; }

echo "Installing MyVector component + UDFs..." >&2
# The component registers its own UDFs (myvector_construct, myvector_distance,
# myvector_ann_set, ...); myvector_row_distance & co. are supplemental and need
# CREATE FUNCTION ... SONAME. The stored procedures come from MYVECTOR_PROCS
# (run by the init dir on first boot) -- which may itself INSTALL the component
# and CREATE those functions, so each step below is skipped if already present.
# That keeps the install idempotent WITHOUT swallowing errors: any failure here
# aborts the script (set -e) instead of printing READY on a broken server.
count() { mysql_root -N -e "$1"; }
udf_count() {
  count "SELECT COUNT(*) FROM performance_schema.user_defined_functions WHERE udf_name = '$1';"
}

if [ "$(count "SELECT COUNT(*) FROM mysql.component WHERE component_urn = 'file://myvector';")" = "0" ]; then
  mysql_root -e "INSTALL COMPONENT 'file://myvector';"
fi
for spec in "myvector_row_distance REAL" "myvector_is_valid INTEGER" \
            "myvector_search_open_udf STRING"; do
  fn="${spec% *}"; ret="${spec#* }"
  if [ "$(udf_count "$fn")" = "0" ]; then
    mysql_root -e "CREATE FUNCTION $fn RETURNS $ret SONAME 'myvector.so';"
  fi
done
mysql_root -e "
  CREATE USER IF NOT EXISTS 'bench'@'%' IDENTIFIED BY 'bench';
  GRANT ALL PRIVILEGES ON *.* TO 'bench'@'%' WITH GRANT OPTION;
  FLUSH PRIVILEGES;
"

# Sanity: every object the harness profile depends on must exist. Check
# performance_schema, not mysql.func: the component's own UDFs are registered
# dynamically and never appear in mysql.func.
missing=""
for fn in myvector_construct myvector_distance myvector_ann_set myvector_row_distance; do
  if [ "$(udf_count "$fn")" = "0" ]; then missing="$missing $fn()"; fi
done
# The profile's build_sql calls DROP then BUILD, so both procedures are required.
for proc in MYVECTOR_INDEX_DROP MYVECTOR_INDEX_BUILD; do
  if [ "$(count "SELECT COUNT(*) FROM information_schema.routines
                 WHERE routine_schema = 'mysql' AND routine_name = '$proc';")" = "0" ]; then
    missing="$missing mysql.$proc"
  fi
done
# The profile declares online=Y (MyVector follows the binlog to index rw_bench's
# concurrent writes), which needs row-based binary logging -- the stock image's
# default, but check rather than assume.
if [ "$(count "SELECT CONCAT(@@log_bin, @@binlog_format);")" != "1ROW" ]; then
  missing="$missing log_bin=ON+binlog_format=ROW"
fi
if [ -n "$missing" ]; then
  echo "ERROR: MyVector objects missing:$missing -- check the component build" \
       "and MYVECTOR_PROCS." >&2
  exit 1
fi

echo "READY host=$HOST port=$PORT" >&2
echo "tcp:$HOST:$PORT"
