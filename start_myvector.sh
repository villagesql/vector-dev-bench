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
#   PORT            host port to publish (default 3306). The harness --port / tcp:.
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
PORT="${PORT:-3306}"
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

echo "Starting mysql:$MYSQL_TAG as '$NAME' with the MyVector component..." >&2
# Plugin dir is /usr/lib/mysql/plugin in the official image; the component loader
# resolves 'file://myvector' to <plugindir>/myvector.so + myvector.json.
$DOCKER run -d --name "$NAME" -p "${PORT}:3306" \
  -e MYSQL_ROOT_PASSWORD="$ROOT_PW" \
  -v "$MYVECTOR_SO":/usr/lib/mysql/plugin/myvector.so:ro \
  -v "$MYVECTOR_JSON":/usr/lib/mysql/plugin/myvector.json:ro \
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
# myvector_distance auto-registers (dynamic UDF); the rest need CREATE FUNCTION.
# The stored procedures come from MYVECTOR_PROCS (run by the init dir on first
# boot). myvector.cnf lets the index build connect back to the server.
mysql_root -e "
  INSTALL COMPONENT 'file://myvector';
  CREATE FUNCTION myvector_row_distance    RETURNS REAL    SONAME 'myvector.so';
  CREATE FUNCTION myvector_is_valid        RETURNS INTEGER SONAME 'myvector.so';
  CREATE FUNCTION myvector_search_open_udf RETURNS STRING  SONAME 'myvector.so';
  CREATE USER IF NOT EXISTS 'bench'@'%' IDENTIFIED BY 'bench';
  GRANT ALL PRIVILEGES ON *.* TO 'bench'@'%' WITH GRANT OPTION;
  FLUSH PRIVILEGES;
" 2>&1 | grep -vi "already exists" || true

# Sanity: the query-side UDF the harness profile depends on must exist.
if ! mysql_root -N -e "SELECT COUNT(*) FROM mysql.func WHERE name='myvector_ann_set';" 2>/dev/null | grep -q '^[1-9]'; then
  echo "WARNING: myvector_ann_set not registered -- check the component build / init SQL." >&2
fi

echo "READY host=$HOST port=$PORT" >&2
echo "tcp:$HOST:$PORT"
