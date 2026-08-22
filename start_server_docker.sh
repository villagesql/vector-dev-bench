#!/bin/bash
# start_server_docker.sh -- boot a vector-bench engine from a prebuilt DOCKER
# IMAGE (e.g. one pulled from the registry) and expose it over TCP for the
# recall harness. The container's own entrypoint installs the extension + sets
# the KNN gates + creates the bench account, so unlike the native
# start_server.sh there is nothing to configure here -- we just run it, publish
# the port, and wait until it answers.
#
# Mirrors start_server.sh / start_postgres.sh: fresh state each run, prints the
# connection spec on success. But it prints a TCP host/port (not a socket),
# because a container's unix socket lives inside the container. Drive the
# harness with:  recall_harness.py --host 127.0.0.1 --port $PORT --mysql <client>
#
# Env:
#   IMAGE   REQUIRED. The image to run, e.g.
#           us-central1-docker.pkg.dev/PROJECT/REPO/villagesql-runtime:TAG
#   PORT    host port to publish (default 3306). The harness --port.
#   NAME    container name (default vdb-engine)
#   SERVER_ARGS  extra args appended to the image's `server` command
#                (e.g. --innodb-buffer-pool-size=4G for perf runs)
#   READY_TIMEOUT  seconds to wait for readiness (default 120)
#
# Prints on success:  a line "READY host=127.0.0.1 port=$PORT" to stderr, and
# the bare "127.0.0.1:$PORT" to stdout (so a caller can capture it).
set -euo pipefail

IMAGE="${IMAGE:-}"
PORT="${PORT:-3306}"
NAME="${NAME:-vdb-engine}"
READY_TIMEOUT="${READY_TIMEOUT:-120}"
HOST="127.0.0.1"

if [ -z "$IMAGE" ]; then
  echo "ERROR: set IMAGE=<registry>/<engine>-runtime:<tag> (the image to run)" >&2
  exit 1
fi

# `docker` may need sudo on a fresh VM (user not yet in the docker group).
DOCKER="docker"
docker info >/dev/null 2>&1 || DOCKER="sudo docker"

# Fresh container each run: remove any prior one on this name (its datadir goes
# with it, so the extension re-installs cleanly on first boot -- matching the
# native harness's fresh-datadir-per-run contract).
$DOCKER rm -f "$NAME" >/dev/null 2>&1 || true

echo "Starting $IMAGE as '$NAME' (publishing $HOST:$PORT -> 3306)..." >&2
# The image's ENTRYPOINT is vb-entrypoint; `server` starts mysqld (init on first
# boot). --mysql-native-password / preview gate / extension install are all done
# by the entrypoint + init.sql inside the image.
# shellcheck disable=SC2086  # SERVER_ARGS is intentionally word-split
$DOCKER run -d --name "$NAME" -p "${PORT}:3306" "$IMAGE" server ${SERVER_ARGS:-} \
  >/dev/null

# Wait for the server to accept a TCP query. Probe from INSIDE the container
# (its own mysql client over its socket) -- that avoids needing a client on the
# host just to check readiness, and confirms the server + extension came up.
ready=""
for _ in $(seq 1 "$READY_TIMEOUT"); do
  if $DOCKER exec "$NAME" /opt/villagesql/bin/mysql \
        --socket=/var/run/vbench/villagesql.sock -ubench -pbench \
        -e "SELECT 1" >/dev/null 2>&1; then
    ready=1; break
  fi
  # bail early if the container died (e.g. init error)
  if [ -z "$($DOCKER ps -q -f name="^${NAME}$")" ]; then
    echo "ERROR: container '$NAME' exited during startup; logs:" >&2
    $DOCKER logs --tail 40 "$NAME" >&2 || true
    exit 1
  fi
  sleep 1
done

if [ -z "$ready" ]; then
  echo "ERROR: server did not become ready in ${READY_TIMEOUT}s; logs:" >&2
  $DOCKER logs --tail 40 "$NAME" >&2 || true
  exit 1
fi

# Confirm the extension is actually installed (guards against a server that
# came up but failed to load vsql_vector).
ext="$($DOCKER exec "$NAME" /opt/villagesql/bin/mysql \
        --socket=/var/run/vbench/villagesql.sock -ubench -pbench -N \
        -e "SELECT COUNT(*) FROM information_schema.extensions" 2>/dev/null || echo 0)"
echo "READY host=$HOST port=$PORT container=$NAME extensions=$ext" >&2
echo "${HOST}:${PORT}"
