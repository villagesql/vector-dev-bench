#!/bin/bash
# run_sweep.sh — build-time and/or scan (recall/QPS) sweep for ONE engine,
# run cleanly: caffeinated (no machine sleep) and isolated (this is the only
# DB server the script starts). Wraps recall_bench.py with the right start
# script and client for the chosen engine.
#
# Usage:
#   ENGINE=vsql   SRV_BUILD=/path/to/villagesql/build           bash run_sweep.sh
#   ENGINE=mariadb MB=/opt/homebrew/opt/mariadb@11.8            bash run_sweep.sh
#   ENGINE=pgvector PGBIN=/opt/homebrew/opt/postgresql@17/bin   bash run_sweep.sh
#
# Env knobs (all optional, sane defaults):
#   ENGINE        vsql | mariadb | pgvector            (required)
#   MODE          build | scan | both                 (default both)
#   N_LIST        space-separated row counts           (default "1000 5000 20000")
#   DIM, M, EFC   vector dim / HNSW M / ef_construction (default 32 / 8 / 100)
#   EF_SWEEP      comma ef_search values for scan       (default 100,200,400)
#   METRIC        l2 | cosine | l1 | ip                 (default l2)
#   QUERIES, K    query count / k                       (default 50 / 10)
#   BUF           InnoDB buffer pool (vsql/maria)       (default 4G)
#   REPS          build-only reps per N                 (default 2)
#   engine paths: SRV_BUILD (vsql), MB (mariadb), PGBIN (pgvector)
set -uo pipefail

# Re-exec under caffeinate (macOS) so an idle/locked machine can't throttle a
# long run ~13x, as observed. VB_CAFFEINATED guards against infinite re-exec.
if [ -z "${VB_CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  exec env VB_CAFFEINATED=1 caffeinate -dimsu bash "$0" "$@"
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="${ENGINE:?set ENGINE=vsql|mariadb|pgvector}"
MODE="${MODE:-both}"
N_LIST="${N_LIST:-1000 5000 20000}"
DIM="${DIM:-32}"; M="${M:-8}"; EFC="${EFC:-100}"
EF_SWEEP="${EF_SWEEP:-100,200,400}"; METRIC="${METRIC:-l2}"
QUERIES="${QUERIES:-50}"; K="${K:-10}"; BUF="${BUF:-4G}"; REPS="${REPS:-2}"
# Prefer the repo's own venv (created by ./setup.sh) over system python, so a
# user who ran setup.sh doesn't silently fall back to a python without numpy.
# Override with PY=/path/to/python if you manage the environment yourself.
if [ -n "${PY:-}" ]; then :
elif [ -x "$HERE/.venv/bin/python" ]; then PY="$HERE/.venv/bin/python"
else PY=python3
fi
# Fail early with a clear message if the harness can't even import, rather than
# letting every per-N run crash and still print DONE. This imports the harness
# itself (harness_core) instead of restating the dependency list, so there is a
# single source of truth (requirements.txt). harness_core imports numpy eagerly;
# the per-feature deps (h5py for --dataset, pymysql for the pymysql path) are
# lazy and will raise a clear ImportError at point of use if actually needed.
if ! "$PY" -c "import harness_core" 2>/dev/null; then
  echo "error: python at '$PY' can't load the harness (missing deps?)." >&2
  echo "       run ./setup.sh first, or set PY=/path/to/python with the deps in requirements.txt." >&2
  exit 1
fi

# --- resolve engine: start script, client binary, socket, profile ---------
case "$ENGINE" in
  vsql)
    : "${SRV_BUILD:?set SRV_BUILD=/path/to/villagesql/build}"
    PROFILE=vsql_vector
    CLIENT="$SRV_BUILD/runtime_output_directory/mysql"
    SOCKET="$HERE/.run/mysqld.sock"
    start() { MYSQLD_EXTRA="--innodb-buffer-pool-size=$BUF" SRV_BUILD="$SRV_BUILD" \
              EXTENSIONS="${EXTENSIONS:-vsql_vector}" bash "$HERE/start_server.sh" >/dev/null 2>&1; }
    stop()  { [ -f "$HERE/.run/mysqld.pid" ] && kill "$(cat "$HERE/.run/mysqld.pid")" 2>/dev/null; }
    ;;
  mariadb)
    : "${MB:?set MB=/path/to/mariadb (build tree or installed keg)}"
    PROFILE=mariadb
    CLIENT="$MB/bin/mariadb"; [ -x "$CLIENT" ] || CLIENT="$MB/client/mariadb"
    SOCKET="$HERE/.run-maria/mariadb.sock"
    start() { MARIADBD_EXTRA="--innodb-buffer-pool-size=$BUF" MB="$MB" \
              bash "$HERE/start_mariadb.sh" >/dev/null 2>&1; }
    stop()  { [ -f "$HERE/.run-maria/mariadb.pid" ] && kill "$(cat "$HERE/.run-maria/mariadb.pid")" 2>/dev/null; }
    ;;
  pgvector)
    PGBIN="${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}"
    PROFILE=pgvector
    CLIENT="$PGBIN/psql"
    SOCKET="$HERE/.run-pg"
    # Postgres uses MB/GB units, not the MySQL-style "4G" that BUF carries.
    # Translate 4G -> 4GB (append B to a bare-suffix value) so start_postgres
    # gets a valid shared_buffers.
    PG_BUF="$BUF"; case "$PG_BUF" in *[GMK]) PG_BUF="${PG_BUF}B";; esac
    start() { SHARED_BUFFERS="$PG_BUF" MAINT_WORK_MEM="${MAINT_WORK_MEM:-2GB}" \
              PGBIN="$PGBIN" bash "$HERE/start_postgres.sh" >/dev/null 2>&1; }
    stop()  { [ -d "$HERE/.run-pg/data" ] && "$PGBIN/pg_ctl" -D "$HERE/.run-pg/data" stop -m immediate >/dev/null 2>&1; }
    ;;
  *) echo "unknown ENGINE=$ENGINE" >&2; exit 2;;
esac

hb() { "$PY" "$HERE/recall_bench.py" --profile "$PROFILE" --metric "$METRIC" \
        --dim "$DIM" --M "$M" --ef-construction "$EFC" --insert-batch 1000 \
        --mysql "$CLIENT" --socket "$SOCKET" "$@"; }

# Run recall_bench.py, tee its full output to a temp file so we can (a) filter
# the summary lines for the console AND (b) still detect a crash. Returns the
# child's exit status (via PIPESTATUS) so the caller can fail loud instead of
# printing DONE over a silent traceback.
hb_run() {  # $1 = grep pattern for the lines to surface
  local pat="$1"; shift
  local tmp; tmp="$(mktemp)"
  hb "$@" >"$tmp" 2>&1
  local rc=$?
  grep -E "$pat" "$tmp" | sed 's/^/  /'
  if [ "$rc" -ne 0 ]; then
    echo "  !! recall_bench.py FAILED (exit $rc); last lines:" >&2
    tail -5 "$tmp" | sed 's/^/  !! /' >&2
    rm -f "$tmp"; return "$rc"
  fi
  rm -f "$tmp"; return 0
}

run() {
  stop; sleep 2; start
  echo "### $ENGINE  metric=$METRIC dim=$DIM M=$M efc=$EFC  buf=$BUF"
  local failed=0
  for N in $N_LIST; do
    if [ "$MODE" = build ] || [ "$MODE" = both ]; then
      for r in $(seq 1 "$REPS"); do
        printf "build N=%-8s rep%d\n" "$N" "$r"
        # --build-only isolates build timing: it skips the query phase entirely
        # and returns 0 on a complete build. (Do NOT use --queries 0 here: with
        # --dataset that means "run ALL test queries" and then gates on the 0.95
        # recall threshold, so a build-timing step would spuriously fail at the
        # default ef_search.)
        hb_run 'build_time_s=[0-9.]+|ERROR' --build-only --n "$N" || failed=1
      done
    fi
    if [ "$MODE" = scan ] || [ "$MODE" = both ]; then
      echo "scan N=$N (ef_search sweep $EF_SWEEP):"
      # --no-gate: this is a cross-engine COMPARISON, so a lower recall is a
      # result to report, not a harness failure. (recall_bench.py used directly
      # keeps the 0.95 gate as a single-engine correctness check.) Real errors
      # still fail: hb_run flags any non-zero exit that isn't the recall gate.
      hb_run '^[0-9]|build_time|ERROR' --no-gate --n "$N" --queries "$QUERIES" -k "$K" --ef-search-sweep "$EF_SWEEP" || failed=1
    fi
  done
  stop
  if [ "$failed" -ne 0 ]; then
    echo "FAILED $ENGINE (see errors above)" >&2
    return 1
  fi
  echo "DONE $ENGINE"
}

run
