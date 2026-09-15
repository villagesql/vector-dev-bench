#!/bin/bash
# run_three_way.sh — vsql vs MariaDB vs pgvector, ONE ENGINE AT A TIME.
#
# Runs run_sweep.sh once per engine so only one DB server is ever live (each
# sweep starts + stops its own). This avoids the memory contention that skews
# wall-clock when multiple 4G-buffer-pool servers run concurrently. Each
# run_sweep re-execs under caffeinate, so the whole thing is sleep-safe.
#
# Point it at your three builds via env, then just run it:
#   SRV_BUILD=/path/to/villagesql/build \
#   MB=/opt/homebrew/opt/mariadb@11.8 \
#   PGBIN=/opt/homebrew/opt/postgresql@17/bin \
#     bash run_three_way.sh
#
# Same tuning knobs as run_sweep.sh: MODE, N_LIST, DIM, M, EFC, EF_SWEEP,
# METRIC, QUERIES, K, BUF, REPS. Defaults inherited from run_sweep.sh.
#
# NOTE ASYMMETRIES the numbers carry (state them in any writeup):
#   - MariaDB ignores EFC (hardcoded ef_construction=10 upstream), so at EFC>10
#     it builds with a different (lower) construction budget than the others.
#   - pgvector's native fast path is a bulk `CREATE INDEX`; here every engine is
#     driven incrementally (index present during insert) for an apples-to-apples
#     build model. This is not pgvector's fastest path. See RUNBOOK.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENGINES="${ENGINES:-vsql mariadb pgvector}"
failed=""
for eng in $ENGINES; do
  echo "======================== $eng ========================"
  if ! ENGINE="$eng" bash "$HERE/run_sweep.sh"; then
    failed="$failed $eng"
  fi
  echo
done
if [ -n "$failed" ]; then
  echo "FAILED:$failed" >&2
  exit 1
fi
echo "ALL DONE"
