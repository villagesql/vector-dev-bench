#!/bin/bash
# setup.sh — create the local Python venv and install the harness deps.
#
# One-time setup for a fresh clone:
#   ./setup.sh
# Then the run scripts (run_sweep.sh / run_three_way.sh) auto-select .venv/bin/python.
#
# This installs ONLY the Python side (numpy/pymysql/h5py). The database engines
# (villagesql+vsql_vector, MariaDB, pgvector) are separate prerequisites — see
# the README.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || { echo "error: '$PY' not found; set PYTHON=/path/to/python3" >&2; exit 1; }

if [ ! -x ".venv/bin/python" ]; then
  echo "==> creating venv at .venv (using $PY)"
  "$PY" -m venv .venv
fi

echo "==> installing dependencies from requirements.txt"
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements.txt

echo "==> verifying the harness imports"
# Import the harness itself (single source of truth = requirements.txt, which
# pip just installed). harness_core pulls numpy; import it to confirm the env
# is usable. pip already fails above if any requirement did not install.
.venv/bin/python -c "import harness_core; print('    harness imports OK')"

echo "==> done. The run scripts will use .venv automatically."
echo "    (or activate it yourself: source .venv/bin/activate)"
