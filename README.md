# vector-dev-bench

A lightweight, local (no Docker) harness for **iterating on in-database vector
(HNSW) indexes** — built for the fast rebuild-and-measure development loop, not
for heavyweight published comparisons.

> ⚠️ **vsql_vector requires in-development branches — it does NOT work on the
> `main` branch of villagesql-server or vsql-vector.** The custom vector index
> (`CREATE INDEX ... USING EXTENDED(hnsw)`) is not yet enabled on `main`: on main
> the DDL is gated off (returns "Extended Index feature not yet implemented" in
> non-debug builds) and the hypergraph optimizer that routes the KNN scan is not
> compiled into release builds. You must build from the feature branches (see
> Prerequisites) and with `-DWITH_HYPERGRAPH_OPTIMIZER=ON`. MariaDB and pgvector
> work from their normal releases; this constraint is vsql-only.

It drives three engines through one harness and judges results by **recall** —
the fraction of the true k-nearest neighbours the index returns — because ANN
indexes are *approximate* and exact-match testing (what MTR-style `.result`
comparison does) is meaningless for them. It reports **recall@k**, **QPS /
per-query latency**, and **index build time**, and exits non-zero if mean recall
is below a threshold (so it doubles as a gate).

## Supported engines (profiles)

| profile        | engine                    | client | vector input | index |
|----------------|---------------------------|--------|--------------|-------|
| `vsql_vector`  | VillageSQL SVECTOR + HNSW | mysql  | text `'[...]'` | `CREATE INDEX ... USING EXTENDED(hnsw)` |
| `mariadb`      | MariaDB MHNSW             | mariadb| `Vec_FromText` | inline `VECTOR INDEX` |
| `pgvector`     | PostgreSQL + pgvector     | psql   | text `'[...]'` | `CREATE INDEX ... USING hnsw` |

## What this is NOT

This is the **development** harness. For a rigorous, external-facing comparison
(real ANN datasets like SIFT/GloVe with true ground truth, concurrency scaling,
p50/p99 latency, normalized-vs-tuned fairness passes, reproducible env manifests)
use a full framework such as ann-benchmarks or vector-bench. This harness trades
that rigour for a fast iteration loop:

- **Synthetic data.** Vectors are `numpy.random.standard_normal`, not real
  embeddings — ground truth is computed exactly in numpy. Recall numbers are a
  trustworthy *relative* signal for iterating; they do not claim to reflect
  real-embedding recall.
- **Single-threaded, client-per-batch.** QPS/latency go through the CLI one query
  batch at a time (no connection pooling / concurrency), so absolute latency
  includes client overhead. Use it for *relative* comparisons; subtract the
  PK-order floor for a cleaner per-query compute estimate.
- **Build time** is real and the harness's most trustworthy number.

## Prerequisites

- Built server(s) for whichever engine(s) you want to run:
  - **vsql**: a VillageSQL server build + the `vsql_vector` extension built
    against it (staged into the server's `veb_output_directory/`). **Both must be
    the custom-index feature branches, NOT `main`** (see the warning above):
    - server `villagesql/villagesql-server` @ branch `tomas/deb-6-optimizer-scan`,
      built with `-DWITH_HYPERGRAPH_OPTIMIZER=ON`.
    - extension `villagesql/vsql-vector` @ branch `tomas/deb-absolute-minimal-bridge`,
      built `Release`/`-O3` against that server's SDK.
    (Branch names are point-in-time; these are the in-flight branches that carry
    the feature until it lands on `main`.)
  - **mariadb**: a MariaDB 11.7+ build tree *or* an installed package (e.g.
    Homebrew `mariadb@11.8`). Both layouts are auto-detected.
  - **pgvector**: PostgreSQL 15+ with the `vector` extension available.
- Python venv with numpy (and h5py for real `--dataset` runs):
  ```
  python3 -m venv .venv && . .venv/bin/activate && pip install numpy h5py
  ```
  (mariadb runs also want the `mariadb` Python connector if you extend past the
  CLI path; the current harness shells out to the CLI clients.)

## Usage

Each engine has a `start_*.sh` that boots a throwaway scratch cluster (fresh
datadir, unix socket, gates baked in) and prints its socket. Point it at your
build via env vars, then run `recall_harness.py` against that socket.

### vsql
```bash
SRV_BUILD=/path/to/villagesql/build \
EXTENSIONS=vsql_vector \
  bash start_server.sh                       # boots on .run/mysqld.sock

python recall_harness.py --profile vsql_vector --metric l2 --dim 32 \
  --n 20000 --M 8 --ef-construction 100 --ef-search-sweep 50,100,200,400 \
  --mysql /path/to/villagesql/build/runtime_output_directory/mysql
```

### MariaDB (build tree OR installed package)
```bash
MB=/opt/homebrew/opt/mariadb@11.8 bash start_mariadb.sh   # install layout
# or:  MB=~/githome/mariadb-server/build-release bash start_mariadb.sh  # build tree

python recall_harness.py --profile mariadb --metric l2 --dim 32 --n 20000 \
  --M 8 --ef-search-sweep 100,200,400 \
  --mysql /opt/homebrew/opt/mariadb@11.8/bin/mariadb \
  --socket .run-maria/mariadb.sock
```

### pgvector
```bash
SHARED_BUFFERS=4GB MAINT_WORK_MEM=2GB bash start_postgres.sh   # .run-pg socket dir

python recall_harness.py --profile pgvector --metric l2 --dim 32 --n 20000 \
  --M 8 --ef-construction 100 --ef-search-sweep 100,200,400 \
  --mysql /opt/homebrew/opt/postgresql@17/bin/psql --socket .run-pg
```

## Key flags

- `--profile`  vsql_vector | mariadb | pgvector
- `--metric`   l2 | cosine | l1 | ip   (must be supported by the profile)
- `--dim --n --queries -k`   dataset shape
- `--M --ef-construction`    HNSW build params (note: MariaDB ignores
  ef_construction — it is hardcoded to 10 upstream)
- `--ef-search N` / `--ef-search-sweep a,b,c`   query-time breadth; a sweep builds
  the index once then re-queries at each value (the recall/QPS Pareto)
- `--queries 0`   build-only mode (report build time, skip queries — useful when
  an engine can't route the KNN query, or for pure build benchmarking)
- `--no-index`   build the column with no index (brute-force / build-floor probe)
- `--insert-batch N`   rows per INSERT statement (default 1000; `0` = one giant
  statement — parse-cost probe only, exceeds MySQL/MariaDB max_allowed_packet on
  real datasets)
- `--threshold`   pass/fail recall gate (default 0.95)

## Methodology notes (learned the hard way)

- **Compile with optimization.** Verify the engine *and any extension* are built
  `-O2`/`-O3`, not `-O0`. An unoptimized extension build understated results by
  ~6-10x here; the SIMD-vs-scalar codegen difference is enormous for the distance
  kernel.
- **Prevent machine sleep** on long unattended runs: wrap them in
  `caffeinate -dimsu` (macOS). An idle/locked machine throttled one overnight run
  ~13x, silently.
- **Run one engine at a time** for build/scan timing at scale — concurrent
  servers with large buffer pools contend and skew wall-clock.
- **Match config** across engines (buffer pool, durability) and note asymmetries
  you cannot match (e.g. MariaDB's fixed ef_construction=10, pgvector's native
  path being bulk `CREATE INDEX` rather than incremental).
- **Guard against silent full-scan.** Recall pinned at exactly 1.0 that does not
  fall with lower ef_search usually means the index is not being used (the
  optimizer chose a brute-force scan). `EXPLAIN` to confirm the index is in plan.
