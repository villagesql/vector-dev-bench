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

It drives three engines and judges ANN quality by **recall** — the fraction of
the true k-nearest neighbours the index returns — because ANN indexes are
*approximate* and exact-match testing (MTR-style `.result` comparison) is
meaningless for them.

The suite is **three CLIs over a shared `harness_core.py`** (which owns the
profiles, connections, dataset loading, and the shared build path):

| CLI | what it measures |
|-----|------------------|
| `recall_bench.py` | recall@k + QPS + per-query latency across an ef_search sweep (the original; exits non-zero below a recall threshold, so it doubles as a gate). Also `--emit-queries` (dump the validated KNN SQL for a native driver) and `--dry-run` (transport floor). |
| `build_bench.py` | index build time + row-count; `--build-threads N` drives a **client-side parallel build** (N connections inserting into the same index concurrently). |
| `rw_bench.py` | **read-under-write**: N reader threads run KNN while M writer threads insert into the same index; reports per-second read QPS over time (read-alone → dip → recover), write rows/s, and read-latency p50/p95/p99 per phase. |

All three build the index first (via `harness_core.build_index`), then do their
own thing. They share the same `--profile`/`--metric`/`--dim`/`--n`/`--M`/
`--ef-construction`/`--dataset`/`--socket`/`--mysql` flags.

## Supported engines (profiles)

| profile        | engine                    | client | vector input | index |
|----------------|---------------------------|--------|--------------|-------|
| `vsql_vector`  | VillageSQL SVECTOR + HNSW | mysql  | text `'[...]'` | `CREATE INDEX ... USING EXTENDED(hnsw)` |
| `mariadb`      | MariaDB MHNSW             | mariadb| `Vec_FromText` | inline `VECTOR INDEX` |
| `pgvector`     | PostgreSQL + pgvector     | psql   | text `'[...]'` | `CREATE INDEX ... USING hnsw` |

> **Client note (fairness).** The `client` column names the SQL *dialect*, not the
> transport. `vsql_vector` and `mariadb` both speak the MySQL protocol and are
> driven by the **same PyMySQL code path** — identical connection setup, batching,
> and result handling; only the emitted SQL differs. So any client-side overhead is
> symmetric between them and cancels out in relative comparison. `pgvector` is the
> one asymmetric client: it shells out to the `psql` CLI. (The `--mysql` flag is
> therefore vestigial for the query path — PyMySQL drives it regardless of which
> `mysql`/`mariadb` binary you point at; the flag still feeds the `start_*.sh`
> scripts and config reporting.)

## What this is NOT

This is the **development** harness. For a rigorous, external-facing comparison
(real ANN datasets like SIFT/GloVe with true ground truth, concurrency scaling,
p50/p99 latency, normalized-vs-tuned fairness passes, reproducible env manifests)
use a full framework such as ann-benchmarks or vector-bench. This harness trades
that rigour for a fast iteration loop:

- **Data is synthetic by default, real datasets optional.** With no `--dataset`,
  vectors are `numpy.random.standard_normal` and ground truth is computed exactly
  in numpy — a trustworthy *relative* signal for iterating that does not claim to
  reflect real-embedding recall. Pass `--dataset NAME` to run a standard
  ann-benchmarks dataset (e.g. `fashion-mnist-784-euclidean`) with its
  precomputed ground truth instead; that path uses real embeddings.
- **Client overhead is in the numbers.** QPS/latency go through a Python client
  (pymysql), so absolute latency includes client + round-trip overhead (~20µs
  transport floor — see `recall_bench.py --dry-run`). Use for *relative*
  comparisons. All three CLIs can drive real concurrency (multiprocessing, one
  connection per worker, no GIL): `recall_bench.py --readers N` fans queries
  across N reader processes for aggregate QPS, `build_bench --build-threads N`
  shards the build, and `rw_bench` runs readers against writers. Concurrency is
  for measuring contention/scaling, not absolute single-query latency (at
  `--readers 1` the query phase is serial per batch).
- **Build time** is real and the most trustworthy number.
- **read-under-write is a non-standard regime.** Mainstream ANN benchmarks
  (ann-benchmarks, VectorDBBench) are build-once-then-static-query. `rw_bench`
  fills the gap between those and YCSB/sysbench-style mixed-workload testing —
  see the read/write section below.

## Prerequisites

- Built server(s) for whichever engine(s) you want to run:
  - **vsql**: a VillageSQL server build + the `vsql_vector` extension built
    against it (staged into the server's `veb_output_directory/`). **Both must be
    the in-development custom-index feature branches, NOT `main`** (see the
    warning above):
    - the server built with `-DWITH_HYPERGRAPH_OPTIMIZER=ON`, from a branch that
      carries the custom vector index and the read-under-write fix (a
      `row_not_found` skip in the KNN scan).
    - the `vsql_vector` extension, built `Release`/`-O3` against that server's
      SDK, from a branch that carries the full-ef-pool cursor for read-under-
      write backfill.
    (These are in-flight feature branches that carry the vector index until it
    lands on `main`. `rw_bench.py` in particular NEEDS both fixes — without them
    concurrent KNN-read + INSERT fails with error 122.)
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
build via env vars, then run the CLI you want against that socket.

### recall + QPS (recall_bench.py)
```bash
SRV_BUILD=/path/to/villagesql/build \
EXTENSIONS=vsql_vector \
  bash start_server.sh                       # boots on .run/mysqld.sock

python recall_bench.py --profile vsql_vector --metric l2 --dim 32 \
  --n 20000 --M 8 --ef-construction 100 --ef-search-sweep 50,100,200,400 \
  --mysql /path/to/villagesql/build/runtime_output_directory/mysql
```

MariaDB / pgvector are the same, with their profile + start script:
```bash
MB=/opt/homebrew/opt/mariadb@11.8 bash start_mariadb.sh
python recall_bench.py --profile mariadb --metric l2 --dim 32 --n 20000 \
  --M 8 --ef-search-sweep 100,200,400 \
  --mysql /opt/homebrew/opt/mariadb@11.8/bin/mariadb --socket .run-maria/mariadb.sock

SHARED_BUFFERS=4GB MAINT_WORK_MEM=2GB bash start_postgres.sh
python recall_bench.py --profile pgvector --metric l2 --dim 32 --n 20000 \
  --M 8 --ef-construction 100 --ef-search-sweep 100,200,400 \
  --mysql /opt/homebrew/opt/postgresql@17/bin/psql --socket .run-pg
```

### parallel build (build_bench.py)
Client-driven parallel index build — N connections insert into the same index
concurrently. (Server-internal parallel-DDL `innodb_parallel_read_threads`
DEGRADES; client-side sharding is the working path, ~3x at 4 threads.)
```bash
python build_bench.py --profile vsql_vector --dataset fashion-mnist-784-euclidean \
  --M 16 --ef-construction 200 --build-threads 4 \
  --mysql .../mysql --socket .run/mysqld.sock
# reports build_time_s + rows=N/N committed
# --no-index isolates plain-column insert cost from graph-build cost.
```

### read-under-write (rw_bench.py)
N readers run KNN while M writers insert random vectors into the SAME index for a
window in the middle. Reports per-second read QPS (read-alone → dip → recover), a
sparkline, write rows/s, and read-latency p50/p95/p99 per phase.
```bash
python rw_bench.py --profile vsql_vector --metric l2 --dim 32 --n 20000 \
  --M 16 --ef-construction 200 --ef-search 100 \
  --readers 4 --rw-duration 12 \
  --rw-write-threads 4 --rw-write-start 4 --rw-write-duration 4 --rw-write-delay 0 \
  --mysql .../mysql --socket .run/mysqld.sock
```
Note: this exercises a real MVCC read-under-write path — see the server fix that
made concurrent KNN-read + INSERT work at REPEATABLE READ (full-ef-pool cursor +
skip-on-not-found). `--rw-read-uncommitted` is a diagnostic (READ UNCOMMITTED
reader); `--rw-read-sql "SELECT MAX(id) FROM t"` is a plain-read control.

## Key flags

Shared (all three CLIs):
- `--profile`  vsql_vector | mariadb | pgvector | varchar_ctrl (a plain-VARCHAR
  control — no vector engine, insert-only; use to prove a failure is server-side
  vs harness)
- `--metric`   l2 | cosine | l1 | ip   (must be supported by the profile)
- `--dim --n -k`   dataset shape;  `--dataset NAME`  real ann-benchmarks dataset
  (precomputed ground truth; sets dim + metric)
- `--M --ef-construction`    HNSW build params (MariaDB ignores ef_construction —
  hardcoded to 10 upstream)
- `--no-index`   build the column with no index (build-floor / insert-cost probe)
- `--insert-batch N`   rows per INSERT statement (default 1000)
- `--mysql --socket` / `--host --port`   client + connection

recall_bench.py:
- `--queries N` / `-k` / `--ef-search N` / `--ef-search-sweep a,b,c` (build once,
  re-query at each ef — the recall/QPS Pareto) / `--threshold` (recall gate,
  default 0.95) / `--emit-queries FILE` / `--dry-run` / `--build-only`

build_bench.py:
- `--build-threads N`   client-side parallel build (N concurrent inserters)

rw_bench.py:
- `--readers N` / `--rw-duration S` / `--rw-write-threads M` / `--rw-write-start S`
  / `--rw-write-duration S` / `--rw-write-delay S` (0 = busy-write; dials write
  pressure) / `--rw-write-batch N` / `--rw-csv FILE` / `--rw-read-uncommitted`
  (diagnostic) / `--rw-read-sql SQL` (control)

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
