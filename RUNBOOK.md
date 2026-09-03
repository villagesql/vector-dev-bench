# RUNBOOK — reproducing the benchmark runs

End-to-end recipe for driving `vector-dev-bench`, including the parts that are
*not* in this repo (building the engines). Read the README first for what the
harness is and its caveats.

## 0. One-time host setup

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install numpy h5py
```

`h5py` is needed for `--dataset` (the ann-benchmarks HDF5 files); `numpy` alone
suffices for the synthetic path.

Everything below assumes you `cd` into this repo and have that venv active. The
runner scripts (`run_sweep.sh` / `run_three_way.sh`) also auto-use `./.venv` if
present, so they work whether or not you have activated it.

---

## 1. Build the engines (not in this repo)

### vsql (VillageSQL server + vsql_vector extension)

Two artifacts: the **server**, and the **extension** built against it and staged
into the server's `veb_output_directory/`.

```bash
# server (RelWithDebInfo == -O2 is fine for the server)
# CRITICAL: -DWITH_HYPERGRAPH_OPTIMIZER=ON. The custom KNN index ONLY routes under
# the hypergraph optimizer, and that optimizer defaults ON *only in debug builds* —
# in an optimized build it is compiled OUT unless you set this flag. Without it the
# server builds fine but the vector index silently never routes (the runtime
# `SET optimizer_switch='hypergraph_optimizer=on'` cannot enable what isn't
# compiled in). This is a COMPILE flag, not a debug-vs-release thing — the build
# stays optimized.
cd <villagesql-server-src>
mkdir -p build-rel && cd build-rel
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DWITH_HYPERGRAPH_OPTIMIZER=ON \
      -DWITH_SSL=/opt/homebrew/opt/openssl@3
make -j"$(getconf _NPROCESSORS_ONLN)"
SRV=$(pwd)

# extension — CRITICAL: build it OPTIMIZED. The extension CMakeLists historically
# set NO build type, so it defaulted to -O0 and understated results ~6-10x.
# ALWAYS pass -DCMAKE_BUILD_TYPE=Release (=> -O3 -DNDEBUG, auto-vectorizes the
# distance kernel). Verify below.
cd <vsql-vector-src>
mkdir -p build-rel && cd build-rel
cmake .. -DVillageSQL_BUILD_DIR="$SRV" -DCMAKE_BUILD_TYPE=Release
make -j"$(getconf _NPROCESSORS_ONLN)"

# VERIFY the extension is actually optimized (the lesson that mattered most):
grep -oE '\-O[0-3s]' CMakeFiles/svector.dir/flags.make | head -1   # want -O3
otool -tv $(find . -name graph.cc.o | head -1) | grep -c 'fmul.4s' # want >0 (NEON)

# STAGE the freshly built .veb into the server's veb dir (the recurring trap:
# a stale staged .veb means you benchmark old code — always re-copy + check mtime)
cp vsql_vector.veb "$SRV/veb_output_directory/"
ls -al "$SRV/veb_output_directory/vsql_vector.veb"   # confirm it's the one you built
```

`SRV_BUILD` for the harness is that `$SRV` (e.g. `.../build-rel`).

**Runtime gates (required for the KNN index to install and route).** `start_server.sh`
applies these automatically at boot; if you drive the server yourself, set them
too, or the index won't install / the custom scan won't route (and the classic
optimizer path crashes on it):
```sql
SET PERSIST vsql_allow_preview_extensions = ON;          -- needed to INSTALL + use the extension
SET GLOBAL optimizer_switch = 'hypergraph_optimizer=on'; -- route the KNN scan (needs the compile flag above)
SET GLOBAL debug = '+d,villagesql_custom_index_proceed'; -- custom-index gate; DEBUG builds only (no-op otherwise)
```

### MariaDB

Either build from source (RelWithDebInfo) **or** install a stock package — the
harness auto-detects both layouts.

```bash
# stock (recommended for a clean, probe-free reference; needs 11.7+ for MHNSW):
brew install mariadb@11.8         # -> MB=/opt/homebrew/opt/mariadb@11.8
# or from source:
#   cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DWITH_SSL=... -DBISON_EXECUTABLE=$(brew --prefix bison)/bin/bison
#   make -j...   -> MB=<mariadb-src>/build-release
```

Note: if you build MariaDB from source, `git status` its tree first — a leftover
instrumentation patch (e.g. an eval-count probe in `sql/vector_mhnsw.cc`) will
compile into the binary and inflate its numbers. Use `git stash`/`checkout` for a
clean reference.

### pgvector

```bash
brew install postgresql@17 pgvector     # -> PGBIN=/opt/homebrew/opt/postgresql@17/bin
```

---

## 2. Sanity smoke (always do this first after a rebuild)

Confirm the engine works and — crucially — that recall is **< 1.0** (a real
approximate index). Recall pinned at exactly 1.0 that does not drop at low
ef_search means the index is NOT being used (full-scan fallback).

```bash
SRV_BUILD=<...> EXTENSIONS=vsql_vector bash start_server.sh
python recall_bench.py --profile vsql_vector --metric l2 --dim 32 --n 2000 \
  --queries 20 -k 10 --M 8 --ef-construction 100 --ef-search 100 \
  --mysql <...>/runtime_output_directory/mysql
# expect: build_time_s=..., recall@10 around 0.95-1.0, no ERROR
kill "$(cat .run/mysqld.pid)"
```

---

## 3. Single-engine sweep (build + scan), caffeinated + isolated

`run_sweep.sh` starts the engine, runs a build sweep (2 reps/N) and an
ef_search scan sweep, then stops it — all under `caffeinate` so an idle machine
can't throttle it.

```bash
# vsql across a few N, build + scan:
ENGINE=vsql SRV_BUILD=<...> bash run_sweep.sh

# MariaDB, scan only, at 1M:
ENGINE=mariadb MB=/opt/homebrew/opt/mariadb@11.8 MODE=scan N_LIST=1000000 bash run_sweep.sh

# pgvector, build only, custom params:
ENGINE=pgvector MODE=build N_LIST="100000 1000000" DIM=32 M=8 EFC=100 bash run_sweep.sh
```

Knobs: `MODE` (build|scan|both), `N_LIST`, `DIM`, `M`, `EFC`, `EF_SWEEP`,
`METRIC`, `QUERIES`, `K`, `BUF`, `REPS`.

---

## 4. Three-way comparison (one engine at a time)

```bash
SRV_BUILD=<villagesql-build> \
MB=/opt/homebrew/opt/mariadb@11.8 \
PGBIN=/opt/homebrew/opt/postgresql@17/bin \
N_LIST=1000000 \
  bash run_three_way.sh | tee results_1M.txt
```

This runs each engine's full build+scan sweep in isolation (only one server live
at a time). At 1M expect a long run (tens of minutes per engine); it is
caffeinate-protected, so you can leave it.

---

## 5. Fair-comparison checklist (before trusting / citing numbers)

- [ ] Extension built **-O3** (§1 verify step). This is the #1 gotcha.
- [ ] **All three engines are RELEASE builds** (a debug/`-O0` build of ANY engine
      makes the comparison meaningless). Verify each — see §6.
- [ ] Fresh `.veb` **staged** into the server veb dir (check mtime), not a stale one.
- [ ] Same `BUF` (buffer pool) across engines; working set fits in it (in-memory,
      not I/O-bound). 1M/D32 ≈ 420 MB; 100k/D768 ≈ 350 MB.
- [ ] `caffeinate` in effect for any long/unattended run (run_sweep does this).
- [ ] One engine at a time for timing at scale.
- [ ] Recall < 1.0 and responsive to ef_search (index in use, not full-scan).
- [ ] Note the unavoidable asymmetries: MariaDB efc hardcoded 10; pgvector's
      native build is bulk `CREATE INDEX` (this harness drives it incrementally).

---

## 6. Verify each engine is a RELEASE build

A debug or `-O0` build of *any* engine silently invalidates the comparison (we
lost real time to a `-O0` vsql extension — the distance kernel ran unvectorized
and ~6–10x slow). Confirm all three before citing numbers. Commands below are
macOS/Homebrew; Linux equivalents noted.

**PostgreSQL** — check configure flags and optimization level:
```bash
PGCFG=$(brew --prefix postgresql@17)/bin/pg_config   # Linux: just `pg_config`
$PGCFG --configure | tr ' ' '\n' | grep -iE 'debug|assert'   # want: --disable-debug, NO --enable-cassert
$PGCFG --cflags     | grep -oE '\-O[0-9s]'                   # want: -O2 (or -O3)
```
A `--enable-debug` or `--enable-cassert` build is NOT valid for benchmarking.

**MariaDB** — the version string is the definitive tell (debug builds append
`-debug`):
```bash
$(brew --prefix mariadb@11.8)/bin/mariadbd --version   # Linux: mariadbd --version
# release: "... 11.8.9-MariaDB ..."   debug: "... 11.8.9-MariaDB-debug ..."
```

**pgvector and the vsql extension** — these are compiled shared libs; confirm the
distance kernel is vectorized (a scalar-only body signals `-O0`). On macOS use
`otool -tv`, on Linux `objdump -d`:
```bash
# pgvector (macOS: vector.dylib; Linux: vector.so under pg_config --pkglibdir)
VEC=$(brew --prefix pgvector)/lib/postgresql@17/vector.dylib
otool -tv "$VEC" | grep -cE '\.4s|\.2d|fmadd|fmla'   # want: > 0 (NEON SIMD present)
# Linux: objdump -d $(pg_config --pkglibdir)/vector.so | grep -cE 'v?fmadd|mulps|vmul'

# vsql extension: already covered by §1 (grep flags.make for -O3), or disassemble
# the staged svector lib the same way.
```
`> 0` vector instructions ⇒ optimized. `0` ⇒ almost certainly `-O0`, re-check the
build type. (Grep patterns are formatting-sensitive; if you get 0, eyeball the
distance function's disassembly before concluding it's unoptimized.)

---

## Appendix — pgvector's native bulk build (its real build number)

This harness times pgvector on the *incremental* path (index present during
insert) for an apples-to-apples with vsql/MariaDB, which build incrementally.
pgvector's intended/fast path is **bulk**: load into an unindexed table, then
`CREATE INDEX` (parallel, in-memory). To get pgvector's real build number:

```bash
SHARED_BUFFERS=4GB MAINT_WORK_MEM=4GB PGBIN=<...> bash start_postgres.sh
PSQL=<...>/psql; SOCK=.run-pg
"$PSQL" -h "$SOCK" -d bench -c "ALTER SYSTEM SET max_parallel_maintenance_workers=8;"
# restart to apply, then:
#   CREATE TABLE t (id int primary key, v vector(32));
#   INSERT ... (no index)         -- load phase
#   CREATE INDEX ON t USING hnsw (v vector_l2_ops) WITH (m=8, ef_construction=100);  -- bulk build
# time the CREATE INDEX separately — this is ~100x faster than the incremental path.
```
