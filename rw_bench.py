#!/usr/bin/env python3
# Copyright (c) 2026 VillageSQL Contributors
# SPDX-License-Identifier: Apache-2.0
"""
rw_bench.py — READ-UNDER-WRITE benchmark for the VillageSQL vector index. Builds
the index, then runs a timed KNN read load (--readers threads, --rw-duration
sec) while a concurrent INSERT load of RANDOM vectors runs in the middle
(--rw-write-threads, starting at --rw-write-start for --rw-write-duration,
--rw-write-delay between statements). --rw-write-mix makes the writers also
DELETE and UPDATE the vector of existing rows. Emits per-second read QPS (read-alone ->
read+write -> recover) as a table + ASCII sparkline, read-latency percentiles
(p50/p95/p99) split by phase, and the write rate. mysql only.

Shared infra (profiles, connections, datasets, the build path) lives in
harness_core.py.
"""
import argparse, sys, time
import harness_core as core
from harness_core import _new_mysql_conn


def _rw_read_worker(arg):
    """READ-under-write load: one reader process runs KNN queries in a tight loop
    for `duration` wall-seconds, cycling through a fixed pool of pre-built query
    SQL strings, and stamps (cumulative_count, wall_ts) every `sample_every`
    queries. Owns its own connection. Returns the list of samples; the parent
    merges all readers' samples into a per-second QPS time series. It also
    records each query's row count, so the parent can count SHORT results:
    fewer than k rows, although the table holds at least k.
    All workers share an absolute WALL-CLOCK start (t_start, from the parent's
    time.time()) so read and write timelines align across processes; samples are
    stamped as (count, ts - t_start) = seconds-since-start.
    arg = (socket, database, query_sqls, t_start, duration_s, sample_every, k,
           read_uncommitted)."""
    import time as _t
    (socket, database, query_sqls, t_start, duration_s, sample_every, k,
     read_uncommitted) = arg
    conn = _new_mysql_conn(socket, database=database)
    samples = []  # (cumulative_count, seconds_since_start) -- for per-sec QPS
    lat = []      # (latency_us, completion_second, rows) -- per query
    n = 0
    m = len(query_sqls)
    try:
        cur = conn.cursor()
        if read_uncommitted:
            # DIAGNOSTIC/workaround for the err-1500 read-under-write bug: at
            # READ UNCOMMITTED the reader has no consistent snapshot, so a KNN
            # hit pointing at a concurrently-inserted (otherwise-invisible) row
            # is now visible -> the clustered lookup finds it -> no
            # DB_RECORD_NOT_FOUND. Proves the failure is MVCC visibility; NOT a
            # correctness fix (dirty reads).
            cur.execute("SET SESSION TRANSACTION ISOLATION LEVEL READ UNCOMMITTED")
        while _t.time() < t_start:      # align to shared start
            _t.sleep(0.002)
        deadline = t_start + duration_s
        while True:
            _q0 = _t.time()
            cur.execute(query_sqls[n % m])
            nrows = len(cur.fetchall())
            now = _t.time()
            # per-query latency + the second it completed in (relative to start),
            # so the parent can split latencies into read-only vs read+write
            # phases and compute p50/p95/p99.
            lat.append(((now - _q0) * 1e6, int(now - t_start), nrows))
            n += 1
            if n % sample_every == 0:
                samples.append((n, now - t_start))
                if now >= deadline:
                    break
            elif (n & 63) == 0 and _t.time() >= deadline:
                samples.append((n, _t.time() - t_start))
                break
        return (samples, lat)
    finally:
        conn.close()


def _rw_write_worker(arg):
    """WRITE load (the perturbation): one writer process, after `start_offset`
    wall-seconds, runs batched INSERTs of RANDOM vectors into the same table for
    `write_duration` seconds, sleeping `delay` seconds between insert statements
    (dials write pressure). ids start at id_base+shard_offset and never collide
    with the initial load or other writers. Owns its own connection. Returns
    (rows_inserted, elapsed_write_seconds, error_or_None) -- elapsed is measured
    from the FIRST insert to the last, so the parent can compute an accurate
    rows/sec write rate (not assuming writers filled the whole window).
    Uses the same shared WALL-CLOCK start t_start as the readers, so writes begin
    at t_start + start_offset and the read-QPS graph aligns.
    arg = (socket, database, dim, batch, delay, t_start, start_offset,
           write_duration, id_base, seed, vec_literal, vec_encode, max_rows).
    max_rows (None = unbounded) caps this writer's inserted rows -- needed for
    engines whose index has a fixed capacity (an overflow aborts MyVector).
    vec_literal/vec_encode are the profile's, so concurrent rows use the same
    SQL encoding as the initial load (e.g. myvector_construct('[...]'), or the
    _binary X'..' form) rather than a bare '[...]' string."""
    import time as _t
    import numpy as _np
    (socket, database, dim, batch, delay, t_start, start_offset, write_duration,
     id_base, seed, vec_literal, vec_encode, max_rows) = arg
    lit = core.make_lit(vec_literal, vec_encode, dim)
    conn = _new_mysql_conn(socket, database=database)
    rng = _np.random.default_rng(seed)
    nid = id_base
    n = 0
    w_start = None
    w_end = None
    try:
        cur = conn.cursor()
        # Align to the shared clock: wait until t_start + start_offset.
        while _t.time() < t_start + start_offset:
            _t.sleep(0.01)
        deadline = t_start + start_offset + write_duration
        while _t.time() < deadline:
            if max_rows is not None and n >= max_rows:
                break  # budget spent: stop rather than overflow the index
            # Integer-valued (truncated) components, as before -- short SQL, and
            # for '{lit}'-style profiles the text is byte-identical to the old
            # hand-built '[...]'. int32 first so -0.x truncates to 0, not "-0".
            nb = batch if max_rows is None else min(batch, max_rows - n)
            vecs = (rng.standard_normal((nb, dim))
                    .astype(_np.int32).astype(_np.float32))
            rows = ",\n".join(
                "({}, {})".format(nid + j, lit(vecs[j])) for j in range(nb))
            if w_start is None:
                w_start = _t.time()
            try:
                cur.execute(f"INSERT INTO t VALUES\n{rows}")
            except Exception as e:  # noqa: BLE001
                return (n, (w_end or _t.time()) - (w_start or _t.time()), str(e))
            nid += nb
            n += nb
            w_end = _t.time()
            if delay > 0:
                _t.sleep(delay)
        elapsed = (w_end - w_start) if (w_start and w_end) else 0.0
        return (n, elapsed, None)
    finally:
        conn.close()


WRITE_OPS = ("insert", "delete", "update_vec")
# Lost connection / server gone: stop the writer. Any other error rolls back the
# transaction, is counted by code, and the writer goes on.
_FATAL_ERRNOS = (2006, 2013)


def parse_write_mix(spec):
    """'insert=40,delete=30,update_vec=30' -> {op: weight}. Unnamed ops get 0."""
    mix = dict.fromkeys(WRITE_OPS, 0.0)
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        op, _, w = part.partition("=")
        op = op.strip()
        if op not in mix:
            raise ValueError(f"unknown write op '{op}' (known: {', '.join(WRITE_OPS)})")
        mix[op] = float(w)
        if mix[op] < 0:
            raise ValueError(f"negative weight for '{op}'")
    if sum(mix.values()) <= 0:
        raise ValueError("write mix has no positive weight")
    return mix


def _rw_mix_worker(arg):
    """MIXED write load: one writer process runs transactions of `txn_ops`
    operations drawn from `mix` (insert / delete / update_vec) for
    `write_duration` seconds, starting at t_start + start_offset.

    The writer owns the ids with id % n_writers == w: the initial rows 0..n-1 in
    that class, plus the ids it inserts (from id_base up). It deletes and updates
    only ids it owns, so writers never conflict with each other, and it always
    knows which of its ids exist. Its local state changes only after COMMIT.

    update_vec either replaces the vector with a new random one (update_mode
    'random'), or moves it by Gaussian noise of move_sigma * the per-dimension
    std of the initial data ('move'), so the row stays near its old neighbours.
    The initial vectors are read from data_path (an .npy, memory-mapped), so
    only the touched rows are paged in.

    Inserted vectors are random, as in _rw_write_worker. max_inserts (None =
    unbounded) caps this writer's inserts; when it is spent, an insert pick
    becomes another op.

    Returns (counts_by_op, commits, elapsed_s, errors_by_code, fatal_or_None).
    arg = (socket, database, dim, n, n_writers, w, mix, txn_ops, update_mode,
           move_sigma, data_path, dim_std, delay, t_start, start_offset,
           write_duration, id_base, seed, vec_literal, vec_encode, max_inserts)."""
    import random as _random
    import time as _t
    import numpy as _np
    (socket, database, dim, n, n_writers, w, mix, txn_ops, update_mode,
     move_sigma, data_path, dim_std, delay, t_start, start_offset,
     write_duration, id_base, seed, vec_literal, vec_encode, max_inserts) = arg
    lit = core.make_lit(vec_literal, vec_encode, dim)
    rng = _np.random.default_rng(seed)
    pick = _random.Random(seed)
    init = _np.load(data_path, mmap_mode="r") if data_path else None
    std = _np.asarray(dim_std, _np.float32) if dim_std is not None else None

    # Live ids this writer owns: list + position map, for O(1) random pick and
    # swap-remove.
    live = list(range(w, n, n_writers))
    pos = {i: k for k, i in enumerate(live)}
    moved = {}  # id -> current vector, for rows this writer inserted or updated

    def live_remove(i):
        k = pos.pop(i)
        last = live.pop()
        if last != i:
            live[k] = last
            pos[last] = k

    def live_add(i):
        pos[i] = len(live)
        live.append(i)

    def rand_vec():
        # Same shape as _rw_write_worker's rows: integer-valued components.
        return rng.standard_normal(dim).astype(_np.int32).astype(_np.float32)

    def current_vec(i):
        v = moved.get(i)
        return v if v is not None else _np.asarray(init[i], _np.float32)

    def new_vec_for(i):
        if update_mode == "random":
            return rand_vec()
        return (current_vec(i) + rng.standard_normal(dim).astype(_np.float32)
                * move_sigma * std)

    ops = [op for op in WRITE_OPS if mix[op] > 0]
    weights = [mix[op] for op in ops]
    counts = dict.fromkeys(WRITE_OPS, 0)
    errors = {}
    commits = 0
    inserted = 0
    nid = id_base
    w_start = w_end = None
    conn = _new_mysql_conn(socket, database=database)
    try:
        cur = conn.cursor()
        while _t.time() < t_start + start_offset:
            _t.sleep(0.01)
        deadline = t_start + start_offset + write_duration
        while _t.time() < deadline:
            # Plan the transaction against the local state. Ids deleted or
            # updated in this transaction are not picked again in it.
            touched = set()
            plan = []  # (op, id, vec_or_None)
            for _ in range(txn_ops):
                op = pick.choices(ops, weights)[0]
                if op == "insert" and max_inserts is not None and \
                        inserted + sum(p[0] == "insert" for p in plan) >= max_inserts:
                    op = "update_vec" if mix["update_vec"] > 0 else "delete"
                if op != "insert":
                    i = None
                    for _try in range(8):
                        if not live:
                            break
                        c = live[pick.randrange(len(live))]
                        if c not in touched:
                            i = c
                            break
                    if i is None:
                        continue  # nothing left to delete or update
                    touched.add(i)
                    plan.append((op, i, new_vec_for(i) if op == "update_vec" else None))
                else:
                    plan.append(("insert", nid, rand_vec()))
                    nid += 1
            if not plan:
                break  # insert budget spent and no live rows left
            if w_start is None:
                w_start = _t.time()
            ins = [p for p in plan if p[0] == "insert"]
            try:
                cur.execute("BEGIN")
                if ins:
                    rows = ",\n".join(f"({i}, {lit(v)})" for _op, i, v in ins)
                    cur.execute(f"INSERT INTO t VALUES\n{rows}")
                for op, i, v in plan:
                    if op == "delete":
                        cur.execute(f"DELETE FROM t WHERE id = {i}")
                    elif op == "update_vec":
                        cur.execute(f"UPDATE t SET v = {lit(v)} WHERE id = {i}")
                cur.execute("COMMIT")
            except Exception as e:  # noqa: BLE001
                code = e.args[0] if e.args and isinstance(e.args[0], int) else None
                errors[code] = errors.get(code, 0) + 1
                if code in _FATAL_ERRNOS:
                    return (counts, commits, (w_end or _t.time()) - w_start,
                            errors, str(e))
                try:
                    cur.execute("ROLLBACK")
                except Exception:  # noqa: BLE001
                    pass
                continue
            # Committed: apply the plan to the local state.
            for op, i, v in plan:
                counts[op] += 1
                if op == "insert":
                    live_add(i)
                    inserted += 1
                    if update_mode == "move":
                        moved[i] = v
                elif op == "delete":
                    live_remove(i)
                    moved.pop(i, None)
                elif update_mode == "move":
                    moved[i] = v
            commits += 1
            w_end = _t.time()
            if delay > 0:
                _t.sleep(delay)
        elapsed = (w_end - w_start) if (w_start and w_end) else 0.0
        return (counts, commits, elapsed, errors, None)
    finally:
        conn.close()


# Default write budget for fixed-capacity indexes (MyVector's size=): roughly
# what the default 2 unthrottled writers x 10 s insert at small dims, reserved
# as index headroom so the writers can never overflow the graph.
FIXED_CAP_DEFAULT_ROWS = 100_000


def main():
    ap = argparse.ArgumentParser()
    # --- shared build args ---
    ap.add_argument("--profile", choices=core.PROFILES, default="vsql_vector")
    ap.add_argument("--metric", choices=list(core.GROUND_TRUTH), default="l2",
                    help="distance metric; must be supported by the profile")
    ap.add_argument("--dataset", choices=list(core.ANN_DATASETS), default=None,
                    help="use a real ann-benchmarks dataset (HDF5) instead of "
                         "synthetic random vectors. Sets dim + metric from the "
                         "data; --n/--queries cap how much of it to use.")
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--n", type=int, default=None,
                    help="rows in the table (synthetic default 2000; with "
                         "--dataset, unset/0 = use the full base set)")
    ap.add_argument("--queries", type=int, default=None,
                    help="query count -- sizes the KNN query pool the readers "
                         "cycle through (synthetic default 200; with --dataset, "
                         "unset/0 = all test queries)")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--epsilon", type=float, default=1e-3, help=argparse.SUPPRESS)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=64)
    ap.add_argument("--insert-batch", type=int, default=1000,
                    help="rows per INSERT statement for the initial build "
                         "(default 1000).")
    ap.add_argument("--no-index", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--index-mode", choices=["incremental", "post"],
                    default="incremental", help=argparse.SUPPRESS)
    ap.add_argument("--build-threads", type=int, default=1,
                    help="concurrent CLIENT connections for the INITIAL build "
                         "(default 1 = serial).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ef-search", type=int, default=None,
                    help="query-time HNSW search breadth (default 100 in "
                         "rw-bench); set once as a server GLOBAL, readers inherit.")
    ap.add_argument("--readers", type=int, default=1,
                    help="number of concurrent reader processes for the read "
                         "load (default 1).")
    ap.add_argument("--socket", default=core.DEFAULT_SOCKET)
    ap.add_argument("--mysql", default=core.DEFAULT_MYSQL)
    ap.add_argument("--host", default=None,
                    help="connect over TCP to this host instead of --socket "
                         "(e.g. 127.0.0.1 for a container with -p PORT:3306)")
    ap.add_argument("--port", type=int, default=3306,
                    help="TCP port when --host is set (default 3306)")
    # --- read-under-write knobs ---
    ap.add_argument("--rw-duration", type=float, default=30.0,
                    help="rw-bench: total read-load seconds (default 30)")
    ap.add_argument("--rw-sample-every", type=int, default=50,
                    help="rw-bench: each reader stamps (count, ts) every N "
                         "queries (default 50)")
    ap.add_argument("--rw-write-threads", type=int, default=2,
                    help="rw-bench: concurrent INSERT writer processes (default 2)")
    ap.add_argument("--rw-write-start", type=float, default=10.0,
                    help="rw-bench: seconds into the read load before writes "
                         "start (default 10)")
    ap.add_argument("--rw-write-duration", type=float, default=10.0,
                    help="rw-bench: how long the write load runs (default 10)")
    ap.add_argument("--rw-write-delay", type=float, default=0.0,
                    help="rw-bench: seconds to sleep between INSERT statements "
                         "per writer -- dials write pressure (0 = as fast as "
                         "possible; default 0)")
    ap.add_argument("--rw-max-rows", type=int, default=None,
                    help="rw-bench: cap on the TOTAL rows all writers insert "
                         "(split evenly). Unbounded by default, except for "
                         "fixed-capacity indexes (myvector), which default to "
                         f"{FIXED_CAP_DEFAULT_ROWS:,} and reserve that much index "
                         "headroom up front.")
    ap.add_argument("--rw-write-batch", type=int, default=100,
                    help="rw-bench: rows per writer INSERT statement (default 100). "
                         "Insert-only mix only; a mixed load uses --rw-txn-ops.")
    ap.add_argument("--rw-write-mix", default="insert=100",
                    help="rw-bench: weights of the writers' operations, e.g. "
                         "'insert=40,delete=30,update_vec=30' (default "
                         "'insert=100', the insert-only load). With delete or "
                         "update_vec, writer w owns the ids with id %% writers == w "
                         "and deletes/updates only those, so writers never "
                         "conflict.")
    ap.add_argument("--rw-txn-ops", type=int, default=1,
                    help="rw-bench: operations per write transaction in a mixed "
                         "load (default 1). The inserts of a transaction go in "
                         "one multi-row INSERT.")
    ap.add_argument("--rw-update-mode", choices=["move", "random"],
                    default="move",
                    help="rw-bench: update_vec moves the vector by small noise, "
                         "keeping it near its neighbours (move, default), or "
                         "replaces it with a random one (random).")
    ap.add_argument("--rw-move-sigma", type=float, default=0.1,
                    help="rw-bench: noise of an update_vec move, as a fraction of "
                         "the per-dimension std of the initial data (default 0.1)")
    ap.add_argument("--rw-csv", default=None,
                    help="rw-bench: write the per-second QPS series to this CSV")
    ap.add_argument("--rw-read-sql", default=None,
                    help="rw-bench: override the reader's query with this exact "
                         "SQL (e.g. 'SELECT MAX(id) FROM t'). CONTROL for the "
                         "read-under-write failure: a plain non-index read that "
                         "ALSO fails => generic/harness; only the KNN read "
                         "failing => the custom-index fetch path.")
    ap.add_argument("--rw-read-uncommitted", action="store_true",
                    help="rw-bench: set each reader session to READ UNCOMMITTED. "
                         "Diagnostic/workaround for the err-1500 read-under-write "
                         "bug (a KNN hit on a concurrently-inserted, MVCC-"
                         "invisible row -> DB_RECORD_NOT_FOUND -> error 122). At "
                         "READ UNCOMMITTED the row is visible so the fetch "
                         "succeeds. If this makes the failure vanish, the cause "
                         "is confirmed MVCC visibility. NOT a correctness fix "
                         "(dirty reads).")
    args = ap.parse_args()

    # --host switches to TCP: fold host/port into the socket spec so every
    # run_sql call site stays unchanged (run_sql interprets the 'tcp:' prefix).
    if args.host:
        args.socket = f"tcp:{args.host}:{args.port}"

    try:
        mix = parse_write_mix(args.rw_write_mix)
    except ValueError as e:
        print(f"ERROR: --rw-write-mix: {e}", file=sys.stderr)
        return 2
    mixed = mix["delete"] > 0 or mix["update_vec"] > 0
    if args.rw_txn_ops < 1:
        print("ERROR: --rw-txn-ops must be >= 1", file=sys.stderr)
        return 2

    prof, metric, _sweep = core.resolve_profile(args)
    if mixed and not prof.get("supports_dml", True):
        print(f"ERROR: profile '{args.profile}' does not support DELETE/UPDATE "
              f"of indexed rows; use an insert-only --rw-write-mix.",
              file=sys.stderr)
        return 2
    # Fixed-capacity indexes need room for the concurrent writes reserved at
    # build time, and the writers capped to it (an overflow aborts MyVector).
    if prof.get("fixed_capacity") and args.rw_max_rows is None:
        args.rw_max_rows = FIXED_CAP_DEFAULT_ROWS
    if prof.get("fixed_capacity"):
        args.index_headroom = args.rw_max_rows
    r = core.build_index(args, prof, metric)
    metric = r.metric      # possibly dataset-overridden
    lit = r.lit
    queries = r.queries

    # READ-under-WRITE load benchmark. NO ground truth / recall -- pure load.
    import math
    import multiprocessing as mp
    if core.CLIENT == "psql":
        print("ERROR: rw_bench is mysql-only.", file=sys.stderr)
        return 2
    # ef_search once, server-side GLOBAL (readers inherit it). Only meaningful
    # for "set"-style ef params; inline-param engines carry ef in the query.
    ef = args.ef_search if args.ef_search is not None else 100
    ef_param = prof.get("ef_param", {})
    if ef_param.get("style") == "set":
        core.run_sql(args.mysql, args.socket,
                     f"SET GLOBAL {ef_param['var']} = {ef};")
    _db = core.MYSQL_SCHEMA
    # Pre-build the read SQL pool; readers cycle through it. Built ONCE here,
    # outside the timed loop. --rw-read-sql overrides the KNN query with an
    # arbitrary read (e.g. "SELECT MAX(id) FROM t") -- a CONTROL: if a plain
    # non-index read ALSO fails under concurrent writes, the read-under-write
    # failure is generic (or a harness issue), not the ANN/custom-index fetch
    # path; if only the KNN read fails, it's the custom-index path.
    if args.rw_read_sql:
        query_sqls = [args.rw_read_sql]
    elif prof.get("query_sql") and not args.no_index:
        # Engines with a full query template (e.g. MyVector's WHERE-gated,
        # row-distance-ordered ANN read) that isn't a plain ORDER BY <dist>.
        # Such engines take ef_search inline in the query (MyVector only accepts
        # it there), so it's formatted into each query here.
        qtmpl = prof["query_sql"]
        query_sqls = [
            qtmpl.format(qlit=lit(queries[qi]), k=args.k, ef=ef, db=core.MYSQL_SCHEMA)
            for qi in range(args.queries)]
    else:
        order = metric.get("order", "")
        query_sqls = [
            f"SELECT id FROM t ORDER BY "
            f"{metric['dist_fn'].format(qlit=lit(queries[qi]))} {order} "
            f"LIMIT {args.k}"
            for qi in range(args.queries)]

    # Mixed-load setup runs before t_start is fixed, so its time (a full
    # COUNT(*), saving the vectors) doesn't eat into the workers' start slack.
    data_dir = None
    if mixed:
        def _count_rows():
            out = core.run_sql(args.mysql, args.socket, "SELECT COUNT(*) FROM t;",
                               want_rows=True)
            return int(out[0]) if out else -1
        rows_before = _count_rows()
        data_path, dim_std = None, None
        if mix["update_vec"] > 0 and args.rw_update_mode == "move":
            # Writers memory-map the initial vectors instead of each receiving
            # a pickled copy.
            import tempfile
            import numpy as np
            data_dir = tempfile.mkdtemp(prefix="rw_bench_")
            data_path = f"{data_dir}/data.npy"
            np.save(data_path, np.asarray(r.data, np.float32))
            dim_std = np.asarray(r.data, np.float32).std(axis=0).tolist()

    nR = args.readers
    nW = args.rw_write_threads
    t_start = time.time() + 1.0  # shared wall-clock start (align all workers)
    read_tasks = [(args.socket, _db, query_sqls, t_start, args.rw_duration,
                   args.rw_sample_every, args.k, args.rw_read_uncommitted)
                  for _ in range(nR)]
    id_block = 10_000_000  # disjoint id range per writer (no PK collision)
    # Split the write budget so the per-writer caps sum to exactly rw_max_rows.
    def _budget(w):
        if args.rw_max_rows is None:
            return None
        return args.rw_max_rows // nW + (1 if w < args.rw_max_rows % nW else 0)
    if mixed:
        write_tasks = [(args.socket, _db, args.dim, args.n, nW, w, mix,
                        args.rw_txn_ops, args.rw_update_mode,
                        args.rw_move_sigma, data_path, dim_std,
                        args.rw_write_delay, t_start, args.rw_write_start,
                        args.rw_write_duration, args.n + w * id_block, 1000 + w,
                        prof["vec_literal"], prof.get("vec_encode"), _budget(w))
                       for w in range(nW)]
        write_fn = _rw_mix_worker
    else:
        write_tasks = [(args.socket, _db, args.dim, args.rw_write_batch,
                        args.rw_write_delay, t_start, args.rw_write_start,
                        args.rw_write_duration, args.n + w * id_block, 1000 + w,
                        prof["vec_literal"], prof.get("vec_encode"), _budget(w))
                       for w in range(nW)]
        write_fn = _rw_write_worker

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=nR + nW)  # readers + writers run concurrently
    try:
        read_async = pool.map_async(_rw_read_worker, read_tasks)
        write_async = pool.map_async(write_fn, write_tasks)
        read_results = read_async.get()
        write_results = write_async.get()
    finally:
        pool.close()
        pool.join()
        if data_dir:
            import shutil
            shutil.rmtree(data_dir, ignore_errors=True)

    if mixed:
        # Mixed entries are (counts, commits, elapsed, errors, fatal); fold them
        # into the insert-only shape (ops, elapsed, error) for the shared report.
        op_counts = dict.fromkeys(WRITE_OPS, 0)
        err_counts = {}
        commits = 0
        for counts, c, _el, errs, _fatal in write_results:
            commits += c
            for op, k in counts.items():
                op_counts[op] += k
            for code, k in errs.items():
                err_counts[code] = err_counts.get(code, 0) + k
        write_results = [(sum(cn.values()), el, fatal)
                         for cn, _c, el, _e, fatal in write_results]
        rows_after = _count_rows()

    # read_results entries are (samples, lat): samples = [(cum_count, sec)]
    # for per-sec QPS; lat = [(latency_us, completion_sec)] per query.
    # Per-second QPS: diff each reader's cumulative samples, bin by end-second.
    per_sec = {}
    for samples, _lat in read_results:
        prev_n = 0
        for (cnt, ts) in samples:
            per_sec[int(math.floor(ts))] = (
                per_sec.get(int(math.floor(ts)), 0) + (cnt - prev_n))
            prev_n = cnt
    total_reads = sum((s[-1][0] if s else 0) for s, _l in read_results)
    # write_results entries are (rows, elapsed_write_s, error_or_None).
    wrote = sum(r[0] for r in write_results if r)
    werrs = [r[2] for r in write_results if r and r[2]]
    # Aggregate write rate: total rows / the max per-writer write elapsed
    # (writers overlap in wall-clock, so the window length ~= the longest
    # writer's active span, not the sum).
    w_elapsed = max((r[1] for r in write_results if r), default=0.0)
    write_rps = (wrote / w_elapsed) if w_elapsed > 0 else 0.0

    # Bin [0, duration) exclusive: the final partial second (queries stamped
    # right at/after the deadline) lands in bucket floor(duration), which is
    # an incomplete interval and would show a spuriously low QPS -- drop it.
    secs = list(range(0, int(math.floor(args.rw_duration))))
    qps_series = [per_sec.get(s, 0) for s in secs]
    ws, we = args.rw_write_start, args.rw_write_start + args.rw_write_duration

    print(f"\n=== rw-bench: {nR} readers x {args.rw_duration:.0f}s, "
          f"{nW} writers @[{ws:.0f}-{we:.0f}]s delay={args.rw_write_delay} "
          f"batch={args.rw_write_batch} ===")
    unit = "ops" if mixed else "rows"
    wrate = (f"  write_rate={write_rps:,.0f} {unit}/s over {w_elapsed:.1f}s"
             if wrote else "")
    print(f"total reads={total_reads}  writes={wrote} {unit}{wrate}"
          f"{'  WRITE ERRORS: ' + str(werrs[:2]) if werrs else ''}")
    if mixed:
        per_op = "  ".join(
            f"{op}={op_counts[op]:,} ({op_counts[op] / w_elapsed:,.0f}/s)"
            if w_elapsed > 0 else f"{op}={op_counts[op]:,}"
            for op in WRITE_OPS if mix[op] > 0)
        print(f"mix {args.rw_write_mix}  txn_ops={args.rw_txn_ops}  "
              f"commits={commits:,}  update_mode={args.rw_update_mode}")
        print(f"  {per_op}")
        if err_counts:
            print("  failed transactions by error code: " + ", ".join(
                f"{code}={k}" for code, k in sorted(
                    err_counts.items(), key=lambda kv: -kv[1])))
        print(f"  table rows before={rows_before:,} after={rows_after:,}")
    if args.rw_max_rows is not None and wrote >= args.rw_max_rows:
        print(f"NOTE: writers reached --rw-max-rows={args.rw_max_rows:,} before "
              f"the write window ended (stopped after {w_elapsed:.1f}s of "
              f"{args.rw_write_duration:.0f}s); the read+write QPS below includes "
              f"seconds with no writes. Raise --rw-max-rows or --rw-write-delay "
              f"to cover the whole window.")
    peak = max(qps_series) or 1
    blocks = "▁▂▃▄▅▆▇█"
    spark = "".join(blocks[min(7, int(q / peak * 7))] for q in qps_series)
    print(f"peak {peak} qps  |{spark}|")
    # read-only vs read+write mean QPS -- only meaningful when writers ran.
    # (skip second 0: partial warmup bucket.)
    if nW > 0 and nR > 0:
        ro = [q for s, q in zip(secs, qps_series) if not (ws <= s < we) and 0 < s]
        rw = [q for s, q in zip(secs, qps_series) if ws <= s < we]
        ro_m = (sum(ro) / len(ro)) if ro else 0
        rw_m = (sum(rw) / len(rw)) if rw else 0
        if ro_m > 0:
            print(f"read-only mean={ro_m:.0f} qps  read+write mean={rw_m:.0f} "
                  f"qps  impact={100*(1-rw_m/ro_m):.0f}% drop")

    # Read-latency percentiles (YCSB-style), split by phase. Each query
    # contributed (latency_us, completion_sec); bucket by whether that second
    # was in the write window. Tail latency (p95/p99) degrades first under
    # write contention, so this is the headline concurrency metric.
    def _pctl(xs, p):
        if not xs:
            return 0.0
        xs = sorted(xs)
        i = min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1))))
        return xs[i]

    ro_lat, rw_lat = [], []
    # Short results: a KNN query with LIMIT k that returned fewer than k rows.
    # Only meaningful for the KNN query (not --rw-read-sql) and while the table
    # holds at least k rows; a mixed load that deletes most rows can make a
    # short result legitimate, so the table's row count is printed with it.
    short = {"read-only": [], "read+write": []}
    nq = {"read-only": 0, "read+write": 0}
    for _s, lat in read_results:
        for (us, sec, nrows) in lat:
            if sec <= 0:
                continue  # skip warmup second
            phase = "read+write" if (ws <= sec < we) else "read-only"
            (rw_lat if phase == "read+write" else ro_lat).append(us)
            nq[phase] += 1
            if nrows < args.k:
                short[phase].append(nrows)
    print("read latency (us)   p50      p95      p99      max      n")
    for label, xs in (("  read-only", ro_lat),
                      ("  read+write", rw_lat if nW > 0 else None)):
        if xs is None:
            continue
        print(f"{label:<18} {_pctl(xs,50):>8.0f} {_pctl(xs,95):>8.0f} "
              f"{_pctl(xs,99):>8.0f} {(max(xs) if xs else 0):>8.0f} "
              f"{len(xs):>8}")
    if not args.rw_read_sql:
        print(f"short results (< k={args.k} rows)")
        for phase in ("read-only", "read+write"):
            if phase == "read+write" and nW == 0:
                continue
            xs = short[phase]
            hist = ""
            if xs:
                counts = {}
                for x in xs:
                    counts[x] = counts.get(x, 0) + 1
                hist = "  rows:count " + " ".join(
                    f"{x}:{c}" for x, c in sorted(counts.items()))
            pct = 100.0 * len(xs) / nq[phase] if nq[phase] else 0.0
            print(f"  {phase:<16} {len(xs):>8} of {nq[phase]:<8} ({pct:.2f}%)"
                  f"{hist}")
        if mixed and min(rows_before, rows_after) < args.k:
            print(f"  NOTE: the table held fewer than k rows during the run, so "
                  f"some short results may be legitimate.")
    print(f"{'sec':>4} {'qps':>7}  {'phase':<12}")
    for s, q in zip(secs, qps_series):
        phase = "read+WRITE" if ws <= s < we else "read-only"
        print(f"{s:>4} {q:>7}  {phase:<12} {'#' * int(q / peak * 40)}")
    if args.rw_csv:
        with open(args.rw_csv, "w") as f:
            f.write("second,qps,phase\n")
            for s, q in zip(secs, qps_series):
                f.write(f"{s},{q},{'read+write' if ws <= s < we else 'read-only'}\n")
        print(f"wrote CSV -> {args.rw_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
