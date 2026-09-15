#!/usr/bin/env python3
# Copyright (c) 2026 VillageSQL Contributors
# SPDX-License-Identifier: Apache-2.0
"""
rw_bench.py — READ-UNDER-WRITE benchmark for the VillageSQL vector index. Builds
the index, then runs a timed KNN read load (--readers threads, --rw-duration
sec) while a concurrent INSERT load of RANDOM vectors runs in the middle
(--rw-write-threads, starting at --rw-write-start for --rw-write-duration,
--rw-write-delay between statements). Emits per-second read QPS (read-alone ->
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
    merges all readers' samples into a per-second QPS time series.
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
    lat = []      # (latency_us, completion_second) -- per query, for percentiles
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
            cur.fetchall()
            now = _t.time()
            # per-query latency + the second it completed in (relative to start),
            # so the parent can split latencies into read-only vs read+write
            # phases and compute p50/p95/p99.
            lat.append(((now - _q0) * 1e6, int(now - t_start)))
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
           write_duration, id_base, seed)."""
    import time as _t
    import numpy as _np
    (socket, database, dim, batch, delay, t_start, start_offset, write_duration,
     id_base, seed) = arg
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
            vecs = rng.standard_normal((batch, dim)).astype(_np.float32)
            rows = ",\n".join(
                "({}, '[{}]')".format(nid + j,
                                      ",".join(str(int(x)) for x in vecs[j]))
                for j in range(batch))
            if w_start is None:
                w_start = _t.time()
            try:
                cur.execute(f"INSERT INTO t VALUES\n{rows}")
            except Exception as e:  # noqa: BLE001
                return (n, (w_end or _t.time()) - (w_start or _t.time()), str(e))
            nid += batch
            n += batch
            w_end = _t.time()
            if delay > 0:
                _t.sleep(delay)
        elapsed = (w_end - w_start) if (w_start and w_end) else 0.0
        return (n, elapsed, None)
    finally:
        conn.close()


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
    ap.add_argument("--rw-write-batch", type=int, default=100,
                    help="rw-bench: rows per writer INSERT statement (default 100)")
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

    prof, metric, _sweep = core.resolve_profile(args)
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
    _db = "recall_bench"
    # Pre-build the read SQL pool; readers cycle through it. Built ONCE here,
    # outside the timed loop. --rw-read-sql overrides the KNN query with an
    # arbitrary read (e.g. "SELECT MAX(id) FROM t") -- a CONTROL: if a plain
    # non-index read ALSO fails under concurrent writes, the read-under-write
    # failure is generic (or a harness issue), not the ANN/custom-index fetch
    # path; if only the KNN read fails, it's the custom-index path.
    if args.rw_read_sql:
        query_sqls = [args.rw_read_sql]
    else:
        order = metric.get("order", "")
        query_sqls = [
            f"SELECT id FROM t ORDER BY "
            f"{metric['dist_fn'].format(qlit=lit(queries[qi]))} {order} "
            f"LIMIT {args.k}"
            for qi in range(args.queries)]

    nR = args.readers
    nW = args.rw_write_threads
    t_start = time.time() + 1.0  # shared wall-clock start (align all workers)
    read_tasks = [(args.socket, _db, query_sqls, t_start, args.rw_duration,
                   args.rw_sample_every, args.k, args.rw_read_uncommitted)
                  for _ in range(nR)]
    id_block = 10_000_000  # disjoint id range per writer (no PK collision)
    write_tasks = [(args.socket, _db, args.dim, args.rw_write_batch,
                    args.rw_write_delay, t_start, args.rw_write_start,
                    args.rw_write_duration, args.n + w * id_block, 1000 + w)
                   for w in range(nW)]

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=nR + nW)  # readers + writers run concurrently
    try:
        read_async = pool.map_async(_rw_read_worker, read_tasks)
        write_async = pool.map_async(_rw_write_worker, write_tasks)
        read_results = read_async.get()
        write_results = write_async.get()
    finally:
        pool.close()
        pool.join()

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
    wrate = (f"  write_rate={write_rps:,.0f} rows/s over {w_elapsed:.1f}s"
             if wrote else "")
    print(f"total reads={total_reads}  writes={wrote} rows{wrate}"
          f"{'  WRITE ERRORS: ' + str(werrs[:2]) if werrs else ''}")
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
    for _s, lat in read_results:
        for (us, sec) in lat:
            if sec <= 0:
                continue  # skip warmup second
            (rw_lat if (ws <= sec < we) else ro_lat).append(us)
    print("read latency (us)   p50      p95      p99      max      n")
    for label, xs in (("  read-only", ro_lat),
                      ("  read+write", rw_lat if nW > 0 else None)):
        if xs is None:
            continue
        print(f"{label:<18} {_pctl(xs,50):>8.0f} {_pctl(xs,95):>8.0f} "
              f"{_pctl(xs,99):>8.0f} {(max(xs) if xs else 0):>8.0f} "
              f"{len(xs):>8}")
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
