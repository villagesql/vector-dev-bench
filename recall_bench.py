#!/usr/bin/env python3
# Copyright (c) 2026 VillageSQL Contributors
# SPDX-License-Identifier: Apache-2.0
"""
recall_bench.py — recall + QPS tester for VillageSQL custom KNN vector indexes.
Generates or loads vectors, builds the custom index, runs KNN queries via the
index, and verifies APPROXIMATELY: compares the index's neighbours against exact
ground truth (computed client-side with numpy) and reports recall@k + QPS +
build time. Exits 0 if mean recall >= threshold, else 1 (usable as a gate).

Also carries the emit-queries / dry-run / build-only / keep-server helpers and
the concurrent-reader (--readers) QPS path. Shared infra (profiles, connections,
datasets, the build path) lives in harness_core.py.
"""
import argparse, sys, time
import numpy as np
import harness_core as core
from harness_core import _reader_init, _reader_warmup, _reader_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=core.PROFILES, default="vsql_vector")
    ap.add_argument("--metric", choices=list(core.GROUND_TRUTH), default="l2",
                    help="distance metric; must be supported by the profile")
    ap.add_argument("--dataset", choices=list(core.ANN_DATASETS), default=None,
                    help="use a real ann-benchmarks dataset (HDF5, precomputed "
                         "ground truth) instead of synthetic random vectors. "
                         "Sets dim + metric from the data; --n/--queries then cap "
                         "how much of it to use (0/unset = all).")
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--n", type=int, default=None,
                    help="rows in the table (synthetic default 2000; with "
                         "--dataset, unset/0 = use the full base set)")
    ap.add_argument("--queries", type=int, default=None,
                    help="query count (synthetic default 200; with --dataset, "
                         "unset/0 = use all test queries)")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--epsilon", type=float, default=1e-3,
                    help="tie tolerance for recall (ann-benchmarks default 1e-3): "
                         "a returned id counts if its TRUE distance is within "
                         "(1+epsilon)x the k-th true distance. Makes recall robust "
                         "to distance ties at the k-th boundary.")
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=64)
    ap.add_argument("--insert-batch", type=int, default=1000,
                    help="rows per INSERT statement (default 1000). 0 = single "
                         "statement of all n — only for the parse-cost probe; it "
                         "exceeds the MySQL/MariaDB max_allowed_packet on real "
                         "datasets (ERROR 2006), so it is NOT the default.")
    ap.add_argument("--no-index", action="store_true",
                    help="skip the custom index (SVECTOR column only) — build-cost "
                         "probe to isolate generic insert from graph maintenance; "
                         "recall will be meaningless")
    ap.add_argument("--index-mode", choices=["incremental", "post"],
                    default="incremental",
                    help="incremental: CREATE INDEX before inserts (graph "
                         "maintained per-insert; the only mode vsql/MariaDB "
                         "support). post: insert into an UNINDEXED table then "
                         "CREATE INDEX at the end (bulk build; pgvector's native "
                         "fast path). 'post' reports insert vs index time "
                         "separately. Requires a profile with a separate "
                         "index_ddl (not the inline-VECTOR-INDEX engines).")
    ap.add_argument("--ef-search", type=int, default=None,
                    help="query-time HNSW search breadth (profile must expose it)")
    ap.add_argument("--ef-search-sweep", default=None,
                    help="comma-separated ef_search values; builds the index ONCE "
                         "then re-queries at each (ef_search is query-time, so no "
                         "rebuild needed). Overrides --ef-search.")
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--no-gate", action="store_true",
                    help="report recall but always exit 0 (do NOT fail when "
                         "recall < threshold). Use for cross-engine comparison "
                         "sweeps, where a lower recall is a result to report, not "
                         "a harness failure. Without this, recall < threshold "
                         "exits 1 so the run is usable as a single-engine gate.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--emit-queries", default=None, metavar="FILE",
                    help="write the real per-query KNN SELECT statements (one "
                         "per line, real vector literals baked in) to FILE and "
                         "exit -- for feeding a native load driver (mysqlslap "
                         "-q FILE / sysbench) the EXACT query the harness "
                         "validates. Builds the index first (so the table+index "
                         "exist for the driver to hit), does NOT run the query "
                         "phase. Use with --keep-server to leave the server up.")
    ap.add_argument("--keep-server", action="store_true",
                    help="after build (and --emit-queries), leave the server "
                         "running and the table+index in place so an external "
                         "load driver can hit the warmed instance. Prints the "
                         "socket/table and exits without dropping anything.")
    ap.add_argument("--dry-run", action="store_true",
                    help="replace the KNN search query with a trivial "
                         "'SELECT id FROM t LIMIT k' (no ORDER BY -> no HNSW "
                         "search). Same statement count, round-trips, and "
                         "process/connection dispatch as a real run, so the "
                         "reported QPS is the harness+round-trip FLOOR -- "
                         "subtract it (or compare) to isolate the actual search "
                         "cost. Combine with --readers N to measure dispatch "
                         "scaling. Recall is meaningless in dry-run (ignored).")
    ap.add_argument("--build-threads", type=int, default=1,
                    help="number of concurrent CLIENT connections for the insert/"
                         "build phase (default 1 = serial). N>1 shards the rows "
                         "across N OS processes, each inserting its shard into the "
                         "SAME table+index concurrently -- drives parallel index "
                         "build FROM THE CLIENT (bypasses the server's internal "
                         "innodb_ddl_threads/parallel_read_threads). Use to test "
                         "whether the graph absorbs concurrent inserts. "
                         "incremental mode + mysql only. After the run the harness "
                         "checks row count and recall -- a broken concurrent build "
                         "shows as a recall collapse.")
    ap.add_argument("--build-only", action="store_true",
                    help="build the table+index and STOP -- report build time and "
                         "the actual row count, run NO queries. Unlike --queries 0 "
                         "(which the --dataset path silently rewrites to 'all test "
                         "queries'), this genuinely skips the query phase, so it "
                         "isolates the insert/build (e.g. to test --build-threads "
                         "concurrent inserts without a slow KNN read phase "
                         "afterwards). Recall is not computed.")
    ap.add_argument("--readers", type=int, default=1,
                    help="number of concurrent reader threads for the query "
                         "phase (default 1 = the original serial path). N>1 "
                         "shards the queries across N threads, each on its own "
                         "connection, and reports AGGREGATE QPS (total queries / "
                         "wall-clock) -- use to measure read scaling / the "
                         "decoded-vector cache's concurrency behaviour. Recall is "
                         "unchanged (per-query correctness is thread-independent). "
                         "mysql only; ignored for psql.")
    ap.add_argument("--socket", default=core.DEFAULT_SOCKET)
    ap.add_argument("--mysql", default=core.DEFAULT_MYSQL)
    # TCP alternative to --socket, for a server in a container with its port
    # published. When --host is given, the harness connects over TCP instead of
    # the unix socket (everything else -- profiles, SQL, queries -- is identical).
    ap.add_argument("--host", default=None,
                    help="connect over TCP to this host instead of --socket "
                         "(e.g. 127.0.0.1 for a container with -p PORT:3306)")
    ap.add_argument("--port", type=int, default=3306,
                    help="TCP port when --host is set (default 3306)")
    args = ap.parse_args()

    # --host switches to TCP: fold host/port into the socket spec so every
    # run_sql call site stays unchanged (run_sql interprets the 'tcp:' prefix).
    if args.host:
        args.socket = f"tcp:{args.host}:{args.port}"

    prof, metric, sweep = core.resolve_profile(args)

    r = core.build_index(args, prof, metric)
    metric = r.metric      # possibly dataset-overridden
    data = r.data
    queries = r.queries
    dataset_truth = r.dataset_truth
    lit = r.lit
    build_s = r.build_s
    split = r.split

    # truth_fn follows a dataset metric override (args.metric may have changed
    # inside build_index for a --dataset run) -- same metric drives numpy truth.
    truth_fn = core.GROUND_TRUTH[args.metric]

    if args.build_only:
        # Insert/build done. Report build time + actual row count and STOP -- no
        # query phase, no ground truth. (Unlike --queries 0, which the --dataset
        # path rewrites to 'all queries', this genuinely runs zero queries, so it
        # isolates the concurrent insert.) A short concurrent build = lost rows.
        cnt = core.run_sql(args.mysql, args.socket, "SELECT COUNT(*) FROM t;",
                           want_rows=True)
        got = int(cnt[0]) if cnt else -1
        ok = "OK" if got == args.n else f"MISMATCH (expected {args.n})"
        print(f"build_time_s={build_s:.2f}{split}  rows={got}/{args.n} {ok}  "
              f"(build-only; no queries)")
        return 0 if got == args.n else 1

    # Ground truth is ef_search-independent — compute once. TIE-TOLERANT
    # (ann-benchmarks-style): the "acceptable" set per query = ids whose TRUE
    # distance is within (1+epsilon) x the k-th true distance. A returned id
    # counts as a hit if it is in that set, so an index is not penalised for
    # returning an equally-distant-but-different-id neighbour at the k-th
    # boundary. Recall is still |returned[:k] intersect acceptable| / k. On
    # tie-free continuous data this equals the plain top-k intersection.
    if dataset_truth is not None:
        # Built from the dataset's precomputed top-100 neighbours+distances
        # (O(100)/query) up in the --dataset branch.
        truth = dataset_truth
    else:
        # Synthetic path: compute true distances query->all base in numpy.
        truth = []
        for qi in range(args.queries):
            d = truth_fn(data, queries[qi])
            kth = np.partition(d, args.k - 1)[args.k - 1]   # k-th smallest, O(n)
            thresh = kth * (1.0 + args.epsilon)
            truth.append(set(np.nonzero(d <= thresh)[0].tolist()))

    def set_ef(ef):
        # ef_search is SESSION-scoped for every engine (vsql_vector.ef_search,
        # MariaDB mhnsw_ef_search, pgvector hnsw.ef_search), so it must be set on
        # the SAME connection that runs the queries. Each core.run_sql() spawns a
        # fresh client/connection, so a SET here would not carry over to the query
        # batch -- the SET is folded into run_queries()/run_queries_parallel()
        # instead, riding inside the query connection. Nothing to do here (kept
        # for call-site symmetry).
        return

    # How this engine takes the ef-search parameter, declared per profile:
    #   {"style": "set", "var": NAME} -- a separate `SET` statement before the
    #       queries (a session/GUC variable the query then reads implicitly).
    #   {"style": "inline"}           -- passed inside the query itself; the
    #       metric's dist_fn carries an {ef} placeholder.
    ef_param = prof.get("ef_param", {})
    ef_inline = ef_param.get("style") == "inline"

    def ef_set_stmt(ef):
        # The SET statement for a "set"-style ef param, or "" (empty) for an
        # inline param / no ef. psql omits the SESSION keyword (session GUC);
        # the mysql family uses SET SESSION.
        if ef is None or ef_param.get("style") != "set":
            return ""
        kw = "" if core.CLIENT == "psql" else "SESSION "
        return f"SET {kw}{ef_param['var']} = {ef};"

    def _one_query_stmt(qi, ef=None):
        # The per-query SELECT. --dry-run replaces the real KNN search with a
        # server-TRIVIAL statement (a single constant row, NO table access at
        # all) so the reported QPS reflects the pure harness+transport floor
        # (process/connection dispatch, round-trip, sentinel/result parsing).
        # NOTE: an earlier version used "SELECT id FROM t LIMIT k" -- that was a
        # mistake: on a 60k x 784 table even a LIMIT-10 scan costs ~160us of
        # SERVER work (clustered-index / column-store touch), so it measured
        # cheap-query cost, not dispatch. A bare "SELECT 1" is ~20us and is the
        # true floor. Result shape differs (1 row, not k) but recall is
        # meaningless in dry-run anyway (ignored by callers), and the parser just
        # sees a non-@@Q digit row per query, which is fine.
        if args.dry_run:
            return "SELECT 1;"
        fmt = {"qlit": lit(queries[qi])}
        if ef_inline:
            fmt["ef"] = ef if ef is not None else args.ef_search
        dist = metric["dist_fn"].format(**fmt)
        order = metric.get("order", "")
        return f"SELECT id FROM t ORDER BY {dist} {order} LIMIT {args.k};"

    def run_queries(ef=None):
        # All queries in ONE connection (one process spawn) so QPS isn't
        # dominated by client startup. Sentinel SELECT delimits each query. A
        # "set"-style ef param must ride INSIDE this batch to apply to the query
        # connection (each run_sql is a fresh connection); an inline param is
        # already in each query, so ef_set_stmt() returns "".
        stmt = ef_set_stmt(ef)
        parts = [stmt] if stmt else []
        for qi in range(args.queries):
            parts.append(f"SELECT '@@Q{qi}' AS m;")
            parts.append(_one_query_stmt(qi, ef))
        t = time.time()
        out = core.run_sql(args.mysql, args.socket, "\n".join(parts), want_rows=True)
        qsec = time.time() - t
        per, cur = {}, None
        for line in out:
            s = line.strip()
            if s.startswith("@@Q"):
                cur = int(s[3:]); per[cur] = []
            elif cur is not None and s.isdigit():
                per[cur].append(int(s))
        hits = sum(len(set(per.get(qi, [])[:args.k]) & truth[qi])
                   for qi in range(args.queries))
        return hits / (args.queries * args.k), args.queries / qsec if qsec > 0 else float("inf")

    def _build_query_sql(qids, ef=None):
        # Build one `;`-batched statement string for the given query indices.
        # Each query is preceded by a '@@Q<i>' sentinel SELECT so the raw output
        # can be split back into per-query id lists during scoring. Mirrors
        # run_queries()'s statement shape exactly (so recall is identical).
        parts = []
        for qi in qids:
            parts.append(f"SELECT '@@Q{qi}' AS m;")
            parts.append(_one_query_stmt(qi, ef))
        return "\n".join(parts)

    def run_queries_parallel(ef, nreaders):
        # Concurrent-reader QPS: shard the queries across `nreaders` OS processes
        # (multiprocessing -> real parallelism, no GIL), each on its own
        # connection. Only the parallel query execution is timed; connection
        # setup happens in the pool initializer (warm-up) and ALL parsing/recall
        # scoring happens here in the parent AFTER the clock stops. Aggregate QPS
        # = total_queries / wall_clock, the number that reveals read scaling.
        import multiprocessing as mp

        qids = list(range(args.queries))
        # Contiguous shards keep each worker's batch a similar size.
        shards = [qids[i::nreaders] for i in range(nreaders)]
        shards = [s for s in shards if s]           # drop empties if readers>queries
        # "set"-style engines set ef per reader connection (prefix the SET);
        # inline-style engines carry ef inside each query (via _build_query_sql).
        stmt = ef_set_stmt(ef)
        ef_prefix = (stmt + "\n") if stmt else ""
        sql_batches = [ef_prefix + _build_query_sql(s, ef) for s in shards]

        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=len(sql_batches),
                        initializer=_reader_init, initargs=(args.socket,))
        try:
            # Warm up: force each worker to open its connection before timing.
            pool.map(_reader_warmup, range(len(sql_batches)))
            t = time.time()
            results = pool.map(_reader_run, sql_batches)   # <-- timed region only
            qsec = time.time() - t
        finally:
            pool.close()
            pool.join()

        # Post-processing (untimed): merge raw lines, parse @@Q -> per-query ids.
        per = {}
        for lines in results:
            cur = None
            for line in lines:
                s = line.strip()
                if s.startswith("@@Q"):
                    cur = int(s[3:]); per[cur] = []
                elif cur is not None and s.isdigit():
                    per[cur].append(int(s))
        hits = sum(len(set(per.get(qi, [])[:args.k]) & truth[qi])
                   for qi in range(args.queries))
        qps = args.queries / qsec if qsec > 0 else float("inf")
        return hits / (args.queries * args.k), qps

    if args.emit_queries:
        # Write the REAL per-query KNN SELECTs (one per line, no @@Q sentinel --
        # a load driver just runs the query) for mysqlslap -q FILE / sysbench.
        # Same _one_query_stmt() the recall path validates, so the driver runs
        # the exact query we checked. For a "set"-style ef param the file opens
        # with the SET (which mysqlslap/sysbench runs on each of its `-c N` client
        # connections); for an inline param each emitted query already carries ef.
        # The index is already built above, so the table+index exist for the
        # driver.
        emit_ef = (args.ef_search if args.ef_search is not None
                   else (sweep[0] if sweep else None))
        with open(args.emit_queries, "w") as f:
            stmt = ef_set_stmt(emit_ef)
            if stmt:
                f.write(stmt + "\n")
            for qi in range(args.queries):
                # Keep the trailing ';' -- mysqlslap -q FILE splits the file into
                # statements on --delimiter=';', so each query MUST end with it
                # (without it, the whole file is read as one malformed statement).
                f.write(_one_query_stmt(qi, emit_ef) + "\n")
        print(f"emitted {args.queries} queries -> {args.emit_queries}")
        print(f"  table 't' + HNSW index built (build_time_s={build_s:.2f}); "
              f"ef_search={emit_ef} set per-connection (SET SESSION in file).")
        print(f"  drive with e.g.:  mysqlslap -S {args.socket} -u root "
              f"--create-schema=recall_bench --no-drop --delimiter=';' "
              f"-q {args.emit_queries} -c <N> --number-of-queries=<TOTAL>")
        if args.keep_server:
            print(f"  KEEP-SERVER: server left running on {args.socket}; "
                  f"table+index NOT dropped.")
        return 0

    if args.keep_server:
        # Build + (optional) recall done; leave everything up for a load driver.
        print(f"build_time_s={build_s:.2f}{split}  (--keep-server: server left "
              f"up on {args.socket}, table 't' + index in place)")
        return 0

    if args.queries == 0:
        # Build-only probe (e.g. a build without the optimizer, so KNN SELECT
        # won't route). Report build time; no queries, no recall.
        print(f"build_time_s={build_s:.2f}{split}  (build-only; queries skipped)")
        return 0

    # --readers N (>1) fans queries out across N OS processes for AGGREGATE QPS
    # (read-scaling / cache-concurrency). mysql only -- psql keeps the serial
    # path (its ef_search GUC is session-scoped and rides inside the batch).
    parallel_readers = args.readers if (args.readers > 1 and core.CLIENT != "psql") else 1
    if args.readers > 1 and core.CLIENT == "psql":
        print("NOTE: --readers is mysql-only; using serial path for psql.")

    def query_at(ef):
        return (run_queries_parallel(ef, parallel_readers)
                if parallel_readers > 1 else run_queries(ef))

    dtag = "  DRY-RUN (no HNSW search; qps = harness+round-trip floor)" if args.dry_run else ""

    if sweep:
        # Build once, re-query at each ef_search value.
        rtag = f"  readers={parallel_readers} (aggregate qps)" if parallel_readers > 1 else ""
        print(f"build_time_s={build_s:.2f}{split}  (index built once; sweeping ef_search){rtag}{dtag}")
        print(f"{'ef_search':<10} {'recall@'+str(args.k):<12} {'qps':<8}")
        worst = 1.0
        for ef in sweep:
            rec, qps = query_at(ef)     # ef is set per query connection (SESSION)
            worst = min(worst, rec)
            print(f"{ef:<10} {rec:<12.4f} {qps:<8.1f}")
        if args.dry_run or args.no_gate:
            return 0                    # report-only (dry-run / comparison sweep)
        return 0 if worst >= args.threshold else 1

    # single run
    recall, qps = query_at(args.ef_search)
    rtag = f"  readers={parallel_readers}" if parallel_readers > 1 else ""
    print(f"build_time_s={build_s:.2f}{split}  qps={qps:.1f}{rtag}  recall@{args.k}={recall:.4f}  "
          f"threshold={args.threshold}{dtag}")
    if args.dry_run or args.no_gate:
        return 0                        # report-only (dry-run / comparison sweep)
    if recall >= args.threshold:
        print("PASS"); return 0
    print("FAIL (recall below threshold)"); return 1


if __name__ == "__main__":
    sys.exit(main())
