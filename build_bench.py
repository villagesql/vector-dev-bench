#!/usr/bin/env python3
"""
build_bench.py — index-BUILD benchmark for the VillageSQL vector index. Builds
the table + index (serial or parallel via --build-threads) and reports build
time + the actual committed row count, then STOPS (no queries, no recall). Use
it to measure build cost and to test concurrent-insert correctness in isolation
(a broken concurrent build shows as a row-count MISMATCH).

Shared infra (profiles, connections, datasets, the build path) lives in
harness_core.py.
"""
import argparse, sys
import harness_core as core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=core.PROFILES, default="vsql_vector")
    ap.add_argument("--metric", choices=list(core.GROUND_TRUTH), default="l2",
                    help="distance metric; must be supported by the profile")
    ap.add_argument("--dataset", choices=list(core.ANN_DATASETS), default=None,
                    help="use a real ann-benchmarks dataset (HDF5, precomputed "
                         "ground truth) instead of synthetic random vectors. "
                         "Sets dim + metric from the data; --n then caps how much "
                         "of it to use (0/unset = all).")
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--n", type=int, default=None,
                    help="rows in the table (synthetic default 2000; with "
                         "--dataset, unset/0 = use the full base set)")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--epsilon", type=float, default=1e-3,
                    help="tie tolerance for recall (unused by build_bench; kept "
                         "for shared build_index compatibility).")
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=64)
    ap.add_argument("--insert-batch", type=int, default=1000,
                    help="rows per INSERT statement (default 1000). 0 = single "
                         "statement of all n — only for the parse-cost probe; it "
                         "exceeds the MySQL/MariaDB max_allowed_packet on real "
                         "datasets (ERROR 2006), so it is NOT the default.")
    ap.add_argument("--no-index", action="store_true",
                    help="skip the custom index (SVECTOR column only) — build-cost "
                         "probe to isolate generic insert from graph maintenance")
    ap.add_argument("--index-mode", choices=["incremental", "post"],
                    default="incremental",
                    help="incremental: CREATE INDEX before inserts (graph "
                         "maintained per-insert; the only mode vsql/MariaDB "
                         "support). post: insert into an UNINDEXED table then "
                         "CREATE INDEX at the end (bulk build; pgvector's native "
                         "fast path). 'post' reports insert vs index time "
                         "separately. Requires a profile with a separate "
                         "index_ddl (not the inline-VECTOR-INDEX engines).")
    ap.add_argument("--build-threads", type=int, default=1,
                    help="number of concurrent CLIENT connections for the insert/"
                         "build phase (default 1 = serial). N>1 shards the rows "
                         "across N OS processes, each inserting its shard into the "
                         "SAME table+index concurrently -- drives parallel index "
                         "build FROM THE CLIENT (bypasses the server's internal "
                         "innodb_ddl_threads/parallel_read_threads). Use to test "
                         "whether the graph absorbs concurrent inserts. "
                         "incremental mode + mysql only. The harness checks the "
                         "committed row count -- a broken concurrent build shows "
                         "as a MISMATCH.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--socket", default=core.DEFAULT_SOCKET)
    ap.add_argument("--mysql", default=core.DEFAULT_MYSQL)
    # TCP alternative to --socket, for a server in a container with its port
    # published. When --host is given, the harness connects over TCP instead of
    # the unix socket (everything else -- profiles, SQL -- is identical).
    ap.add_argument("--host", default=None,
                    help="connect over TCP to this host instead of --socket "
                         "(e.g. 127.0.0.1 for a container with -p PORT:3306)")
    ap.add_argument("--port", type=int, default=3306,
                    help="TCP port when --host is set (default 3306)")
    # build_index expects these attrs; they are unused by the build path but the
    # shared code / print header reads them, so provide the monolith defaults.
    ap.add_argument("--queries", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--ef-search", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    # --host switches to TCP: fold host/port into the socket spec so every
    # run_sql call site stays unchanged (run_sql interprets the 'tcp:' prefix).
    if args.host:
        args.socket = f"tcp:{args.host}:{args.port}"

    prof, metric, _sweep = core.resolve_profile(args)
    r = core.build_index(args, prof, metric)

    # Insert/build done. Report build time + actual row count and STOP -- no
    # query phase, no ground truth. A short concurrent build = lost rows.
    cnt = core.run_sql(args.mysql, args.socket, "SELECT COUNT(*) FROM t;",
                       want_rows=True)
    got = int(cnt[0]) if cnt else -1
    ok = "OK" if got == args.n else f"MISMATCH (expected {args.n})"
    print(f"build_time_s={r.build_s:.2f}{r.split}  rows={got}/{args.n} {ok}  "
          f"(build-only; no queries)")
    return 0 if got == args.n else 1


if __name__ == "__main__":
    sys.exit(main())
