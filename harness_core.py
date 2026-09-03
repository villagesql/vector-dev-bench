#!/usr/bin/env python3
"""
harness_core.py — shared infrastructure for the VillageSQL vector-index
benchmarks. Owns the engine "profiles" (SQL surfaces), connection handling,
dataset loading, numpy ground truth, server-config reporting, and the shared
index-build path (build_index). The three CLIs — recall_bench.py, build_bench.py,
rw_bench.py — import from here.

Engine "profiles" (SQL surfaces) are built in:
  vsql_vector : VillageSQL SVECTOR + HNSW (recall < 1.0 = ANN quality signal).
  mariadb     : MariaDB MHNSW (VECTOR INDEX).
  pgvector    : PostgreSQL + pgvector HNSW.

Each engine is started by its start_*.sh on a scratch unix socket; the harness
talks to that socket via the engine's CLI client.
"""
import subprocess, sys, time, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_SOCKET = os.path.join(HERE, ".run", "mysqld.sock")


def detect_srv_build():
    """Server build dir: $SRV_BUILD, else VillageSQL_BUILD_DIR from the
    extension's build/CMakeCache.txt (same value used to build the .veb)."""
    env = os.environ.get("SRV_BUILD")
    if env:
        return env
    cache = os.path.join(REPO, "build", "CMakeCache.txt")
    if os.path.isfile(cache):
        with open(cache) as f:
            for line in f:
                if line.startswith("VillageSQL_BUILD_DIR:"):
                    return line.split("=", 1)[1].strip()
    return None


def default_mysql():
    srv = detect_srv_build()
    return (os.path.join(srv, "runtime_output_directory", "mysql")
            if srv else "mysql")  # fall back to PATH


DEFAULT_MYSQL = default_mysql()

# Per-engine SQL surface. {dim} etc. filled at runtime.
# Each profile declares the engine's SQL surface. `metrics` maps a metric key
# (l2/cosine/ip/l1) to that engine's SQL for it:
#   idx_modifier : column modifier in the index DDL (or "" if metric is implicit)
#   dist_fn      : the ORDER BY distance expression (uses {qlit} = query literal)
# The SAME metric must drive the index build, the query, AND the numpy ground
# truth (GROUND_TRUTH below) — mismatching them makes recall meaningless.
PROFILES = {
    "vsql_vector": {
        "extension": "vsql_vector",
        "coltype": "SVECTOR({dim})",
        # SVECTOR accepts a plain string literal (implicit conversion) — no
        # explicit FROM_STRING needed (matches the extension's own MTR tests).
        "vec_literal": "'{lit}'",
        # query-time HNSW search breadth, settable via SQL. NOTE: read only via
        # SHOW VARIABLES / not `SELECT @@global.<name>` (that read path crashes
        # for extension-namespaced dotted sysvars — separate server bug).
        "ef_search_var": "vsql_vector.ef_search",
        # HNSW: metric is chosen by BOTH the index modifier (build) and the query fn
        "index_ddl": ("CREATE INDEX idx_v ON t (v {idx_modifier}) USING EXTENDED(hnsw) "
                      "WITH (M = {M}, ef_construction = {efc})"),
        "metrics": {
            "l2":     {"idx_modifier": "hnsw_l2",            "dist_fn": "L2_DISTANCE(v, {qlit})"},
            "cosine": {"idx_modifier": "hnsw_cosine",        "dist_fn": "COSINE_DISTANCE(v, {qlit})"},
            "l1":     {"idx_modifier": "hnsw_l1",            "dist_fn": "L1_DISTANCE(v, {qlit})"},
            "ip":     {"idx_modifier": "hnsw_inner_product", "dist_fn": "INNER_PRODUCT(v, {qlit})", "order": "DESC"},
        },
    },
    # MariaDB MHNSW (13.1) — an in-database HNSW reference. VECTOR INDEX is INLINE
    # in CREATE TABLE (no separate CREATE INDEX), so this profile overrides the
    # whole table DDL via `table_ddl`. M matches vsql-vector; MariaDB's
    # ef_construction is HARDCODED at 10 (not tunable) vs vsql-vector's 200 —
    # documented, not matched. Vectors via Vec_FromText(); no gates/preview.
    "mariadb": {
        "extension": None,  # native; no INSTALL
        "coltype": "VECTOR({dim})",
        "vec_literal": "Vec_FromText('{lit}')",
        # Query-time search breadth. MariaDB floors it to max_neighbours internally;
        # SET GLOBAL applies to the fresh per-query-batch connection.
        "ef_search_var": "mhnsw_ef_search",
        # {idx_modifier} unused; M set inline. ef_construction ignored (fixed 10).
        "table_ddl": ("CREATE TABLE t (id INT PRIMARY KEY, v VECTOR({dim}) NOT NULL, "
                      "VECTOR INDEX (v) M={M}) ENGINE=InnoDB;"),
        "no_index_table_ddl": ("CREATE TABLE t (id INT PRIMARY KEY, "
                               "v VECTOR({dim}) NOT NULL) ENGINE=InnoDB;"),
        "metrics": {
            "l2":     {"idx_modifier": "", "dist_fn": "vec_distance_euclidean(v, {qlit})"},
            "cosine": {"idx_modifier": "", "dist_fn": "vec_distance_cosine(v, {qlit})"},
        },
    },
    # VARCHAR control -- NOT a vector engine. A plain wide VARCHAR column on the
    # SAME server, storing each vector as its text literal ('[...]'), NO custom
    # column, NO index, NO extension involvement. It reuses the harness's exact
    # insert path (same sharding, batching, --build-threads multiprocessing), so
    # it isolates the harness's insert code from the server's SVECTOR/custom-
    # column path: run `--profile varchar_ctrl --build-only` and if it loses rows
    # under --build-threads, the bug is the harness; if it's clean while
    # vsql_vector loses rows, the bug is server-side. Build/insert-only -- it
    # cannot answer KNN queries, so it has no metrics/index (use --build-only).
    # varchar_len is sized in main() from dim (a dim-d int vector's '[...]' text).
    "varchar_ctrl": {
        "extension": None,
        "coltype": "VARCHAR({varchar_len})",
        "vec_literal": "'{lit}'",           # store the '[...]' text verbatim
        # No index, no ef_search, no metrics -- build/insert-only control.
        "metrics": {
            "l2": {"idx_modifier": "", "dist_fn": "1"},  # placeholder; unused
        },
    },
    # pgvector (PostgreSQL) -- neutral third HNSW reference. Talks psql (client
    #="psql"); started by start_postgres.sh (scratch cluster, CREATE EXTENSION
    # vector, db "bench"). Build knobs m/ef_construction match ours (unlike
    # MariaDB's hardcoded efc). Distance OPERATORS: <-> L2, <=> cosine, <#>
    # negative inner product (pgvector already negates IP so ORDER BY ... LIMIT
    # ascending returns max-IP -- the correct MIPS convention, no DESC needed).
    "pgvector": {
        "client": "psql",
        "extension": None,
        "coltype": "vector({dim})",
        "vec_literal": "'{lit}'",           # '[...]' text cast implicitly to vector
        "ef_search_var": "hnsw.ef_search",  # SET hnsw.ef_search = N (session GUC)
        # {idx_modifier} carries the HNSW opclass, which must match the query
        # operator/metric (l2 -> vector_l2_ops, etc.).
        "index_ddl": ("CREATE INDEX ON t USING hnsw (v {idx_modifier}) "
                      "WITH (m = {M}, ef_construction = {efc})"),
        "metrics": {
            "l2":     {"idx_modifier": "vector_l2_ops",     "dist_fn": "v <-> {qlit}"},
            "cosine": {"idx_modifier": "vector_cosine_ops", "dist_fn": "v <=> {qlit}"},
            "ip":     {"idx_modifier": "vector_ip_ops",     "dist_fn": "v <#> {qlit}"},
        },
    },
}


# numpy exact ground truth per metric. Each returns per-row "distance" where
# SMALLER = nearer (so argsort ascending gives the true neighbours). For inner
# product, "nearest" = largest dot product, so we negate.
def _l2(data, q):     return np.linalg.norm(data - q, axis=1)
def _l1(data, q):     return np.sum(np.abs(data - q), axis=1)
def _cosine(data, q):
    dn = data / (np.linalg.norm(data, axis=1, keepdims=True) + 1e-12)
    qn = q / (np.linalg.norm(q) + 1e-12)
    return 1.0 - dn @ qn
# INNER_PRODUCT is a SIMILARITY: "nearest" = LARGEST dot product (per the
# extension README, "higher means more similar"). To keep the "smaller = nearer"
# invariant of this table (argsort ascending picks the true neighbours), NEGATE
# the dot so the largest dot becomes the smallest value. The IP query must order
# DESCENDING to match (see the "ip" profile's order="DESC").
def _ip(data, q):     return -(data @ q)

GROUND_TRUTH = {"l2": _l2, "l1": _l1, "cosine": _cosine, "ip": _ip}


# Standard ann-benchmarks datasets: real vectors with PRECOMPUTED ground truth,
# so recall is representative (real embeddings, not near-equidistant random) and
# comparable to published ann-benchmarks results. Each is an HDF5 with datasets
# `train` (base vectors), `test` (queries), `neighbors` (true top-k ids per
# query), and a `distance` attr. Downloaded once to DATASET_DIR and cached.
DATASET_DIR = os.path.join(HERE, ".datasets")
ANN_DATASETS = {
    # name -> (url, native metric)
    "fashion-mnist-784-euclidean": (
        "https://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5", "l2"),
    "sift-128-euclidean": (
        "https://ann-benchmarks.com/sift-128-euclidean.hdf5", "l2"),
    "glove-100-angular": (
        "https://ann-benchmarks.com/glove-100-angular.hdf5", "cosine"),
    "gist-960-euclidean": (
        "https://ann-benchmarks.com/gist-960-euclidean.hdf5", "l2"),
}


def load_ann_dataset(name):
    """Download (cache) + load an ann-benchmarks HDF5. Returns
    (train f32[N,D], test f32[Q,D], neighbors int[Q,>=k], native_metric)."""
    import h5py
    if name not in ANN_DATASETS:
        raise SystemExit(f"unknown --dataset '{name}'. Known: "
                         f"{', '.join(ANN_DATASETS)}")
    url, native_metric = ANN_DATASETS[name]
    os.makedirs(DATASET_DIR, exist_ok=True)
    path = os.path.join(DATASET_DIR, name + ".hdf5")
    if not os.path.exists(path):
        print(f"downloading {name} -> {path} ...", file=sys.stderr)
        import urllib.request
        tmp = path + ".part"
        # Some CDNs 403 the default Python-urllib UA; use a browser-like one.
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, open(tmp, "wb") as out:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        os.rename(tmp, path)
    with h5py.File(path, "r") as f:
        train = np.asarray(f["train"], dtype=np.float32)
        test = np.asarray(f["test"], dtype=np.float32)
        neighbors = np.asarray(f["neighbors"], dtype=np.int64)   # [Q, 100] true-NN ids
        distances = np.asarray(f["distances"], dtype=np.float64)  # [Q, 100] their dists
        native_metric = f.attrs.get("distance", native_metric)
    # ann-benchmarks uses "euclidean"/"angular"; map to our metric keys.
    native_metric = {"euclidean": "l2", "angular": "cosine"}.get(
        str(native_metric), str(native_metric))
    return train, test, neighbors, distances, native_metric


# Client kind for the active profile: "mysql" (mysql/mariadb CLI) or "psql"
# (PostgreSQL). Set from the profile via set_client() in each CLI; run_sql
# dispatches on it so the same harness drives MySQL/MariaDB and Postgres/pgvector
# unchanged. Shared functions read the MODULE-level CLIENT (harness_core.CLIENT),
# so set_client() in this module is authoritative across all importers.
CLIENT = "mysql"


def set_client(c):
    """Set the module-level CLIENT (called by each CLI after resolving the
    profile). Shared functions read harness_core.CLIENT, so this is the single
    source of truth even when they're called from another module."""
    global CLIENT
    CLIENT = c


def _parse_conn(socket):
    """Interpret the connection spec. A plain path is a unix socket (native);
    a 'tcp:HOST:PORT' spec is a TCP connection (e.g. a server in a container
    with its port published). Returns ('tcp', host, port) or ('socket', path,
    None). This lets one harness drive both native (socket) and containerised
    (TCP) servers with no other change."""
    if isinstance(socket, str) and socket.startswith("tcp:"):
        _, host, port = socket.split(":", 2)
        return "tcp", host, int(port)
    return "socket", socket, None


# One persistent PyMySQL connection per process, keyed by the connection spec.
# The harness is single-threaded, so a lazily-opened singleton is enough. Reusing
# one connection is the point: it avoids spawning a `mysql` process per statement
# (the old CLI path did), which dominated wall-clock on the many small statements
# a run issues and polluted the timing we care about.
_MYSQL_CONN = None
_MYSQL_CONN_KEY = None


def _mysql_conn(socket):
    """Return the persistent PyMySQL connection for this socket spec, opening it
    on first use. `MULTI_STATEMENTS` is required because callers batch several
    `;`-separated statements in one run_sql() call (as the CLI path allowed)."""
    global _MYSQL_CONN, _MYSQL_CONN_KEY
    import pymysql
    from pymysql.constants import CLIENT as PYMYSQL_CLIENT

    kind, host, port = _parse_conn(socket)
    if _MYSQL_CONN is not None and _MYSQL_CONN_KEY == socket:
        return _MYSQL_CONN

    # Match the server's large packet budget so batched INSERTs (up to
    # --insert-batch rows of high-dim vectors) are not truncated client-side.
    max_packet = 1 << 30  # 1 GiB
    if kind == "tcp":
        # bench/bench is the account the container images create.
        conn = pymysql.connect(host=host, port=port, user="bench",
                               password="bench",
                               client_flag=PYMYSQL_CLIENT.MULTI_STATEMENTS,
                               max_allowed_packet=max_packet)
    else:
        # Native unix socket: root, no password (matches the old CLI path).
        conn = pymysql.connect(unix_socket=host, user="root",
                               client_flag=PYMYSQL_CLIENT.MULTI_STATEMENTS,
                               max_allowed_packet=max_packet)
    conn.autocommit(True)
    _MYSQL_CONN, _MYSQL_CONN_KEY = conn, socket
    return conn


def _new_mysql_conn(socket, database=None):
    """Open a FRESH, independent PyMySQL connection (not the cached global one).
    Used by the concurrent worker processes (--readers / --build-threads): each
    worker is its own OS process (multiprocessing -- no GIL) and owns one
    connection. `database` selects the schema AT CONNECT (so statements need no
    `USE ...;` prefix and can use unqualified table names)."""
    import pymysql
    from pymysql.constants import CLIENT as PYMYSQL_CLIENT

    kind, host, port = _parse_conn(socket)
    max_packet = 1 << 30
    if kind == "tcp":
        conn = pymysql.connect(host=host, port=port, user="bench",
                               password="bench", database=database,
                               client_flag=PYMYSQL_CLIENT.MULTI_STATEMENTS,
                               max_allowed_packet=max_packet)
    else:
        conn = pymysql.connect(unix_socket=host, user="root", database=database,
                               client_flag=PYMYSQL_CLIENT.MULTI_STATEMENTS,
                               max_allowed_packet=max_packet)
    conn.autocommit(True)
    return conn


# Per-worker-process persistent connection, opened once in the pool initializer
# so connection setup stays OUT of the timed query region.
_WORKER_CONN = None


def _reader_init(socket):
    """multiprocessing.Pool initializer: each worker process opens ONE connection
    up front (before any timed work), stored per-process."""
    global _WORKER_CONN
    _WORKER_CONN = _new_mysql_conn(socket)


def _reader_warmup(_i):
    """Untimed pool task: touch the per-process connection so it is fully open
    (initializer already opened it) before the timed query region starts."""
    _WORKER_CONN.ping(reconnect=True)
    return True


def _reader_run(sql_text):
    """One concurrent reader task (runs in a separate OS process -- no GIL). Runs
    its shard of `;`-batched query SQL on the process's pre-opened connection and
    returns the RAW result lines (tab-joined, @@Q-delimited). Does NO parsing or
    recall -- the parent scores the merged lines AFTER the timed region ends, so
    the measured wall-clock is server query execution only, not client-side CPU."""
    cur = _WORKER_CONN.cursor()
    lines = []
    cur.execute(sql_text)
    while True:
        for row in cur.fetchall():
            lines.append("\t".join("" if v is None else str(v) for v in row))
        if not cur.nextset():
            break
    return lines


def _insert_run(args):
    """One concurrent INSERTER task (separate OS process). Owns its connection's
    FULL lifecycle: open -> insert its shard (independent per-batch INSERTs, not
    one mega-statement) -> commit -> close. Closing the connection inside the task
    is the commit barrier: when pool.map() returns, every worker connection is
    closed, so every insert is committed and durable BEFORE the parent counts
    rows -- no race between the tail commits and the COUNT(*). (This is why
    inserts do NOT share the reader's persistent pool connection: readers want the
    connection open across the timed region; inserters need close-as-commit.)
    Drives PARALLEL INDEX BUILD from the client side (N connections inserting into
    the same table+index concurrently) to test whether the graph absorbs
    concurrent inserts, isolating it from the server's internal parallel-DDL.
    `args` = (socket, database, stmts). Returns (rows_executed, error): error is
    None on success, else the exception string of the FIRST failing batch."""
    socket, database, stmts = args
    conn = _new_mysql_conn(socket, database=database)  # schema selected at connect
    n = 0
    try:
        cur = conn.cursor()
        for stmt in stmts:
            try:
                cur.execute(stmt)
            except Exception as e:  # noqa: BLE001 -- surface any insert error
                return (n, str(e))
            if cur.rowcount and cur.rowcount > 0:
                n += cur.rowcount
        conn.commit()  # redundant under autocommit, but makes the barrier explicit
        return (n, None)
    finally:
        conn.close()   # <- commit barrier: rows durable once this returns


def run_sql(client_bin, socket, sql, want_rows=False):
    """Run SQL via the active client. `client_bin` is used only for the psql
    path; the MySQL path uses a persistent PyMySQL connection. `socket` is a
    unix socket path (native), a socket DIR (psql), or a 'tcp:HOST:PORT' spec.
    Returns a list of lines, each row rendered as tab-joined columns with no
    header -- matching the old `mysql --batch --raw --silent` output, so every
    caller's line parsing (split on '\\t', @@Q markers, digit rows) is unchanged."""
    if CLIENT == "psql":
        kind, host, port = _parse_conn(socket)
        # -h selects host: a directory path is the unix socket, a hostname is
        # TCP (add -p for the port). -tA = tuples-only, unaligned. SQL is piped
        # on STDIN (-f -), NOT via -c: -c is subject to ARG_MAX, which high-dim
        # datasets blow past; stdin has no such limit.
        if kind == "tcp":
            cmd = [client_bin, "-h", host, "-p", str(port), "-U", "bench",
                   "-d", "bench", "-v", "ON_ERROR_STOP=1", "-tA", "-f", "-"]
        else:
            cmd = [client_bin, "-h", host, "-d", "bench", "-v", "ON_ERROR_STOP=1",
                   "-tA", "-f", "-"]
        p = subprocess.run(cmd, input=sql, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"SQL failed (rc={p.returncode}):\n{p.stderr}\n"
                               f"--- sql ---\n{sql}")
        return p.stdout.strip().splitlines()

    # MySQL / vsql / MariaDB via PyMySQL.
    conn = _mysql_conn(socket)
    lines = []
    try:
        cur = conn.cursor()
        cur.execute(sql)
        # Concatenate every result set's rows in order. The CLI streamed all
        # result sets' rows sequentially; the @@Q-marker recall parser and the
        # config parser both rely on that flat ordering.
        while True:
            rows = cur.fetchall()
            for row in rows:
                lines.append("\t".join("" if c is None else str(c) for c in row))
            if not cur.nextset():
                break
        cur.close()
    except Exception as e:
        raise RuntimeError(f"SQL failed: {e}\n--- sql ---\n{sql}")
    return lines


def vec_lit(v):
    return "[" + ",".join(f"{x:.6g}" for x in v) + "]"


# Config the harness pulls FROM THE LIVE SERVER (not from the launch flags — a
# flag can be silently clamped, ignored, or overridden by a config file; only the
# running server tells the truth). Reported as a header so every result file
# self-documents the config it was measured under, and warns (never fails) if the
# buffer pool / shared_buffers is smaller than the dataset's working set, which
# would make the run I/O-bound and not comparable.
def report_config(client_bin, socket, working_set_bytes):
    def q(sql):
        try:
            return run_sql(client_bin, socket, sql)
        except Exception as e:
            return [f"(config query failed: {e})"]

    pool_bytes = None
    if CLIENT == "psql":
        rows = q("SELECT name||'='||setting||COALESCE(' '||unit,'') FROM pg_settings "
                 "WHERE name IN ('shared_buffers','maintenance_work_mem','work_mem',"
                 "'max_parallel_maintenance_workers','max_worker_processes',"
                 "'fsync','synchronous_commit','wal_level');")
        # shared_buffers is reported in 8kB blocks
        sb = q("SELECT setting::bigint * (SELECT setting::bigint FROM pg_settings "
               "WHERE name='block_size') FROM pg_settings WHERE name='shared_buffers';")
        try:
            pool_bytes = int(sb[0])
        except (ValueError, IndexError):
            pool_bytes = None
    else:
        # `SHOW GLOBAL VARIABLES WHERE ...` is the portable form: it works on BOTH
        # MySQL/vsql AND MariaDB. (The I_S/P_S global_variables TABLES live in
        # different schemas per engine — I_S on MariaDB, P_S on MySQL 8 — so
        # querying either table directly breaks on the other engine.) Returns
        # `name<TAB>value` rows. mhnsw_max_cache_size is MariaDB-only; it simply
        # doesn't come back on MySQL/vsql.
        raw = q("SHOW GLOBAL VARIABLES WHERE Variable_name IN "
                "('innodb_buffer_pool_size','innodb_flush_log_at_trx_commit',"
                "'innodb_doublewrite','innodb_flush_method','innodb_io_capacity',"
                "'max_allowed_packet','mhnsw_max_cache_size')")
        gv = {}
        for line in raw:
            parts = line.split("\t", 1)
            if len(parts) == 2:
                gv[parts[0].strip()] = parts[1].strip()
        rows = [f"{k}={v}" for k, v in sorted(gv.items())]
        try:
            pool_bytes = int(gv.get("innodb_buffer_pool_size", ""))
        except ValueError:
            pool_bytes = None

        # MariaDB only: mhnsw_max_cache_size caps the in-memory HNSW graph. If it
        # can't hold the graph, MariaDB reset()s the graph context on overflow,
        # which discards the LEARNED ef_power stat that sizes its (approximate,
        # bloom-filter) visited-set — under-sizing the filter, raising its false-
        # positive rate, and lowering BOTH build speed AND recall (verified). The
        # 16MB default is far too small for real datasets. Recommend (and warn
        # below if under) max(512MB, 2 x working set) — enough to hold the graph
        # with headroom, and it self-scales to the dataset (survives 1M+ runs).
        try:
            mhnsw_cache_bytes = int(gv["mhnsw_max_cache_size"])
        except (KeyError, ValueError):
            mhnsw_cache_bytes = None  # not MariaDB (var absent) -> skip the check

    print("server_config: " + " ".join(r.strip() for r in rows if r.strip()))
    # MariaDB HNSW-graph-cache sizing warning (only when the var exists).
    if CLIENT != "psql" and mhnsw_cache_bytes is not None and working_set_bytes:
        recommended = max(512 << 20, 2 * working_set_bytes)
        if mhnsw_cache_bytes < recommended:
            print(f"WARNING: mhnsw_max_cache_size (~{mhnsw_cache_bytes//(1<<20)} MB) "
                  f"is below the recommended ~{recommended//(1<<20)} MB "
                  f"(= max(512MB, 2x working set)) for this dataset. MariaDB will "
                  f"reset() the graph context on cache overflow, which lowers BOTH "
                  f"build speed AND recall. Launch with "
                  f"--mhnsw-max-cache-size={recommended} (e.g. via MARIADBD_EXTRA) "
                  f"to make it comparable.", file=sys.stderr)
            print(f"WARNING: mhnsw_max_cache_size ~{mhnsw_cache_bytes//(1<<20)} MB "
                  f"< recommended ~{recommended//(1<<20)} MB — MariaDB build/recall "
                  f"will be degraded")
    if pool_bytes is not None and working_set_bytes is not None:
        if pool_bytes < working_set_bytes:
            print(f"WARNING: buffer pool / shared_buffers (~{pool_bytes//(1<<20)} MB) "
                  f"is smaller than the dataset working set "
                  f"(~{working_set_bytes//(1<<20)} MB) — this run may be I/O-bound, "
                  f"not memory-resident, and NOT comparable to a run that fits in "
                  f"cache. Raise the pool (e.g. --innodb-buffer-pool-size / "
                  f"SHARED_BUFFERS) to make it fair.", file=sys.stderr)
            print(f"WARNING: cache smaller than working set "
                  f"(~{pool_bytes//(1<<20)} MB < ~{working_set_bytes//(1<<20)} MB) "
                  f"— run may be I/O-bound")


class BuildResult:
    """Everything the CLIs need after build_index(): the generated vectors, the
    per-query ground truth (for recall), the `lit` closure (vector -> SQL
    literal), timing, the 'post'-mode insert/index split, post_index, and the
    (possibly dataset-overridden) metric dict for the query phase."""
    def __init__(self, data, queries, dataset_truth, lit, build_s, insert_s,
                 index_s, split, post_index, metric):
        self.data = data
        self.queries = queries
        self.dataset_truth = dataset_truth
        self.lit = lit
        self.build_s = build_s
        self.insert_s = insert_s
        self.index_s = index_s
        self.split = split
        self.post_index = post_index
        self.metric = metric


def resolve_profile(args):
    """Resolve the profile + metric from args, set the module CLIENT, and parse
    the ef-search sweep. Returns (prof, metric, sweep) or exits (SystemExit) with
    the same messages/codes the monolith used. Callers do this before build_index
    so the returned prof/metric can be reused for the query phase."""
    prof = PROFILES[args.profile]
    set_client(prof.get("client", "mysql"))
    if args.metric not in prof["metrics"]:
        print(f"ERROR: profile '{args.profile}' does not support metric "
              f"'{args.metric}'. Supported: {', '.join(prof['metrics'])}.", file=sys.stderr)
        raise SystemExit(2)
    metric = prof["metrics"][args.metric]

    sweep = None
    if getattr(args, "ef_search_sweep", None) is not None:
        sweep = [int(x) for x in args.ef_search_sweep.split(",") if x.strip()]
    if (getattr(args, "ef_search", None) is not None or sweep) and "ef_search_var" not in prof:
        print(f"ERROR: profile '{args.profile}' does not expose a query-time "
              f"ef_search knob.", file=sys.stderr)
        raise SystemExit(2)
    return prof, metric, sweep


def build_index(args, prof, metric):
    """Shared build path: select the data source (dataset vs synthetic, mutating
    args.n/args.queries/args.dim/args.metric as the monolith did), create the
    schema, insert the rows (serial or parallel via --build-threads), and — in
    'post' mode — build the index in bulk after inserts. Returns a BuildResult.

    Caller must have already run resolve_profile(args) (so CLIENT is set and
    prof/metric are resolved). SystemExit(2) on the same error conditions as the
    monolith."""
    # Data source: a real ann-benchmarks dataset (precomputed ground truth) or
    # synthetic random vectors (ground truth computed in numpy below).
    dataset_truth = None
    if args.dataset:
        train, test, neighbors, gt_dist, native_metric = load_ann_dataset(args.dataset)
        if args.metric != native_metric:
            print(f"NOTE: --dataset {args.dataset} is a {native_metric} dataset; "
                  f"overriding --metric {args.metric} -> {native_metric} "
                  f"(its ground truth is only valid for that metric).",
                  file=sys.stderr)
            args.metric = native_metric
            metric = prof["metrics"][args.metric]
        args.dim = train.shape[1]
        # --n / --queries cap how much of the dataset to use (0/None = all).
        n = args.n if args.n and args.n > 0 else train.shape[0]
        n = min(n, train.shape[0])
        q = args.queries if args.queries and args.queries > 0 else test.shape[0]
        q = min(q, test.shape[0])
        data = train[:n]
        queries = test[:q]
        args.n, args.queries = n, q
        # Tie-tolerant truth from the precomputed top-100 neighbors + distances
        # (ann-benchmarks style, O(100)/query — no full-base scan). Threshold =
        # k-th true distance x (1+epsilon); accept the precomputed neighbors
        # within it. Only valid when using the FULL train set (the precomputed
        # ids are over all of train); if --n subsets, fall back below.
        if n == train.shape[0]:
            dataset_truth = []
            for qi in range(q):
                thresh = gt_dist[qi, args.k - 1] * (1.0 + args.epsilon)
                acc = neighbors[qi][gt_dist[qi] <= thresh]
                dataset_truth.append(set(acc.tolist()))
        else:
            print(f"NOTE: --n {n} subsets a {train.shape[0]}-row dataset; "
                  f"recomputing ground truth over the subset (dataset neighbor "
                  f"ids are only valid over the full base).", file=sys.stderr)
    else:
        if not args.n:
            args.n = 2000
        if not args.queries:
            args.queries = 200
        rng = np.random.default_rng(args.seed)
        data = rng.standard_normal((args.n, args.dim)).astype(np.float32)
        queries = rng.standard_normal((args.queries, args.dim)).astype(np.float32)

    # VARCHAR control sizes its column to hold a dim-d vector's '[...]' text:
    # up to ~ (max-int-chars + comma) per element, plus brackets. Generous cap.
    varchar_len = max(64, args.dim * 12 + 4)
    coltype = prof["coltype"].format(dim=args.dim, varchar_len=varchar_len)
    # index_ddl only exists for the separate-CREATE-INDEX profiles; profiles with
    # a full table_ddl (e.g. mariadb) don't have it.
    index_ddl = (prof["index_ddl"].format(
        M=args.M, efc=args.ef_construction, idx_modifier=metric["idx_modifier"])
        if "index_ddl" in prof else "")

    print(f"profile={args.profile} metric={args.metric} dim={args.dim} n={args.n} queries={args.queries} "
          f"k={args.k} M={args.M} ef_construction={args.ef_construction} ef_search={args.ef_search}")

    # Pull the actual config from the live server (see report_config). Working-set
    # floor = N * dim * 4 bytes (the raw float32 vectors); the HNSW graph adds
    # more, so this is a conservative "at least this must fit in cache" estimate.
    report_config(args.mysql, args.socket, args.n * args.dim * 4)

    # Gates (preview / hypergraph / custom-index debug) and extension install are
    # baked in as server defaults by start_server.sh, so no per-session SET needed.
    # --- setup: schema. --no-index skips the custom index (SVECTOR column only)
    # to isolate generic column-store row insertion from HNSW graph maintenance.
    # (Recall is meaningless without the index — --no-index is a build-cost probe.)
    # Profiles may supply a full table DDL (e.g. MariaDB's inline VECTOR INDEX);
    # otherwise use the default "CREATE TABLE" + separate index_ddl form.
    # index-mode 'post' = insert first, CREATE INDEX after (bulk build). It needs
    # a profile with a SEPARATE index_ddl; the inline-VECTOR-INDEX engines
    # (table_ddl profiles, e.g. MariaDB) declare the index in CREATE TABLE and
    # build incrementally by architecture, so post isn't available for them.
    post_index = None
    if args.index_mode == "post" and not args.no_index:
        if "index_ddl" not in prof:
            print(f"ERROR: --index-mode post needs a profile with a separate "
                  f"index_ddl; '{args.profile}' builds the index inline in "
                  f"CREATE TABLE (incremental only).", file=sys.stderr)
            raise SystemExit(2)
        post_index = index_ddl                       # run AFTER inserts, timed

    if "table_ddl" in prof:
        key = "no_index_table_ddl" if args.no_index else "table_ddl"
        schema = prof[key].format(dim=args.dim, M=args.M)
    else:
        # incremental: index in setup; post/no-index: table only, index later/never
        ddl_line = "" if (args.no_index or post_index) else index_ddl + ";"
        schema = f"CREATE TABLE t (id INT PRIMARY KEY, v {coltype} NOT NULL);\n{ddl_line}"
    # MySQL/MariaDB use a dedicated `recall_bench` database; Postgres uses the
    # pre-created `bench` db (psql already connects to it). No per-statement
    # `USE ...;` prefix -- the connection selects the schema once (below), so
    # every statement runs in-context with unqualified table names.
    use_prefix = ""
    if CLIENT == "psql":
        setup = f"DROP TABLE IF EXISTS t;\n{schema}"
    else:
        setup = f"""
DROP DATABASE IF EXISTS recall_bench;
CREATE DATABASE recall_bench; USE recall_bench;
{schema}
"""
    run_sql(args.mysql, args.socket, setup)
    # Select the working schema on the persistent connection so all later
    # run_sql() statements run in-context (no USE prefix). The setup's own
    # trailing `USE recall_bench` set it for the setup batch, but be explicit so
    # it survives even if run_sql reconnects.
    if CLIENT != "psql":
        _mysql_conn(args.socket).select_db("recall_bench")

    def lit(v): return prof["vec_literal"].format(lit=vec_lit(v), dim=args.dim)

    # --- build: INSERT all rows (timed). ef_search is query-time, so a sweep
    # re-queries this same graph without rebuilding.
    # --insert-batch controls rows per INSERT statement (default 1000; 0 = one
    # giant statement, which blows past MySQL/MariaDB max_allowed_packet on real
    # datasets — kept only for the parse-cost probe). Fixed-size batches keep
    # statement-parse cost constant across N, isolating server insert/index cost
    # from harness SQL-parse cost. ---
    bs = args.insert_batch if args.insert_batch and args.insert_batch > 0 else args.n

    def _insert_batch_stmts(row_range):
        # Build a LIST of independent per-batch INSERT statements for a contiguous
        # row range (one statement per --insert-batch rows), NOT one joined blob.
        # The worker executes them one at a time, so each is a normal separately-
        # parsed INSERT (same shape as the serial path).
        stmts = []
        lo, hi = row_range
        for start in range(lo, hi, bs):
            rows = ",\n".join(f"({i}, {lit(data[i])})"
                              for i in range(start, min(start + bs, hi)))
            stmts.append(f"INSERT INTO t VALUES\n{rows}")
        return stmts

    build_threads = (args.build_threads
                     if args.build_threads > 1 and CLIENT != "psql" else 1)
    if args.build_threads > 1 and CLIENT == "psql":
        print("NOTE: --build-threads is mysql-only; using serial insert for psql.")

    if build_threads > 1:
        # CLIENT-DRIVEN parallel build: shard rows across N OS processes, each
        # inserting its contiguous shard into the SAME table+index concurrently.
        # In incremental mode the index already exists (created in setup), so each
        # INSERT triggers per-row graph maintenance -- this exercises the graph's
        # CONCURRENT-insert path, isolating it from the server's internal
        # parallel-DDL. Each worker OWNS its connection (open->insert->commit->
        # close inside _insert_run), so when pool.map() returns, every insert is
        # committed and durable -- the COUNT(*) that follows is exact, no race.
        import multiprocessing as mp
        edges = [round(k * args.n / build_threads) for k in range(build_threads + 1)]
        shards = [(edges[k], edges[k + 1]) for k in range(build_threads)]
        shards = [s for s in shards if s[1] > s[0]]
        # Each task = (socket, database, per-batch-INSERT-statements). The worker
        # opens its own connection with the schema selected (no USE prefix, no
        # shared pool connection).
        _db = None if CLIENT == "psql" else "recall_bench"
        tasks = [(args.socket, _db, _insert_batch_stmts(s)) for s in shards]
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=len(tasks))
        try:
            t0 = time.time()
            worker_results = pool.map(_insert_run, tasks)  # timed region
            insert_s = time.time() - t0
        finally:
            pool.close()
            pool.join()
        errs = [r for r in worker_results if r and r[1] is not None]
        for r in errs:
            print(f"  build-threads worker error (shard rows_done={r[0]}): {r[1]}")
        # Authoritative correctness gate: the ACTUAL committed row count. All
        # worker connections are closed (pool.map returned), so every insert is
        # durable -- a single fresh read is exact, no poll/settle needed.
        cnt = run_sql(args.mysql, args.socket,
                      "SELECT COUNT(*) FROM t;", want_rows=True)
        got = int(cnt[0]) if cnt else -1
        status = "OK" if got == args.n and not errs else "MISMATCH!"
        print(f"  build-threads={build_threads}: {got}/{args.n} rows committed "
              f"{status}{' (worker errors above)' if errs else ''}")
    else:
        # Build ALL the INSERT SQL strings BEFORE timing -- the vector->text
        # formatting (lit()) is expensive Python work and must NOT be inside the
        # timed region, or the serial baseline is charged for string-building that
        # the parallel path (which pre-builds its statements) excludes. Timing
        # only the server round-trips keeps serial vs --build-threads comparable.
        serial_stmts = _insert_batch_stmts((0, args.n))
        t0 = time.time()
        for stmt in serial_stmts:
            run_sql(args.mysql, args.socket, f"{stmt};")
        insert_s = time.time() - t0

    # 'post' mode: the index does NOT exist yet — build it now (bulk), timed
    # separately so insert vs index cost is visible.
    index_s = 0.0
    if post_index:
        ti = time.time()
        run_sql(args.mysql, args.socket, f"{post_index};")
        index_s = time.time() - ti

    build_s = insert_s + index_s
    # For 'post' mode show the split (insert vs bulk CREATE INDEX); it IS the
    # point of the mode. Empty in incremental mode.
    split = (f" [insert={insert_s:.2f} index={index_s:.2f}]" if post_index else "")

    return BuildResult(data, queries, dataset_truth, lit, build_s, insert_s,
                       index_s, split, post_index, metric)
