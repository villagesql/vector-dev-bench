# AGENTS.md — guide for coding agents

Orientation for agents working in `vector-dev-bench`. Read the
[README](README.md) for what the harness *is* and its caveats, and the
[RUNBOOK](RUNBOOK.md) for how to build the engines and reproduce runs. This file
covers the conventions and traps that aren't obvious from the code.

## What this repo is

A small Python harness that drives several in-database vector (HNSW/ScaNN)
indexes and measures recall, build time, and read-under-write behaviour. It is a
**development** iteration tool, not a published-comparison framework — keep the
language neutral between engines.

## Architecture (where things live)

- **`harness_core.py`** owns almost everything: the engine `PROFILES` dict (SQL
  surfaces), connection handling, dataset loading, numpy ground truth, and the
  shared `build_index()`. Start here.
- **Three thin CLIs** front-end it, all importing `harness_core`:
  - `recall_bench.py` — recall@k + QPS across an ef_search sweep.
  - `build_bench.py` — index build time; `--build-threads N` = client-side
    parallel build.
  - `rw_bench.py` — read-under-write (readers vs writers on one index).
- **`start_*.sh`** — one per engine; each boots a throwaway scratch cluster on a
  unix socket and prints it. `run_sweep.sh` / `run_three_way.sh` orchestrate.

**To add or change an engine:** edit the `PROFILES` dict in `harness_core.py`
(each profile declares its `client`, `ef_param`, and per-`metric` SQL) and add a
matching `start_<engine>.sh`. Registered profiles today: `vsql_vector`,
`mariadb`, `pgvector`, `google` (Google MySQL + ScaNN), and `varchar_ctrl` (a
plain-VARCHAR insert-only control). All but `pgvector` use the `mysql` client
kind; `pgvector` uses `psql`.

**`ef_param` — how the engine takes the query-time search-breadth knob.** Each
profile declares one of two styles, and the harness passes ef accordingly:
- `{"style": "set", "var": NAME}` — a **separate statement** (`SET SESSION NAME =
  v`, or `SET NAME = v` for psql) emitted before the queries; the query then
  reads it implicitly. Used by the session-var engines (`vsql_vector`,
  `mariadb`, `pgvector`).
- `{"style": "inline"}` — ef rides **inside the query** itself; the metric's
  `dist_fn` carries an `{ef}` placeholder the harness formats per query (e.g.
  ScaNN's `APPROX_DISTANCE(..., 'num_leaves_to_search={ef}')`). Used by `google`.
A profile with neither (`varchar_ctrl`) rejects an ef sweep. Both styles are
threaded through the single-query, parallel-reader, and `--emit-queries` paths.

## Asked to RUN a benchmark? Follow this order

Do not start probing the filesystem or inventing parameters. Work these steps in
order; each one either has an answer already, or is a question for the user.
Stop and ask rather than guess — a wrong guess here costs a long run.

1. **What is being measured?** Recall sweep (`recall_bench.py`), build time
   (`build_bench.py`), or read-under-write (`rw_bench.py`)?

2. **Which engine(s)?** `vsql_vector`, `mariadb`, `pgvector`, `google`, or a
   comparison of several (`run_sweep.sh` / `run_three_way.sh` orchestrate
   one-at-a-time runs). If the user didn't say, ask — do not assume vsql.

3. **Where do the binaries come from?** For each engine chosen, establish which
   of these applies — **ask the user, don't go hunting**:
   - already-built trees the user will point you at (the common case here), or
   - installed packages / Homebrew kegs (MariaDB, Postgres), or
   - a Docker image (`start_server_docker.sh`), or
   - you are expected to build them (RUNBOOK §1 — this is slow; confirm first).

   Then set the engine's env var: `SRV_BUILD` (vsql), `MB` (MariaDB), `PG`
   (Postgres). **`SRV_BUILD` has no usable default** — `detect_srv_build()` only
   finds a build when this repo sits *next to* the extension checkout, which is
   not the normal layout. Note the server build dir is whatever the user built
   (e.g. `build-relwithdebinfo`), not necessarily the RUNBOOK's `build-rel`.

4. **Verify the binaries before running anything.** See "Preflight" below. A
   long sweep that dies at minute 40 on a missing `.veb` is the failure mode this
   step exists to prevent.

5. **Smoke-test, then run.** Always do a tiny run first (§ Preflight step 5)
   before the real sweep. Only then run the full job, under `caffeinate` if long.

6. **Report honestly.** If the smoke test shows `recall@10` pinned at 1.0 and
   flat across ef_search, the index is not being used — say so rather than
   reporting the numbers as an index result.

### Preflight (run these before any long job)

For a vsql run, in order — each is a one-liner that fails fast:

1. Server binary exists: `$SRV_BUILD/runtime_output_directory/mysqld`.
2. The `.veb` is **staged and fresh**: the extension's build output must be
   copied into `$SRV_BUILD/veb_output_directory/`, and its mtime must be *newer*
   than the extension build. A stale staged `.veb` means you benchmark old code
   (RUNBOOK §1). This copy is a manual step — nothing does it for you.
3. Boot: `SRV_BUILD=... EXTENSIONS=<ext> bash start_server.sh`.
4. **Confirm the extension actually installed** — `start_server.sh` is tolerant
   of a failed `INSTALL` and still prints READY, so check explicitly:
   `SELECT * FROM information_schema.extensions;` must be non-empty. If it is
   empty, re-run the `INSTALL EXTENSION <ext>;` by hand to see the real error
   (`VEB file not found: <ext>.veb` means step 2 was skipped or the name is
   wrong — the SQL name, `.veb` basename, and manifest `"name"` must all match).
5. Dataset present under `.datasets/`, and its parameters match your flags: the
   filename encodes them — `fashion-mnist-784-euclidean` ⇒ `--dim 784
   --metric l2`. Don't invent a dim/metric that disagrees with the file.
6. Smoke run (small `--n`, seconds not minutes) — expect `recall@10` ≈ 0.95–1.0,
   no ERROR, and recall that *falls* as ef_search drops.

## Setup & verification

- **Setup:** `./setup.sh` — creates `.venv` and installs `requirements.txt`
  (numpy, pymysql, h5py). The run scripts auto-select `.venv/bin/python`. Do not
  `pip install` into the system Python.
- **There are no unit tests, linter config, or CI in this repo.** After editing,
  the real verification path is:
  1. `python3 -m py_compile *.py` (syntax) and `bash -n *.sh` (shell syntax).
  2. `.venv/bin/python -c "import harness_core"` (imports cleanly).
  3. A smoke run against a live engine — see RUNBOOK §2 (small `--n`, expect
     `recall@10` in ~0.95–1.0 and no ERROR).

## Non-obvious traps

- **vsql and mariadb share the same client.** Both are driven by **PyMySQL**, not
  a native CLI — only the emitted SQL differs. The `--mysql` flag is therefore
  *vestigial for the query path* (PyMySQL drives it regardless of which binary
  you point at); it still feeds the `start_*.sh` scripts and config reporting.
  `pgvector` is the one engine that actually shells out to its CLI (`psql`).
- **Optimized builds only.** An engine (or extension) built `-O0` understates
  results many-fold — the distance kernel must be vectorized. This is an
  engine-build concern (RUNBOOK §1/§6), but don't cite numbers from an unverified
  build.
- **Recall pinned at exactly 1.0** that doesn't fall at low ef_search means the
  index is *not being used* (full-scan fallback), not a great index.
- **One engine at a time** for timing at scale; wrap long runs in `caffeinate`
  (an idle macOS machine throttled a run ~13x).

## Repo hygiene (this is a public repo)

- **Keep benchmark result files OUT of git.** `RESULTS.md`, the GCP brief, any
  `scratch-*/` dirs, and generated `queries_*.sql` are intentionally untracked —
  never `git add -A` them in. Prefer explicit `git add <file>` over `-A`.
- **No local/personal paths** (`/Users/...`, `$HOME/...`) or personal branch
  names in tracked files; use env-var placeholders and Homebrew-style examples.
- **Neutral language between engines** — describe asymmetries factually, don't
  editorialize about one engine vs another.
- **License headers.** Every new `.py`/`.sh` source file gets this header right
  after the shebang (docs, config, and `LICENSE` itself are left un-headered):
  ```
  # Copyright (c) 2026 VillageSQL Contributors
  # SPDX-License-Identifier: Apache-2.0
  ```

## Git / PR workflow

`main` is protected: no force-push, and changes land via a PR with at least one
approving review. Branch, push, open a PR (`gh pr create`), and let a human
reviewer approve before merge. Do not commit or push unless asked.
