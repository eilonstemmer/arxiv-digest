# arxiv-digest — Project brief

This is the canonical design document for **arxiv-digest**. It is the source
of truth for every implementation decision. If `README.md`, code comments, or
any other file disagrees with this brief, this brief wins. When the brief
itself needs to change, change it here first and let downstream artifacts
catch up.

---

## 1. What this project is

A local Docker stack that:

1. Pulls fresh arXiv papers weekly across configurable categories.
2. Locally pre-filters papers against the operator's profile embedding to cut
   API spend (see §3).
3. Ranks the survivors against a user-defined research profile.
4. Summarizes the top ~80 with structured extraction and cross-domain pattern
   detection.
5. Serves the result as a browsable local webapp at `http://localhost:8080`.
6. Optionally emails an HTML digest via SMTP.
7. Aggregates weekly digests into monthly and yearly rollups.

It is **open-source, generalized, and single-operator**. Anyone should be able
to clone the public repo, configure their interests, and get a useful digest.
No user-specific context goes in the code or the example config — only in
`data/` (gitignored) and `.env` (gitignored).

**Cost target:** under **$2/week** on the Anthropic API for ~2,000 papers
ingested and ~80 summarized weekly (after the local prefilter; see §3).

---

## 2. Hard constraints

- **This is a public GitHub repository.** Never commit `.env`, never commit
  anything under `data/`, never hardcode secrets, never write example values
  that look like real API keys. The `.env.example` file is the only place env
  var names appear with placeholder values.
- **All operator-specific configuration is runtime, not code.** The relevance
  profile is a markdown file mounted at runtime. The arXiv categories are a
  YAML file mounted at runtime. The API key and email config are stored in
  the SQLite database, written via the webapp's Settings tab (with `.env` as
  bootstrap fallback for headless setups).
- **Three containers** orchestrated by `docker-compose.yml`: `pipeline`
  (batch worker), `webapp` (FastAPI + Jinja2 + HTMX), `scheduler` (Alpine +
  cron, triggers pipeline via `docker exec`).
- **Single SQLite database** at `/data/digest.db` in a mounted volume. No
  Postgres, no separate vector DB.
- **Python 3.12.** `uv` for dependency management. `ruff` and `mypy --strict`
  must pass on every module.
- **All Anthropic API calls use the Batch API** for 50% cost reduction,
  except cluster-labeling and trend-narrative (one-off per run). Use prompt
  caching on static system prompts.
- **All LLM outputs are structured JSON validated against JSON Schema** files
  in `prompts/`. On validation failure: retry once with an error note, then
  skip the item and log the failure. Never accept unstructured output.
- **No secrets in logs.** Log IDs, counts, durations, status codes. Never log
  abstracts, API key fragments, or anything from the database content fields.
- **arXiv rate limit is 3 seconds between requests.** Hard constraint — use a
  token bucket.
- **Reproducible.** `docker compose up` from a clean checkout, with only
  `WEBHOOK_SECRET` set in `.env`, must succeed and let the operator complete
  setup via the webapp Settings tab.

---

## 3. System architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  scheduler  ──cron──▶  docker exec pipeline weekly              │
│                                                                 │
│  pipeline:                                                      │
│    1. ingest    arXiv API → papers table                        │
│    2. embed     sentence-transformers (local CPU)               │
│    3. triage    [a] prefilter (local, free):                    │
│                     • embed profile.md (cached by hash)         │
│                     • cosine sim per paper → prefilter_score    │
│                     • keep top `triage_prefilter_keep_fraction` │
│                       (default 0.4 → ~800 of ~2,000 papers)     │
│                     • dropped papers: relevance_score=0, kept   │
│                       in DB, no Haiku call                      │
│                 [b] Claude Haiku 4.5 Batch on survivors →       │
│                     relevance_score, reason, flags              │
│    4. cluster   HDBSCAN + Claude Sonnet 4.6 cluster labels      │
│    5. summarize Claude Sonnet 4.6 Batch (top 80 papers)         │
│    6. trend     SQL aggregations: W-1, W-4, W-12, W-52          │
│    7. render    Jinja2 → self-contained HTML                    │
│    8. notify    POST /webhook/digest-ready → webapp             │
│                 (webapp triggers optional SMTP)                 │
│                                                                 │
│  webapp:                                                        │
│    GET  /                       latest digest                   │
│    GET  /digests                list all digests                │
│    GET  /digests/{id}           specific digest                 │
│    GET  /papers/{arxiv_id}      single paper detail             │
│    POST /papers/{id}/upvote     mark paper as relevant          │
│    GET  /trends                 interactive trend page          │
│    GET  /search?q=...           full-text + semantic search     │
│    GET  /settings               settings page (UI)              │
│    POST /settings               save settings (HTMX)            │
│    POST /webhook/digest-ready   internal pipeline callback     │
└─────────────────────────────────────────────────────────────────┘
```

The **prefilter** is local, free, and the main lever for hitting the
<$2/week cost target. Its `keep_fraction` is operator-tunable from the
Settings page; the default `0.4` sends ~800 of ~2,000 weekly papers to
Haiku, dropping ~$0.25/week in API spend vs. unfiltered triage at near-zero
quality cost (the dropped papers' profile cosine similarities are too low
to plausibly score >5 anyway).

---

## 4. Settings

This project is meant to be cloned by anyone. The API key, email config, and
other operator preferences are stored in the database and edited via the
webapp.

### Settings table (in SQLite)

```sql
CREATE TABLE settings (
  key         TEXT PRIMARY KEY,
  value       TEXT,                -- nullable; empty string clears
  is_secret   INTEGER DEFAULT 0,   -- 1 = redact in API responses
  updated_at  TEXT NOT NULL
);
```

### Settings keys

| Key | Secret | Default | Purpose |
|-----|--------|---------|---------|
| `anthropic_api_key` | yes | — | For all LLM calls |
| `smtp_host` | no | — | e.g. `smtp.gmail.com` |
| `smtp_port` | no | `587` | |
| `smtp_username` | no | — | usually the same as `notify_email` |
| `smtp_password` | yes | — | App Password for Gmail; not the account password |
| `smtp_use_tls` | no | `true` | |
| `notify_email` | no | — | where to send the digest |
| `notify_enabled` | no | `false` | master switch |
| `digest_top_n` | no | `80` | how many papers to fully summarize |
| `triage_model` | no | `claude-haiku-4-5-20251001` | |
| `summarize_model` | no | `claude-sonnet-4-6` | |
| `triage_prefilter_enabled` | no | `true` | master switch for the local prefilter |
| `triage_prefilter_keep_fraction` | no | `0.4` | top fraction of papers (by cosine sim to profile) sent to Haiku |
| `profile_embedding_hash` | no | — | **pipeline-managed, not user-editable.** SHA-256 of `profile.md` content used to invalidate the cached embedding |
| `profile_embedding_b64` | no | — | **pipeline-managed, not user-editable.** Base64-encoded float32 profile embedding |

### Resolution order

When any module needs a setting:

1. Read from `settings` table.
2. If absent or empty, fall back to environment variable of the same name in
   uppercase (e.g., `ANTHROPIC_API_KEY`).
3. If still absent: for required settings, fail loudly with a clear error
   pointing the user to `/settings`. For optional settings, use the documented
   default above.

This means `.env` is for headless bootstrap (servers, CI). The **normal flow
is**: clone repo → `docker compose up` → visit
`http://localhost:8080/settings` → paste API key → save → run weekly digest.

### Settings page UI

A simple form with sections:

- **Anthropic** — API key (password input, masked when saved).
- **Email notification** — SMTP host/port/username/password (password masked),
  notify email, enabled toggle.
- **Digest behavior** — top-N papers, model overrides, prefilter enabled,
  prefilter keep-fraction (with help text: *"Higher = more thorough but
  costlier triage. Lower = cheaper but may miss off-vocabulary matches.
  Default 0.4 sends ~800/2,000 weekly papers to Haiku."*).
- **Status** — last successful run, current cost-this-week estimate, DB size.

Use HTMX for save (POST `/settings` returns the form section with a "saved"
indicator). No JavaScript framework. Tailwind via CDN for styling.

The `profile_embedding_hash` and `profile_embedding_b64` settings are NOT
shown on the page. They are pipeline-managed cache entries.

### Security on the settings page

- The webapp binds to `127.0.0.1` only. There is no auth on the webapp — the
  security boundary is "you have access to this user's machine."
- Secret fields are stored as plaintext in SQLite (the host machine's
  filesystem is the security boundary). Add a clear note in the UI:
  *"Settings are stored in `data/digest.db` on this machine. Do not share
  this file or commit it to version control."*
- API responses (`Accept: application/json`) redact secret fields as `"***"`.

---

## 5. Database schema

```sql
-- Settings (see §4)
CREATE TABLE settings (
  key TEXT PRIMARY KEY,
  value TEXT,
  is_secret INTEGER DEFAULT 0,
  updated_at TEXT NOT NULL
);

-- One row per paper, ever
CREATE TABLE papers (
  arxiv_id          TEXT PRIMARY KEY,
  title             TEXT NOT NULL,
  authors           TEXT NOT NULL,          -- JSON array
  abstract          TEXT NOT NULL,
  primary_category  TEXT NOT NULL,
  categories        TEXT NOT NULL,          -- JSON array
  published_at      TEXT NOT NULL,
  updated_at        TEXT,
  fetched_at        TEXT NOT NULL,
  url_abs           TEXT NOT NULL,
  url_pdf           TEXT NOT NULL,
  embedding         BLOB,                   -- 384 float32, all-MiniLM-L6-v2
  embedding_model   TEXT,
  prefilter_score   REAL,                   -- cosine sim to profile embedding (0.0–1.0)
  relevance_score   REAL,                   -- 0.0–10.0 from Haiku; 0.0 if dropped at prefilter
  relevance_reason  TEXT,
  triaged_at        TEXT,
  summarized        INTEGER DEFAULT 0,
  upvoted           INTEGER DEFAULT 0
);
CREATE INDEX idx_papers_published  ON papers(published_at);
CREATE INDEX idx_papers_relevance  ON papers(relevance_score DESC);
CREATE INDEX idx_papers_prefilter  ON papers(prefilter_score DESC);
CREATE INDEX idx_papers_category   ON papers(primary_category);

-- FTS5 virtual table for full-text search
CREATE VIRTUAL TABLE papers_fts USING fts5(
  arxiv_id UNINDEXED, title, abstract,
  content='papers', content_rowid='rowid'
);

CREATE TABLE summaries (
  arxiv_id            TEXT PRIMARY KEY REFERENCES papers(arxiv_id),
  problem             TEXT NOT NULL,
  method              TEXT NOT NULL,
  key_result          TEXT NOT NULL,
  why_it_matters      TEXT NOT NULL,
  cross_domain_hooks  TEXT NOT NULL,        -- JSON
  novelty_signal      TEXT NOT NULL,        -- incremental | notable | breakthrough_claim
  tags                TEXT NOT NULL,        -- JSON array
  generated_at        TEXT NOT NULL,
  model               TEXT NOT NULL
);

CREATE TABLE clusters (
  week         TEXT NOT NULL,
  cluster_id   INTEGER NOT NULL,
  label        TEXT NOT NULL,
  description  TEXT NOT NULL,
  subtopics    TEXT NOT NULL,               -- JSON array
  paper_count  INTEGER NOT NULL,
  PRIMARY KEY (week, cluster_id)
);

CREATE TABLE paper_clusters (
  arxiv_id    TEXT NOT NULL REFERENCES papers(arxiv_id),
  week        TEXT NOT NULL,
  cluster_id  INTEGER NOT NULL,
  PRIMARY KEY (arxiv_id, week),
  FOREIGN KEY (week, cluster_id) REFERENCES clusters(week, cluster_id)
);

CREATE TABLE cluster_trends (
  week           TEXT NOT NULL,
  cluster_id     INTEGER NOT NULL,
  delta_vs_w1    REAL,
  delta_vs_w4    REAL,
  delta_vs_w12   REAL,
  delta_vs_w52   REAL,
  is_new         INTEGER DEFAULT 0,
  velocity_class TEXT,                      -- accelerating | growing | flat | shrinking
  PRIMARY KEY (week, cluster_id)
);

CREATE TABLE digests (
  digest_id     TEXT PRIMARY KEY,           -- '2026-W20', '2026-05', '2026'
  kind          TEXT NOT NULL,              -- weekly | monthly | yearly
  generated_at  TEXT NOT NULL,
  html_path     TEXT NOT NULL,
  paper_count   INTEGER NOT NULL,
  trend_narrative TEXT,                     -- JSON
  sent_at       TEXT
);

CREATE TABLE feedback (
  arxiv_id   TEXT NOT NULL REFERENCES papers(arxiv_id),
  signal     TEXT NOT NULL,                 -- upvote | downvote | bookmark
  notes      TEXT,
  created_at TEXT NOT NULL,
  PRIMARY KEY (arxiv_id, signal)
);

CREATE TABLE pipeline_runs (
  run_id              TEXT PRIMARY KEY,
  kind                TEXT NOT NULL,        -- weekly | monthly | yearly | backfill | reprocess
  started_at          TEXT NOT NULL,
  finished_at         TEXT,
  status              TEXT NOT NULL,        -- running | success | failure
  papers_ingested     INTEGER,
  papers_prefiltered  INTEGER,              -- count dropped at prefilter step
  papers_triaged      INTEGER,              -- count sent to Haiku
  papers_summarized   INTEGER,
  cost_usd            REAL,
  error               TEXT
);
```

---

## 6. Build order

Implement these in order. **After each module, commit with a clear message
and stop. Wait for the operator to confirm before continuing.**

1. `pipeline/src/db.py` — schema creation, migrations, connection helpers,
   settings getter/setter.
2. `pipeline/src/settings.py` — typed settings accessors with env-var
   fallback (per §4 resolution order).
3. `pipeline/src/ingest.py` — arXiv API client with 3-second rate limit,
   dedup by `arxiv_id`.
4. `pipeline/src/embed.py` — sentence-transformers wrapper, BLOB storage
   helpers.
5. `pipeline/src/llm.py` — Anthropic client, Batch API helpers, prompt
   caching, JSON Schema validation, retry-once-then-skip.
6. `pipeline/src/triage.py` — **includes the local prefilter step**:
   - **(a)** compute/cache profile embedding. Hash `config/profile.md` with
     SHA-256; if the stored `profile_embedding_hash` matches, decode and use
     the cached `profile_embedding_b64`. Otherwise embed afresh with MiniLM
     and write both back to `settings`.
   - **(b)** score every paper by cosine similarity to the profile embedding
     → `papers.prefilter_score`.
   - **(c)** sort descending; keep top `triage_prefilter_keep_fraction`
     (default `0.4`). For the rest, write
     `relevance_score = 0.0`,
     `relevance_reason = "below profile-similarity prefilter"`,
     `triaged_at = now`. They stay in the DB and remain searchable; they
     just do not consume a Haiku call.
   - **(d)** Haiku 4.5 Batch on the survivors using `prompts/triage.md` +
     `prompts/triage.schema.json`.
   - If `triage_prefilter_enabled` is `false`, steps (a)–(c) are skipped and
     Haiku runs on all papers.
7. `pipeline/src/cluster.py` — HDBSCAN + Sonnet cluster labels using
   `prompts/cluster_label.md` + schema.
8. `pipeline/src/summarize.py` — Sonnet summaries with cross-domain candidate
   selection (FAISS over prior 90 days, different `primary_category`,
   sim > 0.65, top 3); uses `prompts/summarize.md` + schema.
9. `pipeline/src/trend.py` — SQL rollups for W-1/W-4/W-12/W-52 deltas + trend
   narrative call using `prompts/trend_narrative.md` + schema.
10. `pipeline/src/render.py` — Jinja2 HTML generation for
    weekly/monthly/yearly templates; inline CSS, server-rendered SVG charts.
11. `pipeline/src/notify.py` — webhook POST to webapp + optional SMTP send.
12. `pipeline/src/pipeline.py` — CLI orchestration:
    `weekly | monthly | yearly | backfill | reprocess | health`.
13. `pipeline/Dockerfile` and `pipeline/pyproject.toml`.
14. `webapp/src/main.py` plus route modules and templates — Settings page is
    part of this step.
15. `webapp/Dockerfile` and `webapp/pyproject.toml`.
16. `scheduler/Dockerfile` and `scheduler/crontab` — Alpine + cron, triggers
    `docker exec arxiv-digest-pipeline python -m src.pipeline weekly` every
    Monday 07:00.

---

## 7. Testing

- Each module gets `pipeline/tests/test_<module>.py` (or
  `webapp/tests/...`).
- Cache 5–10 real arXiv API responses as JSON fixtures under
  `pipeline/tests/fixtures/` so tests don't hit the live API.
- LLM tests use record-and-replay via `LLM_MODE=replay` env var; cassettes
  under `pipeline/tests/cassettes/`. Default to replay; live mode is opt-in.
- The CI workflow (`.github/workflows/ci.yml`) runs `ruff`, `mypy --strict`,
  and `pytest` on push. No secrets needed since replay mode is the default.

---

## 8. Code style

- `from __future__ import annotations` at the top of every Python file.
- Type hints required. `mypy --strict` must pass.
- Functions over classes unless state genuinely accumulates.
- Pure functions where possible; side effects isolated to clearly named
  modules (`db`, `ingest`, `notify`).
- Logging via `structlog` with JSON output. One event per significant step.
  Never log abstracts, settings values marked secret, or API key fragments.

---

## 9. Cross-domain hooks (the differentiating feature)

When summarizing each top-N paper, before the Sonnet summarize call:

1. Take the paper's embedding.
2. Build an in-memory FAISS index of all papers published in the prior 90
   days.
3. Search for the 20 nearest neighbors with
   `primary_category != this.primary_category`.
4. Filter to top 3 by cosine similarity, requiring similarity > 0.65.
5. Pass these 3 candidates (title + abstract + primary_category + similarity)
   into the Sonnet summarize prompt.
6. Sonnet returns `cross_domain_hooks` as a JSON array of
   `{arxiv_id, connection, strength}` — only including hooks it finds
   genuinely substantive. The prompt explicitly instructs rejection of weak
   parallels. Empty array is correct and common.

Never fabricate connections.

---

## 10. Acceptance criteria (v1)

The v1 build is done when all of the following are demonstrably true:

- `docker compose up` from a clean checkout (with only `WEBHOOK_SECRET` set)
  succeeds and brings the webapp healthy on `http://localhost:8080`.
- A new operator can complete setup end-to-end **via the Settings UI alone**
  — pasting an Anthropic API key, optional SMTP creds, saving, and
  triggering a weekly run without editing any file outside
  `config/profile.md` and `config/categories.yaml`.
- `python -m src.pipeline backfill` respects the 3-second arXiv rate limit
  (verifiable from logs).
- A `weekly` run on a fresh DB produces:
  - A populated `papers` table with embeddings and `prefilter_score`.
  - Roughly `(1 - keep_fraction) × ingested` papers with
    `relevance_reason = "below profile-similarity prefilter"` and
    `relevance_score = 0.0`.
  - A populated `summaries` table for ~`digest_top_n` papers.
  - Populated `clusters` and `cluster_trends` rows for the current week.
  - A `digests` row pointing at a rendered HTML file under `data/digests/`.
  - A successful webhook POST to `/webhook/digest-ready` with the correct
    HMAC signature derived from `WEBHOOK_SECRET`.
- Cross-domain hooks appear only when genuinely substantive — empty arrays
  are observed across a non-trivial fraction of summaries.
- Every LLM call is validated against its JSON Schema; one invalid response
  triggers exactly one retry, and a second failure logs + skips.
- Monthly and yearly aggregations build from existing weekly data without
  re-fetching from arXiv or re-calling Claude.
- Total Anthropic spend for a representative week (~2,000 ingested,
  ~80 summarized) is **under $2.00** as estimated from
  `pipeline_runs.cost_usd`.
- `ruff check` reports no errors, `mypy --strict` reports no errors, and
  `pytest` passes on both `pipeline/` and `webapp/`.

---

## 11. Out of scope for v1

These are deliberately excluded from v1 and will not be implemented in this
build. They are recorded here so future work can pick them up without
re-deriving the boundary.

- Multi-user / multi-profile support. One operator per deployment.
- Webapp authentication. Security boundary is "you have access to this
  machine's localhost."
- PDF parsing of arXiv papers. We only use the abstract.
- Ingestion from Semantic Scholar, ConnectedPapers, OpenReview, or industry
  blogs (Anthropic, OpenAI, DeepMind, etc.). arXiv only.
- Upvote-driven profile retraining. The `feedback` table and `upvoted`
  column exist; nothing currently reads them. v2 may use them to re-weight
  scoring.
- Trend visualizations beyond server-rendered SVG. No D3, Plotly, or
  client-side charts.
- Mobile-specific layouts. Desktop browser only.
- Slack / Discord / Telegram notifications. SMTP only.
