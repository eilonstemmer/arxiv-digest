# arxiv-digest

A self-hosted, open-source weekly arXiv research digest, tailored to your
interests.

Pulls fresh papers across configurable arXiv categories, locally pre-filters
them against your profile, scores the survivors with Claude Haiku, summarizes
the top ~80 with Claude Sonnet, surfaces genuine cross-domain bridges, and
serves the result at `http://localhost:8080`. Optionally emails an HTML
digest.

**Cost:** under $2/week on the Anthropic API for ~2,000 papers ingested and
~80 summarized weekly. The local prefilter is the main lever.

> See [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md) for the full design. This README
> is the getting-started guide; the brief is the source of truth.

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/<your-username>/arxiv-digest.git
cd arxiv-digest

# 2. Generate the one mandatory env var
echo "WEBHOOK_SECRET=$(openssl rand -hex 32)" > .env

# 3. Bring up the stack
docker compose up -d

# 4. Open the Settings tab and paste your Anthropic API key.
#    Visit http://localhost:8080/settings in your browser.

# 5. Edit your profile to match your interests
#    (the file is mounted into the running webapp — no rebuild needed)
$EDITOR config/profile.md

# 6. Run a one-time backfill of recent papers, then your first weekly digest
docker compose run --rm pipeline python -m src.pipeline backfill
docker compose run --rm pipeline python -m src.pipeline weekly
```

After step 6, visit `http://localhost:8080` to read your first digest.

---

## What you get

- **Weekly digest** at `http://localhost:8080` (also available via SMTP if
  configured).
- **Browsable archive** of all past digests — weekly, monthly, yearly
  rollups.
- **Single-paper view** at `/papers/{arxiv_id}` with the full structured
  summary and any cross-domain bridges.
- **Search** at `/search?q=...` — combined full-text (SQLite FTS5) and
  semantic (cosine over MiniLM embeddings).
- **Trends page** showing which clusters are accelerating, cooling, or
  emerging week over week.
- **Settings page** for API key, SMTP, model overrides, and the prefilter
  keep-fraction knob.

---

## Architecture (high level)

Three containers orchestrated by `docker-compose.yml`:

- **`webapp`** — FastAPI + Jinja2 + HTMX, bound to `127.0.0.1:8080`
  (localhost only).
- **`pipeline`** — batch worker. Gated behind `profiles: ["pipeline"]` so
  it does not auto-start with `docker compose up`. Trigger via
  `docker compose run --rm pipeline python -m src.pipeline weekly`.
- **`scheduler`** — Alpine + cron. Triggers the pipeline weekly
  (Monday 07:00).

State lives in `./data/digest.db` (SQLite, with FTS5 and BLOB embeddings).

Pipeline stages:

1. **ingest** — arXiv API client with a 3-second token bucket; dedup by
   `arxiv_id`.
2. **embed** — `sentence-transformers/all-MiniLM-L6-v2` on CPU; 384-dim
   float32 BLOBs.
3. **prefilter** — embed `config/profile.md` once (cached by SHA-256),
   cosine-sim every paper, keep the top `triage_prefilter_keep_fraction`
   (default `0.4`). Dropped papers stay in the DB with
   `relevance_score = 0.0` and remain searchable.
4. **triage** — Claude Haiku 4.5 Batch on the survivors. Returns 0.0–10.0
   relevance scores against your profile.
5. **cluster** — HDBSCAN over embeddings; Sonnet labels each cluster.
6. **summarize** — Claude Sonnet 4.6 Batch on the top `digest_top_n`
   (default 80). FAISS finds cross-domain candidate papers from the prior
   90 days; the prompt rejects weak parallels.
7. **trend** — SQL rollups for W-1 / W-4 / W-12 / W-52 deltas; one Sonnet
   call generates the weekly narrative.
8. **render** — Jinja2 → self-contained HTML with inline CSS and SVG charts.
9. **notify** — signed webhook POST to the webapp; optional SMTP send.

---

## Configuration precedence

For every operator-tunable knob, values are resolved in this order:

1. **Database `settings` table** (edited via the Settings tab at
   `/settings`). This is the canonical store and takes precedence.
2. **Environment variables** in `.env` (for headless deployments).
3. **Documented defaults** in `PROJECT_BRIEF.md` §4.

Required values with no default cause a startup error pointing at
`/settings`.

Profile (`config/profile.md`) and category list (`config/categories.yaml`)
are mounted as read-only volumes — edit on the host, the containers see the
change immediately.

---

## Cost breakdown

For a representative week (~2,000 papers ingested, ~80 summarized):

| Stage | Model | Calls | Approx cost |
|-------|-------|-------|-------------|
| Embed | MiniLM-L6-v2 (local CPU) | 2,000 | $0.00 |
| Prefilter | (local CPU cosine sim) | 2,000 | $0.00 |
| Triage | Haiku 4.5, Batch | ~800 (40% of 2,000) | ~$0.15 |
| Cluster labels | Sonnet 4.6, Direct | ~10 | ~$0.05 |
| Summaries | Sonnet 4.6, Batch | ~80 | ~$1.20 |
| Trend narrative | Sonnet 4.6, Direct | 1 | ~$0.05 |
| **Total** | | | **~$1.45/week** |

The tuning lever is `triage_prefilter_keep_fraction` (default `0.4`):

- Raise it (e.g. `0.6`) for more thorough triage at higher cost.
- Lower it (e.g. `0.25`) to cut cost further, at the risk of missing
  off-vocabulary primary matches.

---

## Security notes

- The webapp binds to `127.0.0.1` only. There is no authentication. The
  security boundary is "you have access to this user's machine."
- The Anthropic API key, SMTP password, and other secrets are stored as
  **plaintext** in `data/digest.db`. **Do not commit this file. Do not
  share it.** It's gitignored by default.
- `.gitignore` excludes `.env` and `data/`. **Verify with `git status`
  before every push** — git history is forever. If you ever accidentally
  commit `.env`, rotate the Anthropic API key immediately.
- The scheduler container mounts the Docker socket so it can run
  `docker exec` against the pipeline container on cron. This grants the
  scheduler container the ability to execute commands inside any
  container on this host. Keep the scheduler image minimal and trusted.
- The internal webhook (`/webhook/digest-ready`) is protected by an HMAC
  signature derived from `WEBHOOK_SECRET`. Set this before first launch.

---

## Contributing

This project enforces a strict baseline:

- **Python 3.12** with type hints on every public function.
- **`ruff check`** must pass with zero errors.
- **`mypy --strict`** must pass with zero errors.
- **`pytest`** must pass under `LLM_MODE=replay` (the default in CI).
- Every LLM call must validate against its JSON Schema in `prompts/`.
- No secrets in logs.

Before opening a PR, run:

```bash
cd pipeline && uv run ruff check && uv run mypy --strict src && uv run pytest
cd ../webapp && uv run ruff check && uv run mypy --strict src && uv run pytest
```

CI runs the same on every push to `main` and every PR.

---

## License

MIT. See [`LICENSE`](LICENSE).
