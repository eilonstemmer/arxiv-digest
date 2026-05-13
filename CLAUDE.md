# Claude instructions for this repo

You are working on **arxiv-digest**, a self-hosted weekly arXiv digest stack.

1. **Read [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md) first.** It is the source of
   truth. If anything in this file, the README, or a code comment disagrees
   with the brief, the brief wins.

2. **Follow the build order in §6 of the brief.** Implement one module,
   commit with a clear message, and stop. Wait for the operator's
   confirmation before starting the next module.

3. **The decisions in the brief are final.** Do not propose architectural
   changes unless the operator explicitly asks. If you spot a real problem,
   raise it briefly and wait — don't unilaterally redesign.

4. **JSON schemas in `prompts/` are non-negotiable.** Every LLM call
   validates against the matching `.schema.json`. On invalid output: retry
   once with the parser error appended; on second failure, skip the item
   and log to `pipeline_runs.error`. Never accept unstructured output.

5. **Batch API + prompt caching are required** for `triage` and `summarize`.
   Direct API is fine for `cluster_label` (one call per cluster per week)
   and `trend_narrative` (one call per digest).

6. **No secrets in logs.** Log IDs, counts, durations, status codes. Never
   log abstracts, secret settings values, or API key fragments.

7. **Code style:**
   - `from __future__ import annotations` at the top of every `.py` file.
   - Full type hints; `mypy --strict` must pass.
   - `ruff check` must pass.
   - `structlog` with JSON output for all logging.
   - `uv` for dependency management.

8. **Public-repo hygiene.** `.env` and `data/` are gitignored. Never commit
   a real API key, SMTP password, or operator-specific profile content. The
   `config/profile.md` in the repo is a template with placeholders only.

9. **Commit-and-pause after each module in §6.** Use commit messages like
   `feat(pipeline): add ingest module with arXiv rate limiting`. Then stop.
