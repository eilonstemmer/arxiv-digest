# Operator relevance profile

> **This is a TEMPLATE.** Edit every section below before running the
> pipeline. The triage and summarize prompts feed this file directly to
> Claude. Be specific. Vague profiles produce vague scoring. Concrete
> profiles produce sharp scoring.
>
> When you edit this file, the next pipeline run will detect the change
> via SHA-256 hash and re-embed the profile automatically.

---

## About me

> Replace this paragraph with a 3–5 sentence self-description: your role,
> your domain, what you build or research, and what you're trying to learn
> from the weekly digest.
>
> Example level of specificity:
> *"I'm a robotics engineer working on long-horizon manipulation in
> cluttered environments. I read arXiv to track planning, perception, and
> memory advances that might transfer to physical agents. I care about
> real-world deployment over benchmark results."*

Replace this with your own about-me paragraph.

---

## Primary interests

> Replace this section with 4–8 themed groups, each containing 4–8 specific
> bullets. **Specificity matters.** Bad bullet: *"machine learning"*. Good
> bullet: *"agent memory systems for long-horizon tasks"*. The triage model
> uses these bullets verbatim to match papers.

### Theme 1 — Replace this heading (e.g. "Agentic systems")

- Replace this bullet with a specific interest, e.g.
  *agent memory systems for long-horizon tasks.*
- Replace this bullet with another specific interest, e.g.
  *tool-use protocols and verification of tool outputs.*
- Add 2–6 more bullets at this level of specificity.

### Theme 2 — Replace this heading (e.g. "Embodied learning")

- Replace this bullet with a specific interest, e.g.
  *sim-to-real transfer for manipulation with sparse rewards.*
- Replace this bullet with another specific interest, e.g.
  *generalization across object geometries in robotic grasping.*
- Add 2–6 more bullets at this level of specificity.

### Theme 3 — Replace this heading

- Replace this with your own interest.
- Add as many themes (and bullets per theme) as you need.

---

## Secondary interests

> Topics you'd like to see but at a lower bar. Short list — 5–10 items max.
> The triage model uses these as moderate-signal matches (typically scored
> 5.0–7.5).

- Replace this with a topic you'd skim if it came up.
- Replace this with another.
- Add or remove as you like.

---

## Anti-interests

> Topics or paper genres to filter aggressively. The triage model caps any
> paper matching an anti-interest at score 4.0 and flags the match in
> `anti_interest_flags`.

- Survey papers (unless they survey a specific primary interest).
- Position papers without empirical results.
- Pure theory papers with no clear application path.
- Replace these or add your own anti-interests.

---

## Cross-domain bridges to surface

> 2–6 example bridges describing the kinds of connections you find
> valuable. The summarize prompt uses this section to decide what counts
> as a *substantive* cross-domain hook (versus a weak topical parallel).

- **Example bridge 1:** Replace with your own — e.g.
  *biological neural coding mechanisms that suggest new architectures for
  artificial agents.*
- **Example bridge 2:** Replace with your own — e.g.
  *control-theoretic guarantees being imported into RL policy training.*
- Add more bridges as needed. Vague entries (*"interdisciplinary
  research"*) produce vague hooks. Concrete entries produce concrete
  hooks.

---

## Reading style preferences

> Short list — how you want summaries to read. The summarize model honors
> this.

- Prefer concrete numerical results over qualitative claims.
- Skip background framing; assume I know the field.
- Flag breakthrough claims explicitly so I can decide whether to read the
  paper directly.
- Add or remove as you like.

---

## Calibration (score band reference)

> The triage model uses these bands. You generally don't need to edit
> them, but you may tune the thresholds if your weekly digest is too
> noisy or too sparse.

- **9.0–10.0** — Must-read. Directly advances a primary interest with
  concrete contribution.
- **7.0–8.9** — Worth a full summary. Solid primary match or strong
  secondary match with novelty.
- **5.0–6.9** — Title-only. Tangentially relevant; nice to know it exists.
- **0.0–4.9** — Noise. Off-topic or anti-interest match.

Caps:

- Any anti-interest match: cap at 4.0.
- Survey / position paper (unless explicitly in primary interests): cap
  at 5.0.
