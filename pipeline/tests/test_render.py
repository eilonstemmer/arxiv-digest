"""Tests for src.render: HTML generation, DB persistence, idempotency.

Fixture DB setup inserts the minimum rows needed for each test scenario.
All tests use tmp_path for output_dir to keep the filesystem clean.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest

from src import db, render

# ── Helpers ───────────────────────────────────────────────────────────────────

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "src" / "templates"


class _TagCollector(HTMLParser):
    """Minimal HTML parser that collects tag names and text."""

    def __init__(self) -> None:
        super().__init__()
        self._tags: list[str] = []
        self._texts: list[str] = []
        self._buf = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._tags.append(tag.lower())
        # Flush text buffer
        text = self._buf.strip()
        if text:
            self._texts.append(text)
        self._buf = ""

    def handle_endtag(self, tag: str) -> None:
        text = self._buf.strip()
        if text:
            self._texts.append(text)
        self._buf = ""

    def handle_data(self, data: str) -> None:
        self._buf += data

    def handle_entityref(self, name: str) -> None:
        self._buf += " "

    def handle_charref(self, name: str) -> None:
        self._buf += " "

    @property
    def tags(self) -> list[str]:
        return self._tags

    @property
    def all_text(self) -> str:
        return " ".join(self._texts)


def _parse_html(html: str) -> _TagCollector:
    """Parse html and return a collector with tags and concatenated text."""
    collector = _TagCollector()
    collector.feed(html)
    return collector


def _assert_well_formed(html: str) -> None:
    """Verify the HTML parses without errors (using html.parser)."""
    collector = _TagCollector()
    collector.feed(html)
    # If feed() completes without raising, we consider it parseable.


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    c = db.connect(tmp_path / "test.db")
    db.init_schema(c)
    yield c
    c.close()


def _insert_paper(
    conn: sqlite3.Connection,
    arxiv_id: str,
    *,
    title: str = "A Title",
    authors: list[str] | None = None,
    abstract: str = "Abstract text.",
    primary_category: str = "cs.AI",
    published_at: str = "2026-05-01T00:00:00Z",
    relevance_score: float = 7.5,
    url_abs: str | None = None,
    url_pdf: str | None = None,
) -> None:
    if authors is None:
        authors = ["Author One", "Author Two"]
    conn.execute(
        """
        INSERT OR IGNORE INTO papers(
          arxiv_id, title, authors, abstract, primary_category, categories,
          published_at, fetched_at, url_abs, url_pdf, relevance_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, '2026-05-01T00:00:00Z', ?, ?, ?)
        """,
        (
            arxiv_id,
            title,
            json.dumps(authors),
            abstract,
            primary_category,
            json.dumps([primary_category]),
            published_at,
            url_abs or f"https://arxiv.org/abs/{arxiv_id}",
            url_pdf or f"https://arxiv.org/pdf/{arxiv_id}",
            relevance_score,
        ),
    )


def _insert_cluster(
    conn: sqlite3.Connection,
    week: str,
    cluster_id: int,
    *,
    label: str = "Test Cluster",
    description: str = "A test cluster description.",
    paper_count: int = 2,
    subtopics: list[str] | None = None,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO clusters(week, cluster_id, label, description, subtopics, paper_count)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            week,
            cluster_id,
            label,
            description,
            json.dumps(subtopics or []),
            paper_count,
        ),
    )


def _insert_paper_cluster(
    conn: sqlite3.Connection,
    arxiv_id: str,
    week: str,
    cluster_id: int,
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO paper_clusters(arxiv_id, week, cluster_id)
        VALUES (?, ?, ?)
        """,
        (arxiv_id, week, cluster_id),
    )


def _insert_summary(
    conn: sqlite3.Connection,
    arxiv_id: str,
    *,
    why_it_matters: str = "This matters because of X.",
    novelty_signal: str = "notable",
    tags: list[str] | None = None,
    cross_domain_hooks: list[dict[str, Any]] | None = None,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO summaries(
          arxiv_id, problem, method, key_result, why_it_matters,
          cross_domain_hooks, novelty_signal, tags, generated_at, model
        ) VALUES (?, 'problem', 'method', 'key result', ?, ?, ?, ?, '2026-01-01', 'test')
        """,
        (
            arxiv_id,
            why_it_matters,
            json.dumps(cross_domain_hooks or []),
            novelty_signal,
            json.dumps(tags or ["tag1", "tag2"]),
        ),
    )


def _insert_cluster_trend(
    conn: sqlite3.Connection,
    week: str,
    cluster_id: int,
    *,
    delta_vs_w4: float | None = None,
    velocity_class: str | None = "growing",
    is_new: int = 0,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO cluster_trends(
          week, cluster_id, delta_vs_w1, delta_vs_w4, delta_vs_w12, delta_vs_w52,
          is_new, velocity_class
        ) VALUES (?, ?, NULL, ?, NULL, NULL, ?, ?)
        """,
        (week, cluster_id, delta_vs_w4, is_new, velocity_class),
    )


# ── Weekly render tests ───────────────────────────────────────────────────────


def test_weekly_render_creates_html_file(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """render_weekly writes an HTML file to output_dir."""
    week = "2026-W20"
    _insert_paper(conn, "2605.00001", title="Neural Scaling Laws Revisited")
    _insert_paper(conn, "2605.00002", title="Sparse Attention Mechanisms")
    _insert_cluster(conn, week, 0, label="LLM Scaling", paper_count=2)
    _insert_paper_cluster(conn, "2605.00001", week, 0)
    _insert_paper_cluster(conn, "2605.00002", week, 0)

    result = render.render_weekly(
        conn,
        week=week,
        output_dir=tmp_path,
        templates_dir=TEMPLATES_DIR,
    )

    assert result.html_path.exists()
    assert result.digest_id == week
    assert result.paper_count == 2


def test_weekly_render_inserts_digests_row(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """render_weekly upserts a row in the digests table."""
    week = "2026-W20"
    _insert_paper(conn, "2605.00001")
    _insert_cluster(conn, week, 0, paper_count=1)
    _insert_paper_cluster(conn, "2605.00001", week, 0)

    render.render_weekly(
        conn,
        week=week,
        output_dir=tmp_path,
        templates_dir=TEMPLATES_DIR,
    )

    row = conn.execute(
        "SELECT * FROM digests WHERE digest_id = ?", (week,)
    ).fetchone()
    assert row is not None
    assert str(row["kind"]) == "weekly"
    assert int(row["paper_count"]) == 1
    assert row["sent_at"] is None


def test_weekly_render_paper_count_correct(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """paper_count in RenderResult and digests row equals actual papers in clusters."""
    week = "2026-W21"
    for i in range(5):
        _insert_paper(conn, f"2605.0100{i}", title=f"Paper {i}")
    _insert_cluster(conn, week, 0, label="Cluster A", paper_count=3)
    _insert_cluster(conn, week, 1, label="Cluster B", paper_count=2)
    for i in range(3):
        _insert_paper_cluster(conn, f"2605.0100{i}", week, 0)
    for i in range(3, 5):
        _insert_paper_cluster(conn, f"2605.0100{i}", week, 1)

    result = render.render_weekly(
        conn,
        week=week,
        output_dir=tmp_path,
        templates_dir=TEMPLATES_DIR,
    )
    assert result.paper_count == 5


def test_weekly_html_contains_cluster_label(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """HTML output includes the cluster label."""
    week = "2026-W20"
    _insert_paper(conn, "2605.00001")
    _insert_cluster(conn, week, 0, label="Quantum Error Correction")
    _insert_paper_cluster(conn, "2605.00001", week, 0)

    result = render.render_weekly(
        conn,
        week=week,
        output_dir=tmp_path,
        templates_dir=TEMPLATES_DIR,
    )
    html = result.html_path.read_text(encoding="utf-8")
    assert "Quantum Error Correction" in html


def test_weekly_html_contains_paper_title(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """HTML output includes each paper title."""
    week = "2026-W20"
    _insert_paper(conn, "2605.00001", title="Attention Is All You Need Redux")
    _insert_cluster(conn, week, 0, label="Transformers")
    _insert_paper_cluster(conn, "2605.00001", week, 0)

    result = render.render_weekly(
        conn,
        week=week,
        output_dir=tmp_path,
        templates_dir=TEMPLATES_DIR,
    )
    html = result.html_path.read_text(encoding="utf-8")
    assert "Attention Is All You Need Redux" in html


def test_weekly_html_contains_why_it_matters(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """HTML output includes why_it_matters text from summaries."""
    week = "2026-W20"
    _insert_paper(conn, "2605.00001")
    _insert_cluster(conn, week, 0)
    _insert_paper_cluster(conn, "2605.00001", week, 0)
    _insert_summary(
        conn,
        "2605.00001",
        why_it_matters="This is a landmark result for humanity.",
    )

    result = render.render_weekly(
        conn,
        week=week,
        output_dir=tmp_path,
        templates_dir=TEMPLATES_DIR,
    )
    html = result.html_path.read_text(encoding="utf-8")
    assert "This is a landmark result for humanity." in html


def test_weekly_html_is_well_formed(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """HTML output is parseable by Python's html.parser."""
    week = "2026-W20"
    _insert_paper(conn, "2605.00001", title="Test Paper <Special> &amp; Chars")
    _insert_cluster(conn, week, 0)
    _insert_paper_cluster(conn, "2605.00001", week, 0)

    result = render.render_weekly(
        conn,
        week=week,
        output_dir=tmp_path,
        templates_dir=TEMPLATES_DIR,
    )
    html = result.html_path.read_text(encoding="utf-8")
    _assert_well_formed(html)


def test_weekly_html_contains_svg_chart(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """HTML output includes an SVG bar chart with <svg> and <rect> elements."""
    week = "2026-W20"
    _insert_paper(conn, "2605.00001")
    _insert_cluster(conn, week, 0, label="ML Theory", paper_count=1)
    _insert_paper_cluster(conn, "2605.00001", week, 0)

    result = render.render_weekly(
        conn,
        week=week,
        output_dir=tmp_path,
        templates_dir=TEMPLATES_DIR,
    )
    html = result.html_path.read_text(encoding="utf-8")
    assert "<svg" in html
    assert "<rect" in html


def test_weekly_empty_week_produces_valid_page(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """A week with no clusters produces a valid HTML page with empty state."""
    week = "2026-W00"
    result = render.render_weekly(
        conn,
        week=week,
        output_dir=tmp_path,
        templates_dir=TEMPLATES_DIR,
    )
    assert result.html_path.exists()
    assert result.paper_count == 0
    html = result.html_path.read_text(encoding="utf-8")
    _assert_well_formed(html)
    # Some empty-state indicator should be present
    assert "no papers" in html.lower() or "empty" in html.lower()


def test_weekly_trend_narrative_headline_visible(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """When trend_narrative is provided with a 'headline' key, it appears in HTML."""
    week = "2026-W20"
    _insert_paper(conn, "2605.00001")
    _insert_cluster(conn, week, 0)
    _insert_paper_cluster(conn, "2605.00001", week, 0)

    narrative = {"headline": "Breakthroughs in protein folding dominate this week."}
    result = render.render_weekly(
        conn,
        week=week,
        output_dir=tmp_path,
        trend_narrative=narrative,
        templates_dir=TEMPLATES_DIR,
    )
    html = result.html_path.read_text(encoding="utf-8")
    assert "Breakthroughs in protein folding dominate this week." in html


def test_weekly_trend_narrative_persisted_in_digests(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """trend_narrative JSON is stored in the digests row."""
    week = "2026-W20"
    _insert_paper(conn, "2605.00001")
    _insert_cluster(conn, week, 0)
    _insert_paper_cluster(conn, "2605.00001", week, 0)

    narrative = {"headline": "Big week for RL."}
    render.render_weekly(
        conn,
        week=week,
        output_dir=tmp_path,
        trend_narrative=narrative,
        templates_dir=TEMPLATES_DIR,
    )

    row = conn.execute(
        "SELECT trend_narrative FROM digests WHERE digest_id = ?", (week,)
    ).fetchone()
    assert row is not None
    stored = json.loads(str(row["trend_narrative"]))
    assert stored["headline"] == "Big week for RL."


def test_weekly_rerender_overwrites_file_and_digest(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """Re-running render_weekly overwrites the HTML and updates the digests row."""
    week = "2026-W20"
    _insert_paper(conn, "2605.00001")
    _insert_cluster(conn, week, 0)
    _insert_paper_cluster(conn, "2605.00001", week, 0)

    render.render_weekly(
        conn,
        week=week,
        output_dir=tmp_path,
        templates_dir=TEMPLATES_DIR,
    )
    # Run again
    render.render_weekly(
        conn,
        week=week,
        output_dir=tmp_path,
        trend_narrative={"headline": "Updated narrative."},
        templates_dir=TEMPLATES_DIR,
    )

    count = conn.execute(
        "SELECT COUNT(*) AS c FROM digests WHERE digest_id = ?", (week,)
    ).fetchone()["c"]
    assert int(count) == 1  # no duplicate rows

    row = conn.execute(
        "SELECT trend_narrative FROM digests WHERE digest_id = ?", (week,)
    ).fetchone()
    stored = json.loads(str(row["trend_narrative"]))
    assert stored["headline"] == "Updated narrative."


def test_weekly_html_contains_velocity_badge(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """Velocity badge appears in the HTML when cluster_trends has a velocity_class."""
    week = "2026-W20"
    _insert_paper(conn, "2605.00001")
    _insert_cluster(conn, week, 0, label="Robotics Control")
    _insert_paper_cluster(conn, "2605.00001", week, 0)
    _insert_cluster_trend(conn, week, 0, velocity_class="accelerating", is_new=1)

    result = render.render_weekly(
        conn,
        week=week,
        output_dir=tmp_path,
        templates_dir=TEMPLATES_DIR,
    )
    html = result.html_path.read_text(encoding="utf-8")
    assert "accelerating" in html
    assert "New" in html


def test_weekly_html_contains_cross_domain_hooks(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """Cross-domain hooks appear in the HTML when summary has non-empty hooks."""
    week = "2026-W20"
    _insert_paper(conn, "2605.00001")
    _insert_cluster(conn, week, 0)
    _insert_paper_cluster(conn, "2605.00001", week, 0)
    _insert_summary(
        conn,
        "2605.00001",
        cross_domain_hooks=[
            {
                "arxiv_id": "2605.99999",
                "connection": "Applies RL to drug discovery pipeline.",
                "strength": "strong",
            }
        ],
    )

    result = render.render_weekly(
        conn,
        week=week,
        output_dir=tmp_path,
        templates_dir=TEMPLATES_DIR,
    )
    html = result.html_path.read_text(encoding="utf-8")
    assert "Applies RL to drug discovery pipeline." in html
    assert "strong" in html


def test_weekly_multi_cluster_ordering(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """Clusters are rendered in paper_count descending order."""
    week = "2026-W20"
    for i in range(6):
        _insert_paper(conn, f"2605.0200{i}")
    # Cluster 1: 4 papers, Cluster 0: 2 papers
    _insert_cluster(conn, week, 0, label="Small Cluster", paper_count=2)
    _insert_cluster(conn, week, 1, label="Big Cluster", paper_count=4)
    for i in range(2):
        _insert_paper_cluster(conn, f"2605.0200{i}", week, 0)
    for i in range(2, 6):
        _insert_paper_cluster(conn, f"2605.0200{i}", week, 1)

    result = render.render_weekly(
        conn,
        week=week,
        output_dir=tmp_path,
        templates_dir=TEMPLATES_DIR,
    )
    html = result.html_path.read_text(encoding="utf-8")
    big_pos = html.find("Big Cluster")
    small_pos = html.find("Small Cluster")
    assert big_pos < small_pos, "Bigger cluster should appear before smaller cluster"


# ── Monthly render tests ──────────────────────────────────────────────────────


def _setup_two_weeks(conn: sqlite3.Connection) -> None:
    """Insert papers and clusters across two weeks in 2026-05."""
    for i in range(3):
        _insert_paper(
            conn,
            f"2605.0300{i}",
            title=f"Week20 Paper {i}",
            published_at="2026-05-11T00:00:00Z",
        )
    _insert_cluster(conn, "2026-W20", 0, label="NLP Research", paper_count=3)
    for i in range(3):
        _insert_paper_cluster(conn, f"2605.0300{i}", "2026-W20", 0)

    for i in range(2):
        _insert_paper(
            conn,
            f"2605.0400{i}",
            title=f"Week21 Paper {i}",
            published_at="2026-05-18T00:00:00Z",
        )
    _insert_cluster(conn, "2026-W21", 0, label="Vision Models", paper_count=2)
    for i in range(2):
        _insert_paper_cluster(conn, f"2605.0400{i}", "2026-W21", 0)


def test_monthly_render_creates_html_file(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """render_monthly writes an HTML file to output_dir."""
    _setup_two_weeks(conn)
    result = render.render_monthly(
        conn,
        month="2026-05",
        output_dir=tmp_path,
        templates_dir=TEMPLATES_DIR,
    )
    assert result.html_path.exists()
    assert result.digest_id == "2026-05"


def test_monthly_render_aggregates_paper_count(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """render_monthly paper_count spans all weeks in the month."""
    _setup_two_weeks(conn)
    result = render.render_monthly(
        conn,
        month="2026-05",
        output_dir=tmp_path,
        templates_dir=TEMPLATES_DIR,
    )
    assert result.paper_count == 5  # 3 + 2


def test_monthly_digests_row_inserted(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """render_monthly upserts a digests row with kind='monthly'."""
    _setup_two_weeks(conn)
    render.render_monthly(
        conn,
        month="2026-05",
        output_dir=tmp_path,
        templates_dir=TEMPLATES_DIR,
    )
    row = conn.execute(
        "SELECT * FROM digests WHERE digest_id = '2026-05'"
    ).fetchone()
    assert row is not None
    assert str(row["kind"]) == "monthly"


def test_monthly_html_contains_cluster_labels(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """Monthly HTML contains cluster labels from all weeks."""
    _setup_two_weeks(conn)
    result = render.render_monthly(
        conn,
        month="2026-05",
        output_dir=tmp_path,
        templates_dir=TEMPLATES_DIR,
    )
    html = result.html_path.read_text(encoding="utf-8")
    assert "NLP Research" in html
    assert "Vision Models" in html


def test_monthly_html_is_well_formed(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """Monthly HTML parses without errors."""
    _setup_two_weeks(conn)
    result = render.render_monthly(
        conn,
        month="2026-05",
        output_dir=tmp_path,
        templates_dir=TEMPLATES_DIR,
    )
    html = result.html_path.read_text(encoding="utf-8")
    _assert_well_formed(html)


def test_monthly_rerender_no_duplicate_digest(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """Re-running render_monthly produces exactly one digests row."""
    _setup_two_weeks(conn)
    render.render_monthly(
        conn,
        month="2026-05",
        output_dir=tmp_path,
        templates_dir=TEMPLATES_DIR,
    )
    render.render_monthly(
        conn,
        month="2026-05",
        output_dir=tmp_path,
        templates_dir=TEMPLATES_DIR,
    )
    count = conn.execute(
        "SELECT COUNT(*) AS c FROM digests WHERE digest_id = '2026-05'"
    ).fetchone()["c"]
    assert int(count) == 1


def test_monthly_empty_month_produces_valid_page(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """A month with no data produces a valid HTML page."""
    result = render.render_monthly(
        conn,
        month="2026-01",
        output_dir=tmp_path,
        templates_dir=TEMPLATES_DIR,
    )
    assert result.paper_count == 0
    html = result.html_path.read_text(encoding="utf-8")
    _assert_well_formed(html)


def test_monthly_svg_chart_emitted(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """Monthly HTML contains SVG bar chart elements."""
    _setup_two_weeks(conn)
    result = render.render_monthly(
        conn,
        month="2026-05",
        output_dir=tmp_path,
        templates_dir=TEMPLATES_DIR,
    )
    html = result.html_path.read_text(encoding="utf-8")
    assert "<svg" in html
    assert "<rect" in html


# ── Yearly render tests ───────────────────────────────────────────────────────


def _setup_yearly_data(conn: sqlite3.Connection) -> None:
    """Insert papers and clusters across two months of 2026."""
    for i in range(3):
        _insert_paper(
            conn,
            f"2605.0500{i}",
            title=f"May Paper {i}",
            published_at="2026-05-11T00:00:00Z",
        )
    _insert_cluster(conn, "2026-W20", 0, label="Reinforcement Learning", paper_count=3)
    for i in range(3):
        _insert_paper_cluster(conn, f"2605.0500{i}", "2026-W20", 0)

    for i in range(2):
        _insert_paper(
            conn,
            f"2607.0600{i}",
            title=f"July Paper {i}",
            published_at="2026-07-14T00:00:00Z",
        )
    _insert_cluster(conn, "2026-W29", 0, label="Generative Models", paper_count=2)
    for i in range(2):
        _insert_paper_cluster(conn, f"2607.0600{i}", "2026-W29", 0)


def test_yearly_render_creates_html_file(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """render_yearly writes an HTML file to output_dir."""
    _setup_yearly_data(conn)
    result = render.render_yearly(
        conn,
        year="2026",
        output_dir=tmp_path,
        templates_dir=TEMPLATES_DIR,
    )
    assert result.html_path.exists()
    assert result.digest_id == "2026"


def test_yearly_render_aggregates_paper_count(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """render_yearly paper_count spans all weeks in the year."""
    _setup_yearly_data(conn)
    result = render.render_yearly(
        conn,
        year="2026",
        output_dir=tmp_path,
        templates_dir=TEMPLATES_DIR,
    )
    assert result.paper_count == 5  # 3 + 2


def test_yearly_digests_row_inserted(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """render_yearly upserts a digests row with kind='yearly'."""
    _setup_yearly_data(conn)
    render.render_yearly(
        conn,
        year="2026",
        output_dir=tmp_path,
        templates_dir=TEMPLATES_DIR,
    )
    row = conn.execute(
        "SELECT * FROM digests WHERE digest_id = '2026'"
    ).fetchone()
    assert row is not None
    assert str(row["kind"]) == "yearly"


def test_yearly_html_contains_cluster_labels(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """Yearly HTML contains cluster labels from all months."""
    _setup_yearly_data(conn)
    result = render.render_yearly(
        conn,
        year="2026",
        output_dir=tmp_path,
        templates_dir=TEMPLATES_DIR,
    )
    html = result.html_path.read_text(encoding="utf-8")
    assert "Reinforcement Learning" in html
    assert "Generative Models" in html


def test_yearly_html_is_well_formed(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """Yearly HTML parses without errors."""
    _setup_yearly_data(conn)
    result = render.render_yearly(
        conn,
        year="2026",
        output_dir=tmp_path,
        templates_dir=TEMPLATES_DIR,
    )
    html = result.html_path.read_text(encoding="utf-8")
    _assert_well_formed(html)


def test_yearly_rerender_no_duplicate_digest(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """Re-running render_yearly produces exactly one digests row."""
    _setup_yearly_data(conn)
    render.render_yearly(
        conn,
        year="2026",
        output_dir=tmp_path,
        templates_dir=TEMPLATES_DIR,
    )
    render.render_yearly(
        conn,
        year="2026",
        output_dir=tmp_path,
        templates_dir=TEMPLATES_DIR,
    )
    count = conn.execute(
        "SELECT COUNT(*) AS c FROM digests WHERE digest_id = '2026'"
    ).fetchone()["c"]
    assert int(count) == 1


def test_yearly_empty_year_produces_valid_page(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """A year with no data produces a valid HTML page."""
    result = render.render_yearly(
        conn,
        year="2025",
        output_dir=tmp_path,
        templates_dir=TEMPLATES_DIR,
    )
    assert result.paper_count == 0
    html = result.html_path.read_text(encoding="utf-8")
    _assert_well_formed(html)


def test_yearly_svg_chart_emitted(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """Yearly HTML contains SVG bar chart elements."""
    _setup_yearly_data(conn)
    result = render.render_yearly(
        conn,
        year="2026",
        output_dir=tmp_path,
        templates_dir=TEMPLATES_DIR,
    )
    html = result.html_path.read_text(encoding="utf-8")
    assert "<svg" in html
    assert "<rect" in html


# ── SVG chart unit tests ──────────────────────────────────────────────────────


def test_svg_bar_chart_empty() -> None:
    """Empty input produces a minimal SVG string (no rects)."""
    svg = render._svg_bar_chart([])
    assert "<svg" in svg
    assert "<rect" not in svg


def test_svg_bar_chart_single_row() -> None:
    """Single-row input produces one rect element."""
    svg = render._svg_bar_chart([("Topic A", 10)])
    assert svg.count("<rect") == 1
    assert "Topic A" in svg


def test_svg_bar_chart_multiple_rows() -> None:
    """Multiple rows produce the correct number of rect elements."""
    rows = [("Topic A", 10), ("Topic B", 7), ("Topic C", 3)]
    svg = render._svg_bar_chart(rows)
    assert svg.count("<rect") == 3
    for label, _ in rows:
        assert label in svg


def test_svg_bar_chart_zero_count_no_crash() -> None:
    """Zero paper count in all rows doesn't raise."""
    svg = render._svg_bar_chart([("Topic", 0)])
    assert "<svg" in svg


# ── Filter unit tests ─────────────────────────────────────────────────────────


def test_filter_format_score() -> None:
    assert render._filter_format_score(7.456) == "7.5"
    assert render._filter_format_score(0.0) == "0.0"
    assert render._filter_format_score(None) == "—"
    assert render._filter_format_score(10.0) == "10.0"


def test_filter_pct_delta() -> None:
    assert render._filter_pct_delta(12.3) == "+12%"
    assert render._filter_pct_delta(-8.0) == "-8%"
    assert render._filter_pct_delta(0.0) == "+0%"
    assert render._filter_pct_delta(None) == "—"


# ── Output dir auto-creation ──────────────────────────────────────────────────


def test_weekly_creates_nested_output_dir(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """output_dir is created if it does not exist (parents=True)."""
    nested = tmp_path / "a" / "b" / "c"
    _insert_paper(conn, "2605.00001")
    _insert_cluster(conn, "2026-W20", 0)
    _insert_paper_cluster(conn, "2605.00001", "2026-W20", 0)

    result = render.render_weekly(
        conn,
        week="2026-W20",
        output_dir=nested,
        templates_dir=TEMPLATES_DIR,
    )
    assert nested.exists()
    assert result.html_path.exists()
