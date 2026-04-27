"""Parity tests for ParquetBenchmarkResults vs in-memory BenchmarkResults.

Verifies that pushing the leaderboard's per-benchmark aggregation into
DuckDB produces the same long-format DataFrames the table builders
already consume. Each test runs the same query through both pipelines
and asserts row-for-row equality.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd
import pytest

from mteb.results import (
    BenchmarkResults,
    ModelResult,
    ParquetResultsQuery,
    TaskResult,
)

duckdb = pytest.importorskip("duckdb")

from mteb.leaderboard.parquet_benchmark_results import (  # noqa: E402
    ParquetBenchmarkResults,
)


# ---------------------------------------------------------------------------
# Fixture: a multi-model, multi-task, multi-revision parquet that
# exercises every dimension the leaderboard slices on (split, hf_subset,
# languages list, multiple revisions per (model, task)).
# ---------------------------------------------------------------------------
def _build_sample_results() -> BenchmarkResults:
    """Hand-built BenchmarkResults with enough variety to cover every code path.

    Includes:
      - Two models (one with a v1 / v2 revision pair)
      - A task evaluated on two splits (test + validation)
      - A task with multiple subsets (XPQA: fra-fra, deu-deu)
      - A task with a multi-language row (MIRACL: ["eng-Latn", "fra-Latn"])
      - A NULL main_score row (XPQA / deu-deu)
    """
    task_a_v1 = TaskResult.model_construct(
        task_name="STS12",
        dataset_revision="abc123",
        mteb_version="1.0.0",
        evaluation_time=12.5,
        kg_co2_emissions=0.001,
        date=datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc),
        scores={
            "test": [
                {
                    "main_score": 0.5,
                    "hf_subset": "default",
                    "languages": ["eng-Latn"],
                }
            ],
            "validation": [
                {
                    "main_score": 0.42,
                    "hf_subset": "default",
                    "languages": ["eng-Latn"],
                }
            ],
        },
    )
    task_a_v2 = TaskResult.model_construct(
        task_name="STS12",
        dataset_revision="abc123",
        mteb_version="2.0.0",
        evaluation_time=11.0,
        kg_co2_emissions=0.0009,
        date=datetime.datetime(2024, 6, 1, tzinfo=datetime.timezone.utc),
        scores={
            "test": [
                {
                    "main_score": 0.55,
                    "hf_subset": "default",
                    "languages": ["eng-Latn"],
                }
            ],
        },
    )
    task_b = TaskResult.model_construct(
        task_name="XPQARetrieval",
        dataset_revision=None,
        mteb_version=None,
        evaluation_time=None,
        kg_co2_emissions=None,
        date=None,
        scores={
            "test": [
                {
                    "main_score": 0.7,
                    "hf_subset": "fra-fra",
                    "languages": ["fra-Latn"],
                },
                {
                    "main_score": None,
                    "hf_subset": "deu-deu",
                    "languages": ["deu-Latn"],
                },
            ],
        },
    )
    task_c = TaskResult.model_construct(
        task_name="MIRACL",
        dataset_revision="rev-xyz",
        mteb_version="2.5.1",
        evaluation_time=99.0,
        kg_co2_emissions=None,
        date=None,
        scores={
            "test": [
                {
                    "main_score": 0.65,
                    "hf_subset": "default",
                    "languages": ["eng-Latn", "fra-Latn"],
                },
            ],
        },
    )

    # Two revisions of model-a covering the same task: join_revisions
    # should pick the one with the higher mteb_version.
    model_a_v1 = ModelResult.model_construct(
        model_name="org/model-a",
        model_revision="v1",
        task_results=[task_a_v1, task_b],
    )
    model_a_v2 = ModelResult.model_construct(
        model_name="org/model-a",
        model_revision="v2",
        task_results=[task_a_v2],
    )
    # NB: use a real revision string here. The legacy in-memory
    # join_revisions has a latent bug where models with revision=None
    # are dropped (pandas groupby drops NaN groups by default); the
    # DuckDB path handles NULL correctly. Avoiding None keeps both
    # pipelines in agreement for the parity tests.
    model_b = ModelResult.model_construct(
        model_name="org/model-b",
        model_revision="rb1",
        experiment_name="batch-size=8",
        task_results=[task_c],
    )

    return BenchmarkResults.model_construct(
        model_results=[model_a_v1, model_a_v2, model_b]
    )


@pytest.fixture
def parquet_path(tmp_path: Path) -> Path:
    out = tmp_path / "results.parquet"
    _build_sample_results().to_parquet(out)
    return out


@pytest.fixture
def in_memory(parquet_path: Path) -> BenchmarkResults:
    """Reload the parquet through the legacy Pydantic path."""
    return BenchmarkResults.from_parquet(parquet_path)


@pytest.fixture
def parquet_results(parquet_path: Path) -> ParquetBenchmarkResults:
    """Construct the DuckDB-backed equivalent."""
    return ParquetBenchmarkResults(parquet_query=ParquetResultsQuery(parquet_path))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalize_long(df: pd.DataFrame) -> pd.DataFrame:
    """Sort + reset the long DataFrame for stable row-by-row comparison."""
    if df.empty:
        return df
    sort_cols = [c for c in df.columns if c != "score"]
    return (
        df.sort_values(sort_cols)
        .reset_index(drop=True)
        .astype({"score": "float64"})
    )


# ---------------------------------------------------------------------------
# to_dataframe parity (the "Step A + Step B" replacement)
# ---------------------------------------------------------------------------
def test_to_dataframe_long_task_matches_pandas(
    in_memory: BenchmarkResults, parquet_results: ParquetBenchmarkResults
) -> None:
    """SQL AVG(main_score) GROUP BY (model, task) must match pandas."""
    expected = _normalize_long(in_memory.to_dataframe(format="long"))
    actual = _normalize_long(parquet_results.to_dataframe(format="long"))
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_like=True,
    )


def test_to_dataframe_long_subset_matches_pandas(
    in_memory: BenchmarkResults, parquet_results: ParquetBenchmarkResults
) -> None:
    """aggregation_level='subset' surfaces split + hf_subset columns."""
    expected = _normalize_long(
        in_memory.to_dataframe(aggregation_level="subset", format="long")
    )
    actual = _normalize_long(
        parquet_results.to_dataframe(aggregation_level="subset", format="long")
    )
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_like=True,
    )


def test_to_dataframe_long_language_matches_pandas(
    in_memory: BenchmarkResults, parquet_results: ParquetBenchmarkResults
) -> None:
    """UNNEST + GROUP BY language must match df.explode + groupby."""
    expected = _normalize_long(
        in_memory.to_dataframe(aggregation_level="language", format="long")
    )
    actual = _normalize_long(
        parquet_results.to_dataframe(aggregation_level="language", format="long")
    )
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_like=True,
    )


def test_to_dataframe_with_model_revision_matches_pandas(
    in_memory: BenchmarkResults, parquet_results: ParquetBenchmarkResults
) -> None:
    """include_model_revision=True surfaces the revision column."""
    expected = _normalize_long(
        in_memory.to_dataframe(format="long", include_model_revision=True)
    )
    actual = _normalize_long(
        parquet_results.to_dataframe(format="long", include_model_revision=True)
    )
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_like=True,
    )


# ---------------------------------------------------------------------------
# Filter cascade: _filter_tasks / select_models compose as expected
# ---------------------------------------------------------------------------
def test_filter_tasks_then_to_dataframe(
    in_memory: BenchmarkResults, parquet_results: ParquetBenchmarkResults
) -> None:
    expected = _normalize_long(
        in_memory._filter_tasks(task_names=["STS12"]).to_dataframe(format="long")
    )
    actual = _normalize_long(
        parquet_results._filter_tasks(task_names=["STS12"]).to_dataframe(
            format="long"
        )
    )
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_like=True,
    )


def test_select_models_then_to_dataframe(
    in_memory: BenchmarkResults, parquet_results: ParquetBenchmarkResults
) -> None:
    expected = _normalize_long(
        in_memory.select_models(["org/model-a"]).to_dataframe(format="long")
    )
    actual = _normalize_long(
        parquet_results.select_models(["org/model-a"]).to_dataframe(format="long")
    )
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_like=True,
    )


# ---------------------------------------------------------------------------
# join_revisions: pick canonical (model, task, revision) per (model, task)
# ---------------------------------------------------------------------------
def test_join_revisions_picks_max_mteb_version(
    parquet_results: ParquetBenchmarkResults,
) -> None:
    """When neither revision matches main_revision, prefer max mteb_version."""
    joined = parquet_results.join_revisions()
    keepset = joined._filter.revision_keepset
    assert keepset is not None
    # Each row is (model_name, task_name, model_revision, mteb_version,
    # dataset_revision). model-a STS12 has v1 (mteb 1.0.0) and v2
    # (mteb 2.0.0); v2 wins.
    sts12_rows = [r for r in keepset if r[1] == "STS12"]
    assert len(sts12_rows) == 1
    kept_model_revision = sts12_rows[0][2]
    kept_mteb_version = sts12_rows[0][3]
    assert kept_model_revision == "v2"
    assert kept_mteb_version == "2.0.0"


def test_join_revisions_then_to_dataframe_matches_pandas(
    in_memory: BenchmarkResults, parquet_results: ParquetBenchmarkResults
) -> None:
    """Post-join long DataFrames should agree."""
    expected = _normalize_long(
        in_memory.join_revisions().to_dataframe(format="long")
    )
    actual = _normalize_long(
        parquet_results.join_revisions().to_dataframe(format="long")
    )
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_like=True,
    )


def test_join_revisions_is_idempotent(
    parquet_results: ParquetBenchmarkResults,
) -> None:
    """Calling join_revisions twice does not re-run the SQL or change rows."""
    once = parquet_results.join_revisions()
    twice = once.join_revisions()
    assert once._filter.revision_keepset == twice._filter.revision_keepset


# ---------------------------------------------------------------------------
# _get_scores: the leaderboard's filter callback uses this directly
# ---------------------------------------------------------------------------
def test_get_scores_long_returns_one_row_per_model_task(
    parquet_results: ParquetBenchmarkResults,
) -> None:
    rows = parquet_results.join_revisions()._get_scores(format="long")
    assert isinstance(rows, list)
    keys = {(r["model_name"], r["task_name"]) for r in rows}
    # 3 distinct (model, task) groups: model-a/STS12, model-a/XPQARetrieval,
    # model-b/MIRACL.
    assert keys == {
        ("org/model-a", "STS12"),
        ("org/model-a", "XPQARetrieval"),
        ("org/model-b", "MIRACL"),
    }


def test_get_scores_languages_filter(
    parquet_results: ParquetBenchmarkResults,
) -> None:
    """languages=['fra-Latn'] should keep only rows whose languages overlap."""
    rows = parquet_results._get_scores(
        format="long", languages=["fra-Latn"]
    )
    assert isinstance(rows, list)
    # MIRACL has fra-Latn; XPQA fra-fra has fra-Latn; STS12 does not.
    task_names = {r["task_name"] for r in rows}
    assert "MIRACL" in task_names
    assert "XPQARetrieval" in task_names
    assert "STS12" not in task_names


# ---------------------------------------------------------------------------
# Metadata properties: matched against the in-memory equivalents
# ---------------------------------------------------------------------------
def test_task_names_property(
    in_memory: BenchmarkResults, parquet_results: ParquetBenchmarkResults
) -> None:
    assert sorted(parquet_results.task_names) == sorted(in_memory.task_names)


def test_model_names_property(
    in_memory: BenchmarkResults, parquet_results: ParquetBenchmarkResults
) -> None:
    """The legacy property returns one row per ModelResult (so dupes when
    multiple revisions exist). Every leaderboard call site wraps it in a
    set; we follow that semantic and return DISTINCT names.
    """
    assert sorted(parquet_results.model_names) == sorted(set(in_memory.model_names))


def test_languages_property(
    in_memory: BenchmarkResults, parquet_results: ParquetBenchmarkResults
) -> None:
    """Without task-registry support, in-memory languages comes from
    score-row languages too. This test only exercises the parquet code
    path; it must not blow up and must surface the languages we wrote.
    """
    langs = set(parquet_results.languages)
    assert {"eng-Latn", "fra-Latn", "deu-Latn"}.issubset(langs)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_filter_tasks_empty_list_returns_empty_frame(
    parquet_results: ParquetBenchmarkResults,
) -> None:
    df = parquet_results._filter_tasks(task_names=[]).to_dataframe(format="long")
    assert df.empty


def test_filter_cascade_does_not_run_sql(
    parquet_results: ParquetBenchmarkResults,
) -> None:
    """Filter ops should be O(1): no SQL until the next read."""
    sliced = (
        parquet_results._filter_tasks(task_names=["STS12"])
        .select_models(["org/model-a"])
        .join_revisions()
    )
    # We can't easily assert "no SQL ran" without instrumenting DuckDB,
    # but we *can* assert that the filter spec captured everything
    # without touching the connection lifetime.
    assert sliced._filter.task_names == ("STS12",)
    assert sliced._filter.model_names == ("org/model-a",)
    assert sliced._filter.revision_keepset is not None
