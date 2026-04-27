"""Tests for the DuckDB-backed ParquetResultsQuery."""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from mteb.results import (
    BenchmarkResults,
    ModelResult,
    ParquetResultsQuery,
    TaskResult,
)

# Skip the whole module if duckdb is not installed -- the class is
# import-safe without duckdb, but every test method exercises the
# connection.
duckdb = pytest.importorskip("duckdb")


def _build_sample_results() -> BenchmarkResults:
    """Build a small BenchmarkResults covering filter / aggregation cases.

    Mirrors the fixture in ``test_benchmark_results_parquet.py`` but is
    duplicated here so each test file owns its own fixture.
    """
    task_a = TaskResult.model_construct(
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

    model_a = ModelResult.model_construct(
        model_name="org/model-a",
        model_revision="v1",
        task_results=[task_a, task_b],
    )
    model_b = ModelResult.model_construct(
        model_name="org/model-b",
        model_revision=None,
        experiment_name="batch-size=8",
        task_results=[task_c],
    )

    return BenchmarkResults.model_construct(model_results=[model_a, model_b])


@pytest.fixture
def parquet_path(tmp_path: Path) -> Path:
    """Write the sample BenchmarkResults to a temporary parquet file."""
    out = tmp_path / "results.parquet"
    _build_sample_results().to_parquet(out)
    return out


# ----------------------------------------------------------------------
# Schema introspection
# ----------------------------------------------------------------------
def test_num_rows_matches_flat_layout(parquet_path: Path) -> None:
    with ParquetResultsQuery(parquet_path) as q:
        # task_a: 2 splits x 1 subset = 2; task_b: 1 split x 2 subsets = 2;
        # task_c: 1 split x 1 subset = 1. Total = 5.
        assert q.num_rows == 5


def test_tasks_returns_sorted_distinct_task_names(parquet_path: Path) -> None:
    with ParquetResultsQuery(parquet_path) as q:
        assert q.tasks() == ["MIRACL", "STS12", "XPQARetrieval"]


def test_splits_returns_sorted_distinct_splits(parquet_path: Path) -> None:
    with ParquetResultsQuery(parquet_path) as q:
        assert q.splits() == ["test", "validation"]


def test_hf_subsets_returns_sorted_distinct_subsets(parquet_path: Path) -> None:
    with ParquetResultsQuery(parquet_path) as q:
        assert q.hf_subsets() == ["default", "deu-deu", "fra-fra"]


def test_languages_flattens_list_column(parquet_path: Path) -> None:
    with ParquetResultsQuery(parquet_path) as q:
        assert q.languages() == ["deu-Latn", "eng-Latn", "fra-Latn"]


def test_models_returns_distinct_model_revision_experiment(parquet_path: Path) -> None:
    with ParquetResultsQuery(parquet_path) as q:
        df = q.models()

    rows = {
        (r["model_name"], r["model_revision"], r["experiment_name"])
        for _, r in df.iterrows()
    }
    # NaN comparison via pandas: convert NaN to None for set membership.
    rows_normalized = {
        (
            n,
            None if (rev != rev) else rev,  # noqa: PLR0124  (NaN check)
            None if (exp != exp) else exp,  # noqa: PLR0124
        )
        for (n, rev, exp) in rows
    }
    assert rows_normalized == {
        ("org/model-a", "v1", None),
        ("org/model-b", None, "batch-size=8"),
    }


# ----------------------------------------------------------------------
# Filtered slicing
# ----------------------------------------------------------------------
def test_query_scores_filter_by_models(parquet_path: Path) -> None:
    with ParquetResultsQuery(parquet_path) as q:
        df = q.query_scores(models=["org/model-b"])
    assert len(df) == 1
    assert df["model_name"].unique().tolist() == ["org/model-b"]
    assert df["task_name"].unique().tolist() == ["MIRACL"]


def test_query_scores_filter_by_tasks_and_splits_is_anded(
    parquet_path: Path,
) -> None:
    with ParquetResultsQuery(parquet_path) as q:
        df = q.query_scores(tasks=["STS12"], splits=["test"])
    assert len(df) == 1
    row = df.iloc[0]
    assert row["task_name"] == "STS12"
    assert row["split"] == "test"
    assert row["main_score"] == pytest.approx(0.5)


def test_query_scores_languages_uses_list_has_any(parquet_path: Path) -> None:
    with ParquetResultsQuery(parquet_path) as q:
        df = q.query_scores(languages=["fra-Latn"])
    # task_b/test/fra-fra has ["fra-Latn"]; task_c has ["eng-Latn", "fra-Latn"]
    task_subset = sorted(zip(df["task_name"], df["hf_subset"]))
    assert task_subset == [("MIRACL", "default"), ("XPQARetrieval", "fra-fra")]


def test_query_scores_min_main_score_drops_nulls_and_below(
    parquet_path: Path,
) -> None:
    with ParquetResultsQuery(parquet_path) as q:
        df = q.query_scores(min_main_score=0.5)
    # Drops 0.42 (validation), drops the None row, keeps 0.5/0.7/0.65.
    scores = sorted(df["main_score"].tolist())
    assert scores == [pytest.approx(0.5), pytest.approx(0.65), pytest.approx(0.7)]


def test_query_scores_max_main_score_drops_above(parquet_path: Path) -> None:
    with ParquetResultsQuery(parquet_path) as q:
        df = q.query_scores(max_main_score=0.5)
    scores = sorted(df["main_score"].tolist())
    assert scores == [pytest.approx(0.42), pytest.approx(0.5)]


def test_query_scores_limit_is_applied(parquet_path: Path) -> None:
    with ParquetResultsQuery(parquet_path) as q:
        df = q.query_scores(limit=2)
    assert len(df) == 2


def test_query_scores_empty_in_list_returns_no_rows(parquet_path: Path) -> None:
    with ParquetResultsQuery(parquet_path) as q:
        df = q.query_scores(models=[])
    assert len(df) == 0


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------
def test_aggregate_main_score_default_groups_by_model_and_task(
    parquet_path: Path,
) -> None:
    with ParquetResultsQuery(parquet_path) as q:
        df = q.aggregate_main_score()

    by_pair = {
        (r["model_name"], r["task_name"]): r["main_score"] for _, r in df.iterrows()
    }
    # STS12 mean of (0.5, 0.42) = 0.46
    assert by_pair[("org/model-a", "STS12")] == pytest.approx(0.46)
    # XPQARetrieval mean over non-null (only 0.7)
    assert by_pair[("org/model-a", "XPQARetrieval")] == pytest.approx(0.7)
    # MIRACL only has one score
    assert by_pair[("org/model-b", "MIRACL")] == pytest.approx(0.65)
    assert len(df) == 3


def test_aggregate_main_score_count_counts_non_null_main_score(
    parquet_path: Path,
) -> None:
    with ParquetResultsQuery(parquet_path) as q:
        df = q.aggregate_main_score(group_by=["task_name"], agg="count")

    by_task = dict(zip(df["task_name"], df["main_score"]))
    # task_b has 2 rows but one has main_score=NULL; only 1 counts.
    assert by_task["XPQARetrieval"] == 1
    assert by_task["STS12"] == 2
    assert by_task["MIRACL"] == 1


def test_aggregate_main_score_min_max_median(parquet_path: Path) -> None:
    with ParquetResultsQuery(parquet_path) as q:
        mins = q.aggregate_main_score(group_by=["task_name"], agg="min")
        maxs = q.aggregate_main_score(group_by=["task_name"], agg="max")
        meds = q.aggregate_main_score(group_by=["task_name"], agg="median")

    by_min = dict(zip(mins["task_name"], mins["main_score"]))
    by_max = dict(zip(maxs["task_name"], maxs["main_score"]))
    by_med = dict(zip(meds["task_name"], meds["main_score"]))

    assert by_min["STS12"] == pytest.approx(0.42)
    assert by_max["STS12"] == pytest.approx(0.5)
    assert by_med["STS12"] == pytest.approx(0.46)
    assert by_min["MIRACL"] == pytest.approx(0.65)


def test_aggregate_main_score_pre_filter_applies_before_group(
    parquet_path: Path,
) -> None:
    with ParquetResultsQuery(parquet_path) as q:
        df = q.aggregate_main_score(
            group_by=["model_name"],
            tasks=["STS12"],
            splits=["test"],
        )
    by_model = dict(zip(df["model_name"], df["main_score"]))
    assert by_model == {"org/model-a": pytest.approx(0.5)}


def test_aggregate_main_score_invalid_agg_raises(parquet_path: Path) -> None:
    with ParquetResultsQuery(parquet_path) as q, pytest.raises(ValueError):
        q.aggregate_main_score(agg="sum")


def test_aggregate_main_score_empty_group_by_raises(parquet_path: Path) -> None:
    with ParquetResultsQuery(parquet_path) as q, pytest.raises(ValueError):
        q.aggregate_main_score(group_by=[])


def test_aggregate_main_score_unknown_group_by_column_raises(
    parquet_path: Path,
) -> None:
    with ParquetResultsQuery(parquet_path) as q, pytest.raises(ValueError):
        q.aggregate_main_score(group_by=["bogus_column"])


# ----------------------------------------------------------------------
# Escape hatch
# ----------------------------------------------------------------------
def test_sql_escape_hatch_returns_dataframe(parquet_path: Path) -> None:
    with ParquetResultsQuery(parquet_path) as q:
        df = q.sql(
            "SELECT task_name, COUNT(*) AS n FROM scores "
            "WHERE split = ? GROUP BY task_name ORDER BY task_name",
            "test",
        )
    by_task = dict(zip(df["task_name"], df["n"]))
    assert by_task == {"MIRACL": 1, "STS12": 1, "XPQARetrieval": 2}


# ----------------------------------------------------------------------
# Connection lifecycle
# ----------------------------------------------------------------------
def test_context_manager_closes_connection_on_exit(parquet_path: Path) -> None:
    q = ParquetResultsQuery(parquet_path)
    with q:
        assert q._conn is not None
    assert q._conn is None


def test_close_is_idempotent(parquet_path: Path) -> None:
    q = ParquetResultsQuery(parquet_path)
    q._ensure_conn()
    q.close()
    q.close()  # should not raise
    assert q._conn is None


# ----------------------------------------------------------------------
# Error paths
# ----------------------------------------------------------------------
def test_missing_parquet_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ParquetResultsQuery(tmp_path / "does_not_exist.parquet")


# ----------------------------------------------------------------------
# ResultCache integration
# ----------------------------------------------------------------------
def test_result_cache_parquet_query_roundtrip(tmp_path: Path) -> None:
    """ResultCache.parquet_query() finds the file in {cache}/leaderboard/."""
    from mteb.cache import ResultCache

    leaderboard_dir = tmp_path / "leaderboard"
    leaderboard_dir.mkdir()
    parquet_file = leaderboard_dir / "__cached_results.parquet"
    _build_sample_results().to_parquet(parquet_file)

    cache = ResultCache(cache_path=tmp_path)
    with cache.parquet_query() as q:
        assert q.num_rows == 5


def test_result_cache_parquet_query_missing_file_raises(tmp_path: Path) -> None:
    from mteb.cache import ResultCache

    cache = ResultCache(cache_path=tmp_path)
    with pytest.raises(FileNotFoundError):
        cache.parquet_query()
