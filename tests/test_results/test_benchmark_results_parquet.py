"""Round-trip tests for BenchmarkResults.to_parquet / from_parquet."""

from __future__ import annotations

import datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from mteb.results import BenchmarkResults, ModelResult, TaskResult


def _build_sample_results() -> BenchmarkResults:
    """Build a small BenchmarkResults covering the edge cases we care about.

    - Two models, one with experiment_name set and a None model_revision.
    - Multiple tasks per model, including one with multiple splits and one
      with multiple subsets per split.
    - Mix of None and concrete values for evaluation_time, kg_co2_emissions,
      mteb_version, dataset_revision, date.
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


def _to_canonical_records(br: BenchmarkResults) -> list[tuple]:
    """Flatten BenchmarkResults to a sorted list of comparable tuples.

    Order-independent: parquet round-trips do not preserve insertion order
    of rows (the loader rebuilds the nested tree from accumulated dicts).
    """
    rows: list[tuple] = []
    for mr in br.model_results:
        for tr in mr.task_results:
            for split, score_list in tr.scores.items():
                for s in score_list:
                    rows.append(
                        (
                            mr.model_name,
                            mr.model_revision,
                            getattr(mr, "experiment_name", None),
                            tr.task_name,
                            tr.dataset_revision,
                            tr.mteb_version,
                            tr.evaluation_time,
                            tr.kg_co2_emissions,
                            tr.date,
                            split,
                            s.get("hf_subset"),
                            tuple(s.get("languages", [])),
                            s.get("main_score"),
                        )
                    )
    return sorted(rows, key=lambda r: tuple("" if v is None else str(v) for v in r))


def test_to_parquet_creates_file_with_expected_schema(tmp_path: Path) -> None:
    """to_parquet writes a parquet file matching the documented schema."""
    br = _build_sample_results()
    out = tmp_path / "results.parquet"

    br.to_parquet(out)

    assert out.exists()
    table = pq.read_table(out)
    assert table.num_rows == 5  # 2 from task_a, 2 from task_b, 1 from task_c
    assert table.column_names == list(BenchmarkResults._PARQUET_COLUMNS)


def test_round_trip_preserves_all_records(tmp_path: Path) -> None:
    """to_parquet -> from_parquet reconstructs the same flat record set."""
    br = _build_sample_results()
    out = tmp_path / "results.parquet"

    br.to_parquet(out)
    br_back = BenchmarkResults.from_parquet(out)

    assert _to_canonical_records(br_back) == _to_canonical_records(br)


def test_round_trip_preserves_model_and_task_grouping(tmp_path: Path) -> None:
    """The reconstructed object has the right number of models/tasks/splits."""
    br = _build_sample_results()
    out = tmp_path / "results.parquet"

    br.to_parquet(out)
    br_back = BenchmarkResults.from_parquet(out)

    assert len(br_back.model_results) == 2
    by_name = {m.model_name: m for m in br_back.model_results}

    a = by_name["org/model-a"]
    assert a.model_revision == "v1"
    assert {tr.task_name for tr in a.task_results} == {"STS12", "XPQARetrieval"}
    sts = next(tr for tr in a.task_results if tr.task_name == "STS12")
    assert set(sts.scores.keys()) == {"test", "validation"}

    b = by_name["org/model-b"]
    assert b.model_revision is None
    assert b.experiment_name == "batch-size=8"
    assert len(b.task_results) == 1


def test_round_trip_preserves_subsets_within_split(tmp_path: Path) -> None:
    """Multiple subsets within a single split survive the round-trip."""
    br = _build_sample_results()
    out = tmp_path / "results.parquet"

    br.to_parquet(out)
    br_back = BenchmarkResults.from_parquet(out)

    a = next(m for m in br_back.model_results if m.model_name == "org/model-a")
    xpqa = next(tr for tr in a.task_results if tr.task_name == "XPQARetrieval")
    test_scores = xpqa.scores["test"]
    assert len(test_scores) == 2
    by_subset = {s["hf_subset"]: s for s in test_scores}
    assert by_subset["fra-fra"]["main_score"] == pytest.approx(0.7)
    assert by_subset["deu-deu"]["main_score"] is None
    assert by_subset["fra-fra"]["languages"] == ["fra-Latn"]


def test_round_trip_preserves_languages_as_list(tmp_path: Path) -> None:
    """The languages column comes back as a Python list, not a numpy array."""
    br = _build_sample_results()
    out = tmp_path / "results.parquet"

    br.to_parquet(out)
    br_back = BenchmarkResults.from_parquet(out)

    b = next(m for m in br_back.model_results if m.model_name == "org/model-b")
    miracl = b.task_results[0]
    score = miracl.scores["test"][0]
    assert isinstance(score["languages"], list)
    assert score["languages"] == ["eng-Latn", "fra-Latn"]


def test_empty_benchmark_results_round_trips(tmp_path: Path) -> None:
    """Empty BenchmarkResults survives a round-trip without error."""
    br = BenchmarkResults.model_construct(model_results=[])
    out = tmp_path / "results.parquet"

    br.to_parquet(out)
    br_back = BenchmarkResults.from_parquet(out)

    assert br_back.model_results == []
