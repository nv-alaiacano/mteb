"""DuckDB-backed read-only query interface over the cached parquet.

The parquet file produced by ``BenchmarkResults.to_parquet`` (or by the
``embeddings-benchmark/results`` ``cached-data`` workflow) holds millions
of rows in a flat ``(model x task x split x subset)`` layout.
Materializing the whole tree into Pydantic via
``BenchmarkResults.from_parquet`` takes tens of seconds. For
"give me filtered/aggregated scores for one benchmark" queries we don't
need the tree -- DuckDB can answer those directly against the parquet
file in milliseconds, pushing predicates into the parquet reader.

``duckdb`` is lazy-imported so this module is safe to import even when
duckdb isn't installed -- the ImportError will only fire on first
connection use.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    import duckdb
    import pandas as pd


_VALID_AGGS = frozenset({"mean", "min", "max", "median", "count"})

_DUCKDB_IMPORT_ERROR = (
    "duckdb is required for ParquetResultsQuery. Install it with "
    "`pip install duckdb` or `pip install mteb[leaderboard]`."
)


class ParquetResultsQuery:
    """Read-only query interface over a cached results parquet file.

    Backed by a single in-memory DuckDB view over ``read_parquet(path)``.
    The view is named ``scores`` and exposes the same columns as
    ``BenchmarkResults._PARQUET_COLUMNS``.

    Examples:
        >>> from mteb.results import ParquetResultsQuery
        >>> with ParquetResultsQuery("~/.cache/mteb/leaderboard/__cached_results.parquet") as q:
        ...     df = q.query_scores(tasks=["STS12"], splits=["test"])
        ...     agg = q.aggregate_main_score(
        ...         group_by=["model_name", "task_name"],
        ...         agg="mean",
        ...         tasks=["STS12", "STS13"],
        ...     )
    """

    def __init__(self, parquet_path: Path | str) -> None:
        self.parquet_path = Path(parquet_path).expanduser().resolve()
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"parquet file not found: {self.parquet_path}")
        self._conn: duckdb.DuckDBPyConnection | None = None
        self._view = "scores"

    # ------------------------------------------------------------------
    # Connection plumbing
    # ------------------------------------------------------------------
    def _ensure_conn(self) -> duckdb.DuckDBPyConnection:
        """Lazily open the DuckDB connection and create the parquet view."""
        if self._conn is not None:
            return self._conn

        try:
            import duckdb
        except ImportError as e:
            raise ImportError(_DUCKDB_IMPORT_ERROR) from e

        conn = duckdb.connect(":memory:")
        # DuckDB does not support binding parameters inside read_parquet()
        # within a CREATE VIEW statement, so the path is interpolated as
        # a SQL string literal. self.parquet_path was resolved from a
        # caller-supplied Path on construction; single quotes are escaped
        # defensively to keep this safe even on weird filesystem paths.
        path_literal = str(self.parquet_path).replace("'", "''")
        conn.execute(
            f"CREATE VIEW {self._view} AS "  # noqa: S608 -- _view is a hardcoded identifier
            f"SELECT * FROM read_parquet('{path_literal}')"
        )
        self._conn = conn
        return conn

    def close(self) -> None:
        """Release the in-memory DuckDB connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> ParquetResultsQuery:
        self._ensure_conn()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------
    @property
    def num_rows(self) -> int:
        """Total number of rows in the parquet file."""
        conn = self._ensure_conn()
        result = conn.execute(
            f"SELECT COUNT(*) FROM {self._view}"  # noqa: S608 -- _view is a hardcoded identifier
        ).fetchone()
        return int(result[0]) if result is not None else 0

    def models(self) -> pd.DataFrame:
        """Return distinct ``(model_name, model_revision, experiment_name)`` rows."""
        conn = self._ensure_conn()
        return conn.execute(
            f"""
            SELECT DISTINCT model_name, model_revision, experiment_name
            FROM {self._view}
            ORDER BY model_name, model_revision, experiment_name
            """  # noqa: S608 -- _view is a hardcoded identifier
        ).df()

    def tasks(self) -> list[str]:
        """Sorted list of distinct task names."""
        conn = self._ensure_conn()
        rows = conn.execute(
            f"SELECT DISTINCT task_name FROM {self._view} ORDER BY task_name"  # noqa: S608 -- _view is a hardcoded identifier
        ).fetchall()
        return [r[0] for r in rows]

    def splits(self) -> list[str]:
        """Sorted list of distinct split names."""
        conn = self._ensure_conn()
        rows = conn.execute(
            f"SELECT DISTINCT split FROM {self._view} ORDER BY split"  # noqa: S608 -- _view is a hardcoded identifier
        ).fetchall()
        return [r[0] for r in rows]

    def hf_subsets(self) -> list[str]:
        """Sorted list of distinct hf_subset values."""
        conn = self._ensure_conn()
        rows = conn.execute(
            f"SELECT DISTINCT hf_subset FROM {self._view} ORDER BY hf_subset"  # noqa: S608 -- _view is a hardcoded identifier
        ).fetchall()
        return [r[0] for r in rows]

    def languages(self) -> list[str]:
        """Sorted list of distinct languages, flattened from the list column."""
        conn = self._ensure_conn()
        rows = conn.execute(
            f"""
            SELECT DISTINCT lang
            FROM {self._view}, UNNEST(languages) AS t(lang)
            ORDER BY lang
            """  # noqa: S608 -- _view is a hardcoded identifier
        ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Filtered slicing
    # ------------------------------------------------------------------
    def query_scores(
        self,
        *,
        models: Sequence[str] | None = None,
        tasks: Sequence[str] | None = None,
        splits: Sequence[str] | None = None,
        hf_subsets: Sequence[str] | None = None,
        languages: Sequence[str] | None = None,
        min_main_score: float | None = None,
        max_main_score: float | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Return rows matching all supplied filters.

        Filters are AND-combined. Sequence filters use ``column IN (...)``;
        ``languages`` matches via DuckDB's ``list_has_any`` (any element
        of the row's ``languages`` list appears in the requested set).

        Args:
            models: Restrict to these ``model_name`` values.
            tasks: Restrict to these ``task_name`` values.
            splits: Restrict to these ``split`` values.
            hf_subsets: Restrict to these ``hf_subset`` values.
            languages: Match rows whose ``languages`` overlap this list.
            min_main_score: Drop rows with ``main_score < min_main_score``.
                NULL ``main_score`` is also dropped when this is set.
            max_main_score: Drop rows with ``main_score > max_main_score``.
                NULL ``main_score`` is also dropped when this is set.
            limit: Optional row cap (applied after filtering, no ORDER BY).

        Returns:
            A pandas DataFrame with the same columns as the parquet file.
        """
        conn = self._ensure_conn()
        clauses, params = self._build_filters(
            models=models,
            tasks=tasks,
            splits=splits,
            hf_subsets=hf_subsets,
            languages=languages,
            min_main_score=min_main_score,
            max_main_score=max_main_score,
        )
        sql = f"SELECT * FROM {self._view}"  # noqa: S608 -- _view hardcoded
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return conn.execute(sql, params).df()

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def aggregate_main_score(
        self,
        *,
        group_by: Sequence[str] = ("model_name", "task_name"),
        agg: str = "mean",
        models: Sequence[str] | None = None,
        tasks: Sequence[str] | None = None,
        splits: Sequence[str] | None = None,
        hf_subsets: Sequence[str] | None = None,
        languages: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Group rows by ``group_by`` and aggregate ``main_score``.

        Args:
            group_by: Columns to group by. Must be non-empty and a subset
                of the parquet columns.
            agg: One of ``mean``, ``min``, ``max``, ``median``, ``count``.
                ``count`` counts non-NULL ``main_score`` rows.
            models: Optional pre-aggregation filter on ``model_name``.
            tasks: Optional pre-aggregation filter on ``task_name``.
            splits: Optional pre-aggregation filter on ``split``.
            hf_subsets: Optional pre-aggregation filter on ``hf_subset``.
            languages: Optional pre-aggregation filter on ``languages``.

        Returns:
            A DataFrame with one row per group; the aggregate column is
            always named ``main_score`` for ergonomic chaining.

        Raises:
            ValueError: If ``agg`` is not in ``_VALID_AGGS`` or if
                ``group_by`` is empty / contains unknown columns.
        """
        if agg not in _VALID_AGGS:
            raise ValueError(f"agg must be one of {sorted(_VALID_AGGS)}, got {agg!r}")
        group_by_list = list(group_by)
        if not group_by_list:
            raise ValueError("group_by must be non-empty")
        # Whitelist group_by columns against the actual parquet schema to
        # avoid SQL injection via column names (DuckDB does not parameterize
        # identifiers).
        from .benchmark_results import BenchmarkResults

        allowed = set(BenchmarkResults._PARQUET_COLUMNS)
        bad = [c for c in group_by_list if c not in allowed]
        if bad:
            raise ValueError(
                f"group_by contains unknown columns: {bad}. Allowed: {sorted(allowed)}"
            )

        agg_sql = _agg_to_sql(agg)
        conn = self._ensure_conn()
        clauses, params = self._build_filters(
            models=models,
            tasks=tasks,
            splits=splits,
            hf_subsets=hf_subsets,
            languages=languages,
        )
        group_cols = ", ".join(group_by_list)
        sql = (
            f"SELECT {group_cols}, {agg_sql} AS main_score "  # noqa: S608 -- group_cols/agg_sql validated against allowlists
            f"FROM {self._view}"
        )
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += f" GROUP BY {group_cols} ORDER BY {group_cols}"
        return conn.execute(sql, params).df()

    # ------------------------------------------------------------------
    # Escape hatch
    # ------------------------------------------------------------------
    def sql(self, query: str, *params: Any) -> pd.DataFrame:
        """Run an arbitrary SQL query against the parquet view.

        The view is named ``scores`` and has the columns listed on
        ``BenchmarkResults._PARQUET_COLUMNS``. Positional parameters are
        passed straight through to DuckDB; use ``?`` placeholders rather
        than string-formatting values into the query.

        Args:
            query: SQL query string (referencing the ``scores`` view).
            *params: Optional positional parameters bound to ``?`` placeholders.

        Returns:
            A pandas DataFrame of the result.
        """
        conn = self._ensure_conn()
        return conn.execute(query, list(params)).df()

    # ------------------------------------------------------------------
    # Internal: shared filter clause builder
    # ------------------------------------------------------------------
    @staticmethod
    def _build_filters(
        *,
        models: Sequence[str] | None = None,
        tasks: Sequence[str] | None = None,
        splits: Sequence[str] | None = None,
        hf_subsets: Sequence[str] | None = None,
        languages: Sequence[str] | None = None,
        min_main_score: float | None = None,
        max_main_score: float | None = None,
    ) -> tuple[list[str], list[Any]]:
        """Build the WHERE clause fragments + bound params for a query.

        Returns ``(clauses, params)`` where ``clauses`` is the list to
        AND-join into a WHERE clause and ``params`` is the matching
        positional parameter list for ``conn.execute``.
        """
        clauses: list[str] = []
        params: list[Any] = []

        for column, values in (
            ("model_name", models),
            ("task_name", tasks),
            ("split", splits),
            ("hf_subset", hf_subsets),
        ):
            if values is None:
                continue
            values_list = list(values)
            if not values_list:
                # Empty IN-list -> no rows can match; short-circuit.
                clauses.append("FALSE")
                continue
            placeholders = ", ".join(["?"] * len(values_list))
            clauses.append(f"{column} IN ({placeholders})")
            params.extend(values_list)

        if languages is not None:
            languages_list = list(languages)
            if not languages_list:
                clauses.append("FALSE")
            else:
                # list_has_any(a, b) -> true if any element of a is in b.
                # Build the array literal via placeholders so user-supplied
                # values are bound, not interpolated.
                placeholders = ", ".join(["?"] * len(languages_list))
                clauses.append(f"list_has_any(languages, [{placeholders}])")
                params.extend(languages_list)

        if min_main_score is not None:
            clauses.append("main_score >= ?")
            params.append(float(min_main_score))
        if max_main_score is not None:
            clauses.append("main_score <= ?")
            params.append(float(max_main_score))

        return clauses, params


def _agg_to_sql(agg: str) -> str:
    """Map a public ``agg`` name to the DuckDB SQL aggregate expression."""
    if agg == "mean":
        return "AVG(main_score)"
    if agg == "min":
        return "MIN(main_score)"
    if agg == "max":
        return "MAX(main_score)"
    if agg == "median":
        return "MEDIAN(main_score)"
    if agg == "count":
        return "COUNT(main_score)"
    raise ValueError(f"unsupported agg: {agg!r}")  # pragma: no cover
