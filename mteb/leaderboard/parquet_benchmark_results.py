"""DuckDB-backed BenchmarkResults for the leaderboard.

Replaces ``BenchmarkResults._build_pre_agg_df`` (Pydantic tree walk) and
``_aggregate_and_pivot`` (pandas groupby) with a single DuckDB query
against the cached parquet file. The class is duck-typed against
``BenchmarkResults`` so the table builders in
``mteb.benchmarks._create_table`` and the per-benchmark
``Benchmark._create_summary_table`` overrides keep working unchanged.

Filter operations (``_filter_tasks``, ``select_models``, ``select_tasks``,
``join_revisions``) compose an immutable filter spec rather than copying
data; SQL is only run on demand from ``to_dataframe`` / ``_get_scores`` /
the metadata properties.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, cast

import pandas as pd
from pydantic import PrivateAttr

from mteb.results.benchmark_results import (
    BenchmarkResults,
    _get_cached_model_metas,
    _parse_version_cached,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from mteb.abstasks.abstask import AbsTask
    from mteb.abstasks.task_metadata import TaskDomain, TaskType
    from mteb.models import ModelMeta
    from mteb.results.parquet_query import ParquetResultsQuery
    from mteb.types import (
        ISOLanguage,
        ISOLanguageScript,
        Modalities,
        Score,
        ScoresDict,
        SplitName,
    )


logger = logging.getLogger(__name__)


# Discriminator identifying a single canonical TaskResult kept by
# join_revisions: (model_name, task_name, model_revision, mteb_version,
# dataset_revision). The legacy `from_parquet` uses
# (task_name, dataset_revision, mteb_version, evaluation_time,
# kg_co2_emissions, date) to bucket TaskResults; mteb_version +
# dataset_revision capture nearly every duplicate in practice. This
# tuple is materialized as a DuckDB temp table and joined against
# scores to filter to canonical rows only.
_RevisionKeepRow = tuple[str, str, str | None, str | None, str | None]
# (task_name, split, hf_subset) tuple; the allowed-combos set produced by
# AbsTask.eval_splits / AbsTask.hf_subsets and applied via select_tasks().
_AllowedCombo = tuple[str, str, str]


@dataclass(frozen=True)
class _ParquetFilter:
    """Immutable AND-combined filter spec applied to the parquet view.

    ``None`` on any field means "no restriction". Sequence-valued fields
    use exact-match IN-list semantics; ``languages`` matches via
    ``list_has_any`` against the per-row languages list.
    """

    task_names: tuple[str, ...] | None = None
    model_names: tuple[str, ...] | None = None
    languages: tuple[str, ...] | None = None
    splits: tuple[str, ...] | None = None
    hf_subsets: tuple[str, ...] | None = None
    # Result of applying ``select_tasks(tasks)``. Each entry is the set of
    # (split, hf_subset) combinations the corresponding task allows
    # according to its AbsTask metadata. Materialized as a temp table at
    # query time when present.
    allowed_combos: tuple[_AllowedCombo, ...] | None = None
    # Result of ``join_revisions``. Each entry uniquely identifies a
    # canonical TaskResult: (model_name, task_name, model_revision,
    # mteb_version, dataset_revision). Materialized as a temp table.
    revision_keepset: tuple[_RevisionKeepRow, ...] | None = None
    # Result of ``_filter_tasks(is_public=...)``: pre-resolved into a
    # task_names IN-list at filter time using the task registry.

    def restrict_task_names(self, names: Iterable[str]) -> _ParquetFilter:
        """Return a new filter with task_names intersected against ``names``."""
        names_set = set(names)
        if self.task_names is not None:
            names_set &= set(self.task_names)
        return replace(self, task_names=tuple(sorted(names_set)))

    def restrict_model_names(self, names: Iterable[str]) -> _ParquetFilter:
        """Return a new filter with model_names intersected against ``names``."""
        names_set = set(names)
        if self.model_names is not None:
            names_set &= set(self.model_names)
        return replace(self, model_names=tuple(sorted(names_set)))


def _resolve_task_metadata_filter(
    *,
    candidate_task_names: Iterable[str],
    languages: list[str] | None,
    domains: list[TaskDomain] | None,
    task_types: list[TaskType] | None,
    modalities: list[Modalities] | None,
    is_public: bool | None,
) -> set[str]:
    """Resolve metadata-based task filters against the task registry.

    ``BenchmarkResults._filter_tasks`` accepts a mix of result-level
    filters (task_names) and metadata-level filters (languages, domains,
    task_types, modalities, is_public). The metadata filters all read
    from ``AbsTask.metadata``, so we resolve them once against the
    registry and emit the surviving task_name set; the SQL layer only
    needs to filter on task_name afterward.
    """
    from mteb.get_tasks import _TASKS_REGISTRY

    survivors: set[str] = set()
    for name in candidate_task_names:
        cls = _TASKS_REGISTRY.get(name)
        if cls is None:
            continue
        meta = cls.metadata
        if task_types is not None and meta.type not in task_types:
            continue
        if domains is not None:
            task_domains = meta.domains or []
            if not set(task_domains) & set(domains):
                continue
        if modalities is not None:
            task_modalities = meta.modalities or []
            if not set(task_modalities) & set(modalities):
                continue
        if languages is not None:
            task_langs = set(meta.languages or [])
            if not task_langs & set(languages):
                continue
        if is_public is not None and meta.is_public is not is_public:
            continue
        survivors.add(name)
    return survivors


def _allowed_combos_for_tasks(tasks: Iterable[AbsTask]) -> tuple[_AllowedCombo, ...]:
    """Compute the (task_name, split, hf_subset) keep-set for ``select_tasks``.

    Mirrors ``TaskResult.validate_and_filter_scores``: keeps only
    (split, hf_subset) pairs that the task's AbsTask metadata declares.
    Aggregate tasks collapse to ``hf_subset='default'``.
    """
    combos: list[_AllowedCombo] = []
    for task in tasks:
        name = task.metadata.name
        splits = list(task.eval_splits)
        if task.is_aggregate:
            subsets = ["default"]
        else:
            subsets = list(task.hf_subsets)
        for split in splits:
            for subset in subsets:
                combos.append((name, split, subset))
    return tuple(combos)


class ParquetBenchmarkResults(BenchmarkResults):
    """BenchmarkResults that runs every read as a DuckDB query.

    The Pydantic tree (``model_results``) is intentionally never
    populated; ``model_results=[]`` is kept only so this class still
    satisfies the parent's required attribute.

    This class is API-compatible with ``BenchmarkResults`` for the
    surface used by the leaderboard:

    - ``to_dataframe(format="long", aggregation_level=...)``
    - ``_get_scores(format="long", languages=...)``
    - ``_filter_tasks(...)`` / ``select_models(...)`` /
      ``select_tasks(...)`` / ``_filter_models(...)``
    - ``join_revisions()``
    - ``task_names`` / ``languages`` / ``task_types`` / ``domains`` /
      ``modalities`` / ``model_names`` / ``model_revisions``

    Cold construction is O(1) -- no SQL runs until the first read.
    Subsequent filter operations return a new instance with an updated
    filter spec, also O(1).
    """

    _parquet_query: ParquetResultsQuery = PrivateAttr()
    _filter: _ParquetFilter = PrivateAttr(default_factory=_ParquetFilter)
    # Per-instance memo for queries that don't depend on aggregation_level
    # (task_names, languages, etc.). Cleared automatically by being a
    # PrivateAttr on each fresh instance.
    _props_cache: dict[str, Any] = PrivateAttr(default_factory=dict)

    def __init__(
        self,
        *,
        parquet_query: ParquetResultsQuery,
        filter: _ParquetFilter | None = None,
        benchmark: Any = None,
    ) -> None:
        # Skip Pydantic validation: we don't store a real Pydantic tree.
        super().__init__(model_results=[], benchmark=benchmark)
        self._parquet_query = parquet_query
        self._filter = filter or _ParquetFilter()
        self._props_cache = {}

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _replace_filter(self, **updates: Any) -> ParquetBenchmarkResults:
        """Return a new instance with the given filter fields replaced."""
        return ParquetBenchmarkResults(
            parquet_query=self._parquet_query,
            filter=replace(self._filter, **updates),
            benchmark=self.benchmark,
        )

    def __repr__(self) -> str:
        f = self._filter
        return (
            "ParquetBenchmarkResults("
            f"tasks={len(f.task_names) if f.task_names else 'all'}, "
            f"models={len(f.model_names) if f.model_names else 'all'}, "
            f"langs={len(f.languages) if f.languages else 'all'}, "
            f"join_revisions={'yes' if f.revision_keepset is not None else 'no'}"
            ")"
        )

    def __hash__(self) -> int:
        return id(self)

    # ------------------------------------------------------------------
    # Internal: SQL builder shared by every read.
    # ------------------------------------------------------------------
    def _build_where(self) -> tuple[list[str], list[Any]]:
        """Construct WHERE clauses + bound params for the active filter.

        Columns are always qualified with ``scores.`` because temp-table
        joins (allowed_combos / revision_keepset) introduce columns of
        the same name that would otherwise make unqualified references
        ambiguous.
        """
        clauses: list[str] = []
        params: list[Any] = []
        f = self._filter

        for column, values in (
            ("task_name", f.task_names),
            ("model_name", f.model_names),
            ("split", f.splits),
            ("hf_subset", f.hf_subsets),
        ):
            if values is None:
                continue
            if not values:
                clauses.append("FALSE")
                continue
            placeholders = ", ".join(["?"] * len(values))
            clauses.append(f"scores.{column} IN ({placeholders})")
            params.extend(values)

        if f.languages is not None:
            if not f.languages:
                clauses.append("FALSE")
            else:
                placeholders = ", ".join(["?"] * len(f.languages))
                clauses.append(
                    f"list_has_any(scores.languages, [{placeholders}])"
                )
                params.extend(f.languages)

        return clauses, params

    def _execute(
        self, sql_template: str, params: list[Any]
    ) -> pd.DataFrame:
        """Run a query that joins the active allowed_combos / revision_keepset.

        ``sql_template`` must contain a ``{from_clause}`` placeholder.
        Materializes the keep-sets as DuckDB temp tables for the duration
        of the query and joins them into the FROM clause.

        Uses a fresh ``conn.cursor()`` per call. DuckDB connections are
        not thread-safe -- two threads sharing one connection corrupt
        each other's state and the second thread sees ``Attempting to
        execute an unsuccessful or closed pending query result``. Gradio
        runs callbacks on ``anyio.to_thread.run_sync``'s worker pool, so
        boot + interaction routinely fire concurrent reads. ``cursor()``
        returns an independent connection that shares the in-memory
        database (and the ``scores`` view) but has its own catalog state
        for registered views, so concurrent callbacks no longer collide.
        Registered temp tables are scoped to the cursor and released
        when the cursor is closed.
        """
        base_conn = self._parquet_query._ensure_conn()
        cursor = base_conn.cursor()
        try:
            view = self._parquet_query._view
            f = self._filter

            join_sqls: list[str] = []

            if f.allowed_combos is not None:
                if not f.allowed_combos:
                    return pd.DataFrame()
                combos_df = pd.DataFrame(
                    list(f.allowed_combos),
                    columns=["task_name", "split", "hf_subset"],
                )
                cursor.register("__allowed_combos", combos_df)
                join_sqls.append(
                    "INNER JOIN __allowed_combos ac "
                    "ON scores.task_name = ac.task_name "
                    "AND scores.split = ac.split "
                    "AND scores.hf_subset = ac.hf_subset"
                )

            if f.revision_keepset is not None:
                if not f.revision_keepset:
                    return pd.DataFrame()
                keep_df = pd.DataFrame(
                    list(f.revision_keepset),
                    columns=[
                        "model_name",
                        "task_name",
                        "model_revision",
                        "mteb_version",
                        "dataset_revision",
                    ],
                )
                cursor.register("__revision_keepset", keep_df)
                # IS NOT DISTINCT FROM is NULL-aware: a NULL on either
                # side only matches another NULL. Without it the JOIN
                # would drop rows whose mteb_version / dataset_revision
                # is NULL even when the canonical row chosen by
                # join_revisions is also NULL.
                join_sqls.append(
                    "INNER JOIN __revision_keepset k "
                    "ON scores.model_name = k.model_name "
                    "AND scores.task_name = k.task_name "
                    "AND scores.model_revision IS NOT DISTINCT FROM k.model_revision "
                    "AND scores.mteb_version IS NOT DISTINCT FROM k.mteb_version "
                    "AND scores.dataset_revision IS NOT DISTINCT FROM k.dataset_revision"
                )

            from_clause = f"{view} AS scores " + " ".join(join_sqls)
            sql = sql_template.format(from_clause=from_clause)
            return cursor.execute(sql, params).df()
        finally:
            cursor.close()

    # ------------------------------------------------------------------
    # to_dataframe (Step A + Step B fused into one SQL query)
    # ------------------------------------------------------------------
    def to_dataframe(  # noqa: PLR0912
        self,
        aggregation_level: Literal["subset", "split", "task", "language"] = "task",
        aggregation_fn: Callable[[list[Score]], Any] | None = None,
        include_model_revision: bool = False,
        format: Literal["wide", "long"] = "wide",
    ) -> pd.DataFrame:
        """Aggregate and return a DataFrame of scores for the active filter.

        SQL replacement for ``BenchmarkResults._build_pre_agg_df`` +
        ``_aggregate_and_pivot``: scores are averaged inside DuckDB and
        only the post-aggregation rows are materialized in pandas.

        ``aggregation_fn`` is ignored: SQL ``AVG(main_score)`` is always
        used. (The pandas implementation defaults to ``np.mean`` and the
        leaderboard never overrides it.)
        """
        if aggregation_fn is not None:
            logger.debug(
                "ParquetBenchmarkResults.to_dataframe ignores aggregation_fn; "
                "DuckDB AVG() is always used."
            )

        # Mirror BenchmarkResults._build_pre_agg_df: when callers don't
        # care about model_revision, we collapse multiple revisions of
        # the same (model, task) using the canonical revision picked by
        # join_revisions. join_revisions is idempotent so this is cheap
        # if it's already been applied.
        if not include_model_revision:
            return self.join_revisions()._to_dataframe_inner(
                aggregation_level=aggregation_level,
                include_model_revision=False,
                format=format,
            )
        return self._to_dataframe_inner(
            aggregation_level=aggregation_level,
            include_model_revision=include_model_revision,
            format=format,
        )

    def _to_dataframe_inner(  # noqa: PLR0912
        self,
        *,
        aggregation_level: Literal["subset", "split", "task", "language"],
        include_model_revision: bool,
        format: Literal["wide", "long"],
    ) -> pd.DataFrame:
        # Map each desired output dimension to (parquet column, SQL alias).
        # subset/language are renamed in the legacy long DataFrame so we
        # mirror that here.
        if aggregation_level == "task":
            index_dims: list[tuple[str, str]] = [("task_name", "task_name")]
        elif aggregation_level == "split":
            index_dims = [("task_name", "task_name"), ("split", "split")]
        elif aggregation_level == "subset":
            index_dims = [
                ("task_name", "task_name"),
                ("split", "split"),
                ("hf_subset", "subset"),
            ]
        elif aggregation_level == "language":
            # Special-cased below: requires UNNEST.
            index_dims = []
        else:
            raise ValueError(f"unknown aggregation_level: {aggregation_level!r}")

        clauses, params = self._build_where()
        where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""

        column_cols: list[str] = ["model_name"]
        if include_model_revision:
            column_cols.append("model_revision")

        if aggregation_level == "language":
            select_outer_cols = [*column_cols, "lang AS language"]
            group_outer_cols = [*column_cols, "lang"]
            cte = (
                "WITH exploded AS ("
                "  SELECT scores.model_name, scores.model_revision, "
                "         lang, scores.main_score "
                "  FROM {from_clause}, UNNEST(scores.languages) AS u(lang)"
                f"  {where_sql}"
                ")"
            )
            sql_template = (
                f"{cte} "
                f"SELECT {', '.join(select_outer_cols)}, "
                f"AVG(main_score) AS score "
                f"FROM exploded "
                f"GROUP BY {', '.join(group_outer_cols)}"
            )
        else:
            select_cols = [f"scores.{c}" for c in column_cols]
            group_cols = [f"scores.{c}" for c in column_cols]
            for src, alias in index_dims:
                if src == alias:
                    select_cols.append(f"scores.{src}")
                else:
                    select_cols.append(f"scores.{src} AS {alias}")
                group_cols.append(f"scores.{src}")
            select_cols.append("AVG(scores.main_score) AS score")

            sql_template = (
                f"SELECT {', '.join(select_cols)} "
                f"FROM {{from_clause}}"
                f"{where_sql} "
                f"GROUP BY {', '.join(group_cols)}"
            )

        df = self._execute(sql_template, params)

        if df.empty:
            return df

        if format == "long":
            return df

        # Wide format: pivot model_name (and optionally model_revision)
        # into columns. Mirrors pandas pivot_table from the legacy
        # _aggregate_and_pivot. Only used in tests / programmatic use.
        if aggregation_level == "language":
            wide_index = ["language"]
        else:
            wide_index = [alias for _, alias in index_dims]
        return df.pivot_table(
            index=wide_index,
            columns=column_cols,
            values="score",
            aggfunc="mean",
            observed=True,
        ).reset_index()

    # ------------------------------------------------------------------
    # _get_scores (long-format score listing for the leaderboard)
    # ------------------------------------------------------------------
    def _get_scores(  # noqa: PLR0913
        self,
        *,
        splits: list[SplitName] | None = None,
        languages: list[ISOLanguage | ISOLanguageScript] | None = None,
        scripts: list[ISOLanguageScript] | None = None,
        getter: Callable[[ScoresDict], Score] | None = None,
        aggregation: Callable[[list[Score]], Any] | None = None,
        format: Literal["wide", "long"] = "wide",
    ) -> list[dict] | dict:
        """Return per-(model, task) score rows; supports a language overlap filter.

        The leaderboard only uses ``format="long"``; we don't bother
        implementing wide here. ``scripts``, ``getter`` and
        ``aggregation`` from the legacy API are ignored -- the parquet
        cache only stores main_score.
        """
        if format != "long":
            raise NotImplementedError(
                "ParquetBenchmarkResults._get_scores only supports format='long'."
            )
        if scripts is not None or getter is not None or aggregation is not None:
            logger.debug(
                "ParquetBenchmarkResults._get_scores ignores scripts/getter/aggregation; "
                "main_score AVG is always used."
            )

        # The leaderboard composes select_models / _filter_tasks / etc.
        # before calling _get_scores. We layer the splits/languages
        # filter on top of the active filter spec just for this query.
        clauses, params = self._build_where()
        if splits is not None:
            if not splits:
                clauses.append("FALSE")
            else:
                placeholders = ", ".join(["?"] * len(splits))
                clauses.append(f"split IN ({placeholders})")
                params.extend(splits)
        if languages is not None:
            if not languages:
                clauses.append("FALSE")
            else:
                placeholders = ", ".join(["?"] * len(languages))
                clauses.append(
                    f"list_has_any(languages, [{placeholders}])"
                )
                params.extend(languages)

        where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""

        sql_template = (
            "SELECT scores.model_name, scores.model_revision, scores.task_name, "
            "AVG(scores.main_score) AS score, "
            "MAX(scores.mteb_version) AS mteb_version, "
            "MAX(scores.dataset_revision) AS dataset_revision, "
            "MAX(scores.evaluation_time) AS evaluation_time, "
            "MAX(scores.kg_co2_emissions) AS kg_co2_emissions "
            f"FROM {{from_clause}}{where_sql} "
            "GROUP BY scores.model_name, scores.model_revision, scores.task_name"
        )
        df = self._execute(sql_template, params)
        if df.empty:
            return []
        return df.to_dict("records")

    # ------------------------------------------------------------------
    # Filter operations (return new instances; no SQL runs).
    # ------------------------------------------------------------------
    def _filter_tasks(
        self,
        task_names: list[str] | None = None,
        *,
        languages: list[str] | None = None,
        domains: list[TaskDomain] | None = None,
        task_types: list[TaskType] | None = None,
        modalities: list[Modalities] | None = None,
        is_public: bool | None = None,
    ) -> ParquetBenchmarkResults:
        """Restrict to a (subset of) task_names + metadata filters.

        Metadata filters are resolved against the task registry once and
        collapsed into a task_names IN-list; no SQL is run here.
        """
        candidate = (
            set(task_names)
            if task_names is not None
            else set(self.task_names)
        )

        any_metadata_filter = (
            languages is not None
            or domains is not None
            or task_types is not None
            or modalities is not None
            or is_public is not None
        )
        if any_metadata_filter:
            candidate = _resolve_task_metadata_filter(
                candidate_task_names=candidate,
                languages=languages,
                domains=domains,
                task_types=task_types,
                modalities=modalities,
                is_public=is_public,
            )

        return self._replace_filter(
            task_names=tuple(sorted(candidate)),
        )

    def select_tasks(self, tasks: Iterable[AbsTask]) -> ParquetBenchmarkResults:
        """Restrict to the given AbsTask list, validating splits/subsets.

        Mirrors ``BenchmarkResults.select_tasks`` +
        ``TaskResult.validate_and_filter_scores``: keeps only score rows
        whose (task_name, split, hf_subset) appears in the task's
        metadata.
        """
        tasks_list = list(tasks)
        task_names = tuple(sorted({t.metadata.name for t in tasks_list}))
        allowed_combos = _allowed_combos_for_tasks(tasks_list)
        return self._replace_filter(
            task_names=task_names,
            allowed_combos=allowed_combos,
        )

    def select_models(
        self,
        names: list[str] | list[ModelMeta],
        revisions: list[str | None] | None = None,
    ) -> ParquetBenchmarkResults:
        """Restrict to the given model names (revision-aware).

        Matches ``BenchmarkResults.select_models``: when ``revisions`` is
        None (or the entry for a name is None), accept any revision for
        that model; otherwise require an exact match.
        """
        from mteb.models import ModelMeta

        _revisions = revisions if revisions is not None else [None] * len(names)
        if len(names) != len(_revisions):
            raise ValueError(
                "The length of names and revisions must be the same or revisions must be None."
            )

        name_rev: dict[str, str | None] = {}
        for name, revision in zip(names, _revisions):
            if isinstance(name, ModelMeta):
                if name.name is None:
                    raise ValueError(
                        "name in ModelMeta is None. It must be a string."
                    )
                name_rev[name.name] = name.revision
            else:
                name_rev[cast("str", name)] = revision

        # If the active filter already has a revision_keepset, intersect
        # against it so callers get the same (model, task, revision) rows
        # they would have under the legacy code path.
        new_filter = self._filter
        new_filter = replace(
            new_filter,
            model_names=tuple(sorted(name_rev.keys())),
        )

        # Apply revision-specific filtering by trimming the keepset
        # (when one exists) to only entries matching the requested
        # revisions. If revisions is None (or all-None), no further
        # restriction is needed.
        any_revision_pinned = any(
            rev is not None for rev in name_rev.values()
        )
        if any_revision_pinned and new_filter.revision_keepset is not None:
            keepset = tuple(
                row
                for row in new_filter.revision_keepset
                # row layout: (model_name, task_name, model_revision,
                # mteb_version, dataset_revision)
                if row[0] in name_rev
                and (
                    name_rev[row[0]] is None or row[2] == name_rev[row[0]]
                )
            )
            new_filter = replace(new_filter, revision_keepset=keepset)
        elif any_revision_pinned and new_filter.revision_keepset is None:
            # Without an existing keepset, we'd have to query the parquet
            # to produce one. The leaderboard always calls join_revisions
            # before select_models with pinned revisions, so this branch
            # is not exercised in production today. Punt for now.
            logger.warning(
                "select_models with pinned revisions on an un-joined "
                "ParquetBenchmarkResults is not supported; revisions ignored."
            )

        return ParquetBenchmarkResults(
            parquet_query=self._parquet_query,
            filter=new_filter,
            benchmark=self.benchmark,
        )

    def _filter_models(
        self,
        model_names: Iterable[str] | None = None,
        *,
        languages: Iterable[str] | None = None,
        open_weights: bool | None = None,
        frameworks: Iterable[str] | None = None,
        n_parameters_range: tuple[int | None, int | None] = (None, None),
        use_instructions: bool | None = None,
        zero_shot_on: list[AbsTask] | None = None,
    ) -> ParquetBenchmarkResults:
        """Restrict to models matching the given ModelMeta filters."""
        from mteb.models.get_model_meta import get_model_metas

        model_metas = get_model_metas(
            model_names=model_names,
            languages=languages,
            open_weights=open_weights,
            frameworks=frameworks,
            n_parameters_range=n_parameters_range,
            use_instructions=use_instructions,
            zero_shot_on=zero_shot_on,
        )
        names = sorted({m.name for m in model_metas if m.name is not None})
        return self._replace_filter(
            model_names=tuple(names),
        )

    def join_revisions(self) -> ParquetBenchmarkResults:
        """Pick a canonical (model, task, revision) for every (model, task).

        Mirrors ``BenchmarkResults.join_revisions``:

        1) revision matching the model's main ``ModelMeta.revision``
           wins (priority +1000)
        2) presence of a parseable ``mteb_version`` adds +100
        3) presence of a non-null, non-"external" revision adds +10
        4) tiebreak: max ``mteb_version``

        The result is collapsed into a ``revision_keepset`` filter that
        every subsequent query joins against. No score data is
        materialized; the keepset is small (one row per (model, task)
        present in the active filter).
        """
        if self._filter.revision_keepset is not None:
            # Idempotent: already joined.
            return self

        clauses, params = self._build_where()
        where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""

        # Pull the candidate (model, task, revision, mteb_version,
        # dataset_revision) rows once. DISTINCT collapses identical
        # entries (a TaskResult contributes many score rows but they
        # share the same five-tuple).
        sql_template = (
            "SELECT DISTINCT scores.model_name, scores.task_name, "
            "scores.model_revision, scores.mteb_version, "
            "scores.dataset_revision "
            f"FROM {{from_clause}}{where_sql}"
        )
        df = self._execute(sql_template, params)

        if df.empty:
            return self._replace_filter(revision_keepset=())

        # Score the candidates in pandas so we can reuse the existing
        # priority logic (Version-aware tiebreak, main-revision lookup
        # via ModelMeta).
        model_to_main = _get_cached_model_metas()
        df["main_revision"] = df["model_name"].map(model_to_main)
        df["mteb_version_parsed"] = df["mteb_version"].map(_parse_version_cached)

        df["priority"] = 0
        df.loc[df["model_revision"] == df["main_revision"], "priority"] += 1000
        df.loc[df["mteb_version_parsed"].notna(), "priority"] += 100
        valid_revision = df["model_revision"].notna() & (
            df["model_revision"] != "external"
        )
        df.loc[valid_revision, "priority"] += 10

        df = df.sort_values(
            ["model_name", "task_name", "priority", "mteb_version_parsed"],
            ascending=[True, True, False, False],
            na_position="last",
        )
        # NB: dropna=False so groups with NULL revisions don't get
        # silently dropped (a latent bug in the legacy join_revisions).
        keep = df.groupby(
            ["model_name", "task_name"], as_index=False, dropna=False
        ).first()

        def _opt_str(value: Any) -> str | None:
            return None if value is None or pd.isna(value) else cast("str", value)

        keepset: tuple[_RevisionKeepRow, ...] = tuple(
            (
                cast("str", row.model_name),
                cast("str", row.task_name),
                _opt_str(row.model_revision),
                _opt_str(row.mteb_version),
                _opt_str(row.dataset_revision),
            )
            for row in keep.itertuples(index=False)
        )
        return self._replace_filter(revision_keepset=keepset)

    # ------------------------------------------------------------------
    # Metadata properties (lazy DISTINCT queries, memoized per-instance).
    # ------------------------------------------------------------------
    def _distinct(self, column: str) -> list[str]:
        """SELECT DISTINCT helper for single-column metadata properties."""
        if column in self._props_cache:
            return self._props_cache[column]
        clauses, params = self._build_where()
        where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql_template = (
            f"SELECT DISTINCT scores.{column} AS v "
            f"FROM {{from_clause}}{where_sql}"
        )
        df = self._execute(sql_template, params)
        values = [v for v in df["v"].tolist() if v is not None]
        self._props_cache[column] = values
        return values

    @property
    def task_names(self) -> list[str]:  # type: ignore[override]
        if "task_names" in self._props_cache:
            return self._props_cache["task_names"]
        # Fast path: if the active filter already pins task_names and
        # nothing further restricts them (no allowed_combos /
        # revision_keepset), we can return them without a query.
        f = self._filter
        if (
            f.task_names is not None
            and f.allowed_combos is None
            and f.revision_keepset is None
        ):
            result = list(f.task_names)
            self._props_cache["task_names"] = result
            return result
        result = self._distinct("task_name")
        self._props_cache["task_names"] = result
        return result

    @property
    def model_names(self) -> list[str]:  # type: ignore[override]
        if "model_names" in self._props_cache:
            return self._props_cache["model_names"]
        result = self._distinct("model_name")
        self._props_cache["model_names"] = result
        return result

    @property
    def model_revisions(self) -> list[dict[str, str | None]]:  # type: ignore[override]
        if "model_revisions" in self._props_cache:
            return self._props_cache["model_revisions"]
        clauses, params = self._build_where()
        where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql_template = (
            "SELECT DISTINCT scores.model_name, scores.model_revision "
            f"FROM {{from_clause}}{where_sql}"
        )
        df = self._execute(sql_template, params)
        result = [
            {
                "model_name": row.model_name,
                "revision": (
                    None
                    if pd.isna(row.model_revision)
                    else row.model_revision
                ),
            }
            for row in df.itertuples(index=False)
        ]
        self._props_cache["model_revisions"] = result
        return result

    @property
    def languages(self) -> list[str]:  # type: ignore[override]
        if "languages" in self._props_cache:
            return self._props_cache["languages"]
        clauses, params = self._build_where()
        where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql_template = (
            "SELECT DISTINCT lang AS v "
            f"FROM {{from_clause}}, UNNEST(scores.languages) AS u(lang)"
            f"{where_sql}"
        )
        df = self._execute(sql_template, params)
        result = [v for v in df["v"].tolist() if v is not None]
        self._props_cache["languages"] = result
        return result

    @property
    def task_types(self) -> list[str]:  # type: ignore[override]
        if "task_types" in self._props_cache:
            return self._props_cache["task_types"]
        from mteb.get_tasks import _TASKS_REGISTRY

        result_set: set[str] = set()
        for name in self.task_names:
            cls = _TASKS_REGISTRY.get(name)
            if cls is None:
                continue
            result_set.add(cls.metadata.type)
        result = sorted(result_set)
        self._props_cache["task_types"] = result
        return result

    @property
    def domains(self) -> list[str]:  # type: ignore[override]
        if "domains" in self._props_cache:
            return self._props_cache["domains"]
        from mteb.get_tasks import _TASKS_REGISTRY

        result_set: set[str] = set()
        for name in self.task_names:
            cls = _TASKS_REGISTRY.get(name)
            if cls is None:
                continue
            for d in cls.metadata.domains or []:
                result_set.add(d)
        result = sorted(result_set)
        self._props_cache["domains"] = result
        return result

    @property
    def modalities(self) -> list[str]:  # type: ignore[override]
        if "modalities" in self._props_cache:
            return self._props_cache["modalities"]
        from mteb.get_tasks import _TASKS_REGISTRY

        result_set: set[str] = set()
        for name in self.task_names:
            cls = _TASKS_REGISTRY.get(name)
            if cls is None:
                continue
            for m in cls.metadata.modalities or []:
                result_set.add(m)
        result = sorted(result_set)
        # If a task has no modalities annotated, the legacy code falls
        # back to ["text"] via ModelResult.default_modalities. Match
        # that fallback so the leaderboard's modality multiselect still
        # has a sensible default.
        if not result:
            result = ["text"]
        self._props_cache["modalities"] = result
        return result

    # ------------------------------------------------------------------
    # Iteration / indexing: not supported; the Pydantic tree is empty.
    # ------------------------------------------------------------------
    def __iter__(self):  # type: ignore[override]
        raise TypeError(
            "ParquetBenchmarkResults does not materialize a Pydantic tree; "
            "iterate via to_dataframe() / _get_scores() / model_names instead."
        )

    def __getitem__(self, index):  # type: ignore[override]
        raise TypeError(
            "ParquetBenchmarkResults does not support __getitem__; "
            "use to_dataframe() / _get_scores() instead."
        )
