"""Test cases for the _load_from_cache and _rebuild_from_full_repository methods."""

from unittest.mock import MagicMock, patch

import pytest

from mteb.cache import ResultCache
from mteb.results import BenchmarkResults


class TestLoadFromCache:
    """Test the _load_from_cache method."""

    def test_rebuild_flag_forces_full_rebuild(self, tmp_path):  # noqa: PLR6301
        """Test rebuild=True bypasses cache and forces rebuild."""
        cache = ResultCache(cache_path=tmp_path)
        cache_filename = "test_cache.json"
        expected_json_path = tmp_path / "leaderboard" / cache_filename
        expected_parquet_path = tmp_path / "leaderboard" / "__cached_results.parquet"
        expected_json_path.parent.mkdir(parents=True, exist_ok=True)
        expected_json_path.write_text('{"test": "should be ignored"}')

        with patch.object(cache, "_rebuild_from_full_repository") as mock_rebuild:
            mock_result = MagicMock(spec=BenchmarkResults)
            mock_rebuild.return_value = mock_result
            result = cache._load_from_cache(cache_filename, rebuild=True)
            mock_rebuild.assert_called_once_with(
                expected_json_path,
                parquet_cache_path=expected_parquet_path,
            )
            assert result == mock_result

    def test_local_parquet_preferred_over_local_json(self, tmp_path):  # noqa: PLR6301
        """A local parquet cache is preferred over a local JSON cache."""
        cache = ResultCache(cache_path=tmp_path)
        cache_filename = "test_cache.json"
        json_path = tmp_path / "leaderboard" / cache_filename
        parquet_path = tmp_path / "leaderboard" / "__cached_results.parquet"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text("{}")
        parquet_path.write_bytes(b"PAR1...")

        mock_result = MagicMock(spec=BenchmarkResults)
        with (
            patch("mteb.results.BenchmarkResults.from_parquet") as mock_from_parquet,
            patch("mteb.results.BenchmarkResults.from_disk") as mock_from_disk,
        ):
            mock_from_parquet.return_value = mock_result
            result = cache._load_from_cache(cache_filename, rebuild=False)
            mock_from_parquet.assert_called_once_with(parquet_path)
            mock_from_disk.assert_not_called()
            assert result == mock_result

    def test_local_json_used_when_parquet_missing(self, tmp_path):  # noqa: PLR6301
        """Falls back to the local JSON cache when no parquet exists."""
        cache = ResultCache(cache_path=tmp_path)
        cache_filename = "test_cache.json"
        json_path = tmp_path / "leaderboard" / cache_filename
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text("{}")

        mock_result = MagicMock(spec=BenchmarkResults)
        with patch("mteb.results.BenchmarkResults.from_disk") as mock_from_disk:
            mock_from_disk.return_value = mock_result
            result = cache._load_from_cache(cache_filename, rebuild=False)
            mock_from_disk.assert_called_once_with(json_path)
            assert result == mock_result

    def test_downloads_parquet_when_no_local_cache(self, tmp_path):  # noqa: PLR6301
        """When no local cache exists, the parquet download is tried first."""
        cache = ResultCache(cache_path=tmp_path)
        cache_filename = "test_cache.json"
        parquet_path = tmp_path / "leaderboard" / "__cached_results.parquet"
        mock_result = MagicMock(spec=BenchmarkResults)

        with (
            patch.object(cache, "_download_cached_parquet_from_branch") as mock_pq_dl,
            patch.object(cache, "_download_cached_results_from_branch") as mock_json_dl,
            patch("mteb.results.BenchmarkResults.from_parquet") as mock_from_parquet,
        ):
            mock_pq_dl.return_value = parquet_path
            mock_from_parquet.return_value = mock_result
            result = cache._load_from_cache(cache_filename, rebuild=False)
            mock_pq_dl.assert_called_once_with(output_path=parquet_path)
            mock_json_dl.assert_not_called()
            mock_from_parquet.assert_called_once_with(parquet_path)
            assert result == mock_result

    def test_falls_back_to_json_download_when_parquet_download_fails(self, tmp_path):  # noqa: PLR6301
        """If the parquet download fails, the JSON download is tried next."""
        cache = ResultCache(cache_path=tmp_path)
        cache_filename = "test_cache.json"
        json_path = tmp_path / "leaderboard" / cache_filename
        mock_result = MagicMock(spec=BenchmarkResults)

        with (
            patch.object(cache, "_download_cached_parquet_from_branch") as mock_pq_dl,
            patch.object(cache, "_download_cached_results_from_branch") as mock_json_dl,
            patch("mteb.results.BenchmarkResults.from_disk") as mock_from_disk,
        ):
            mock_pq_dl.side_effect = Exception("parquet not published yet")
            mock_json_dl.return_value = json_path
            mock_from_disk.return_value = mock_result
            result = cache._load_from_cache(cache_filename, rebuild=False)
            mock_pq_dl.assert_called_once()
            mock_json_dl.assert_called_once_with(output_path=json_path)
            mock_from_disk.assert_called_once_with(json_path)
            assert result == mock_result

    def test_falls_back_to_full_rebuild_when_all_downloads_fail(self, tmp_path):  # noqa: PLR6301
        """When both downloads fail, the full repository rebuild is invoked."""
        cache = ResultCache(cache_path=tmp_path)
        cache_filename = "test_cache.json"
        json_path = tmp_path / "leaderboard" / cache_filename
        parquet_path = tmp_path / "leaderboard" / "__cached_results.parquet"
        mock_result = MagicMock(spec=BenchmarkResults)

        with (
            patch.object(cache, "_download_cached_parquet_from_branch") as mock_pq_dl,
            patch.object(cache, "_download_cached_results_from_branch") as mock_json_dl,
            patch.object(cache, "_rebuild_from_full_repository") as mock_rebuild,
        ):
            mock_pq_dl.side_effect = Exception("parquet download failed")
            mock_json_dl.side_effect = Exception("json download failed")
            mock_rebuild.return_value = mock_result
            result = cache._load_from_cache(cache_filename, rebuild=False)
            mock_rebuild.assert_called_once_with(
                json_path,
                parquet_cache_path=parquet_path,
            )
            assert result == mock_result

    def test_corrupt_parquet_falls_through_to_json(self, tmp_path):  # noqa: PLR6301
        """A corrupt local parquet cache falls through to the local JSON cache."""
        cache = ResultCache(cache_path=tmp_path)
        cache_filename = "test_cache.json"
        json_path = tmp_path / "leaderboard" / cache_filename
        parquet_path = tmp_path / "leaderboard" / "__cached_results.parquet"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_path.write_bytes(b"not a parquet file")
        json_path.write_text("{}")

        mock_result = MagicMock(spec=BenchmarkResults)
        with (
            patch("mteb.results.BenchmarkResults.from_parquet") as mock_from_parquet,
            patch("mteb.results.BenchmarkResults.from_disk") as mock_from_disk,
        ):
            mock_from_parquet.side_effect = Exception("invalid parquet")
            mock_from_disk.return_value = mock_result
            result = cache._load_from_cache(cache_filename, rebuild=False)
            mock_from_parquet.assert_called_once_with(parquet_path)
            mock_from_disk.assert_called_once_with(json_path)
            assert result == mock_result

    def test_corrupt_local_caches_trigger_full_fallback(self, tmp_path):  # noqa: PLR6301
        """Corrupt local caches with failing downloads still trigger rebuild."""
        cache = ResultCache(cache_path=tmp_path)
        cache_filename = "test_cache.json"
        json_path = tmp_path / "leaderboard" / cache_filename
        parquet_path = tmp_path / "leaderboard" / "__cached_results.parquet"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text("invalid json {{{")
        parquet_path.write_bytes(b"not a parquet file")

        with (
            patch("mteb.results.BenchmarkResults.from_parquet") as mock_from_parquet,
            patch("mteb.results.BenchmarkResults.from_disk") as mock_from_disk,
            patch.object(cache, "_download_cached_parquet_from_branch") as mock_pq_dl,
            patch.object(cache, "_download_cached_results_from_branch") as mock_json_dl,
            patch.object(cache, "_rebuild_from_full_repository") as mock_rebuild,
        ):
            mock_from_parquet.side_effect = Exception("invalid parquet")
            mock_from_disk.side_effect = Exception("invalid JSON")
            mock_pq_dl.side_effect = Exception("download failed")
            mock_json_dl.side_effect = Exception("download failed")
            mock_result = MagicMock(spec=BenchmarkResults)
            mock_rebuild.return_value = mock_result
            result = cache._load_from_cache(cache_filename, rebuild=False)
            mock_rebuild.assert_called_once_with(
                json_path,
                parquet_cache_path=parquet_path,
            )
            assert result == mock_result


class TestRebuildFromFullRepository:
    """Test the _rebuild_from_full_repository method."""

    def test_full_rebuild_process_writes_only_json_when_no_parquet_path(  # noqa: PLR6301
        self, tmp_path
    ):
        """Without a parquet path, rebuild only writes the JSON quick cache."""
        cache = ResultCache(cache_path=tmp_path)
        quick_cache_path = tmp_path / "cache.json"

        with (
            patch.object(cache, "download_from_remote") as mock_download,
            patch.object(cache, "load_results") as mock_load_results,
            patch("mteb.cache.get_model_metas") as mock_get_model_metas,
        ):
            meta1 = MagicMock()
            meta1.name = "model1"
            meta2 = MagicMock()
            meta2.name = "model2"
            meta3 = MagicMock()
            meta3.name = None  # filtered out
            mock_get_model_metas.return_value = [meta1, meta2, meta3]
            mock_results = MagicMock(spec=BenchmarkResults)
            mock_load_results.return_value = mock_results

            result = cache._rebuild_from_full_repository(quick_cache_path)

            mock_download.assert_called_once()
            mock_load_results.assert_called_once_with(
                models=[meta1.name, meta2.name],
                only_main_score=True,
                require_model_meta=False,
                include_remote=True,
            )
            mock_results.to_disk.assert_called_once_with(quick_cache_path)
            mock_results.to_parquet.assert_not_called()
            assert result == mock_results

    def test_full_rebuild_writes_both_when_parquet_path_provided(self, tmp_path):  # noqa: PLR6301
        """When a parquet path is supplied, rebuild writes both JSON and parquet."""
        cache = ResultCache(cache_path=tmp_path)
        quick_cache_path = tmp_path / "cache.json"
        parquet_path = tmp_path / "cache.parquet"

        with (
            patch.object(cache, "download_from_remote"),
            patch.object(cache, "load_results") as mock_load_results,
            patch("mteb.cache.get_model_metas") as mock_get_model_metas,
        ):
            meta = MagicMock()
            meta.name = "model1"
            mock_get_model_metas.return_value = [meta]
            mock_results = MagicMock(spec=BenchmarkResults)
            mock_load_results.return_value = mock_results

            result = cache._rebuild_from_full_repository(
                quick_cache_path,
                parquet_cache_path=parquet_path,
            )

            mock_results.to_disk.assert_called_once_with(quick_cache_path)
            mock_results.to_parquet.assert_called_once_with(parquet_path)
            assert result == mock_results

    def test_parquet_write_failure_does_not_abort_rebuild(self, tmp_path):  # noqa: PLR6301
        """A failing parquet write must not break the rebuild."""
        cache = ResultCache(cache_path=tmp_path)
        quick_cache_path = tmp_path / "cache.json"
        parquet_path = tmp_path / "cache.parquet"

        with (
            patch.object(cache, "download_from_remote"),
            patch.object(cache, "load_results") as mock_load_results,
            patch("mteb.cache.get_model_metas") as mock_get_model_metas,
        ):
            meta = MagicMock()
            meta.name = "model1"
            mock_get_model_metas.return_value = [meta]
            mock_results = MagicMock(spec=BenchmarkResults)
            mock_results.to_parquet.side_effect = RuntimeError("disk full")
            mock_load_results.return_value = mock_results

            result = cache._rebuild_from_full_repository(
                quick_cache_path,
                parquet_cache_path=parquet_path,
            )

            mock_results.to_disk.assert_called_once_with(quick_cache_path)
            mock_results.to_parquet.assert_called_once_with(parquet_path)
            assert result == mock_results

    def test_rebuild_error_propagation(self, tmp_path):  # noqa: PLR6301
        """Test that errors during rebuild are properly propagated."""
        cache = ResultCache(cache_path=tmp_path)
        quick_cache_path = tmp_path / "cache.json"

        with patch.object(cache, "download_from_remote") as mock_download:
            mock_download.side_effect = Exception("Network error")
            with pytest.raises(Exception, match="Network error"):
                cache._rebuild_from_full_repository(quick_cache_path)

        with (
            patch.object(cache, "download_from_remote"),
            patch.object(cache, "load_results") as mock_load_results,
            patch("mteb.cache.get_model_metas") as mock_get_model_metas,
        ):
            meta = MagicMock()
            meta.name = "model1"
            mock_get_model_metas.return_value = [meta]
            mock_load_results.side_effect = Exception("Load failed")
            with pytest.raises(Exception, match="Load failed"):
                cache._rebuild_from_full_repository(quick_cache_path)
