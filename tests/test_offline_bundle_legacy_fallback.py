"""Offline bundle resolution must prefer the namespaced slug and fall back to the legacy, un-namespaced slug."""

from pathlib import Path

import datasets

from lm_eval.api.task import _resolve_offline_bundle_dir


def _stage_bundle(root: Path, slug: str) -> Path:
    bundle = root / slug
    datasets.DatasetDict(
        {"test": datasets.Dataset.from_dict({"value": [1]})}
    ).save_to_disk(str(bundle))
    return bundle


def test_prefers_namespaced_bundle_when_both_exist(tmp_path: Path):
    namespaced = _stage_bundle(tmp_path, "Rowan--hellaswag")
    _stage_bundle(tmp_path, "hellaswag")

    resolved = _resolve_offline_bundle_dir(str(tmp_path), "Rowan/hellaswag", None)

    assert resolved == namespaced


def test_falls_back_to_legacy_bundle_when_namespaced_missing(tmp_path: Path):
    legacy = _stage_bundle(tmp_path, "hellaswag")

    resolved = _resolve_offline_bundle_dir(str(tmp_path), "Rowan/hellaswag", None)

    assert resolved == legacy


def test_falls_back_to_legacy_bundle_with_subset(tmp_path: Path):
    legacy = _stage_bundle(tmp_path, "super_glue--rte")

    resolved = _resolve_offline_bundle_dir(str(tmp_path), "aps/super_glue", "rte")

    assert resolved == legacy


def test_returns_none_when_neither_bundle_exists(tmp_path: Path):
    assert _resolve_offline_bundle_dir(str(tmp_path), "Rowan/hellaswag", None) is None


def test_no_legacy_fallback_for_already_unnamespaced_path(tmp_path: Path):
    # dataset_path without a "/" has no legacy form to fall back to.
    assert _resolve_offline_bundle_dir(str(tmp_path), "hellaswag", None) is None
