"""Tests for job spec path validation in ``_read_job_spec_parameters_from_path``."""

from pathlib import Path

import pytest

import main


@pytest.fixture(autouse=True)
def local_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALHUB_MODE", "local")


@pytest.mark.parametrize("mode", ["local", " LOCAL ", "", "other"])
def test_non_k8s_mode_accepts_external_path(monkeypatch: pytest.MonkeyPatch, mode) -> None:
    monkeypatch.setenv("EVALHUB_MODE", mode)
    path = "/private/tmp/evalhub-jobs/job-1/meta/job.json"
    assert main._resolve_job_spec_path_for_read(path) == Path(path).resolve()


@pytest.mark.parametrize("path", ["/tmp/job.json", "/meta/../etc/passwd"])
@pytest.mark.parametrize("mode", ["k8s", " K8S ", None])
def test_k8s_rejects_path_outside_meta(monkeypatch: pytest.MonkeyPatch, path, mode) -> None:
    if mode is None:
        monkeypatch.delenv("EVALHUB_MODE")
    else:
        monkeypatch.setenv("EVALHUB_MODE", mode)
    assert main._resolve_job_spec_path_for_read(path) is None


def test_k8s_rejects_symlink_escape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EVALHUB_MODE", "k8s")
    root = tmp_path / "meta"
    root.mkdir()
    job = tmp_path / "job.json"
    job.write_text('{}')
    (root / "job.json").symlink_to(job)
    monkeypatch.setattr(main, "_JOB_SPEC_ALLOWED_ROOT", root)
    assert main._resolve_job_spec_path_for_read(str(root / "job.json")) is None


def test_resolve_accepts_local_runtime_path() -> None:
    path = "/private/tmp/evalhub-jobs/job-1/0/provider/benchmark/meta/job.json"
    assert main._resolve_job_spec_path_for_read(path) == Path(path).resolve()


@pytest.mark.parametrize("path", ["", "  ", None, "job\x00.json"])
def test_resolve_rejects_invalid_path(path) -> None:
    assert main._resolve_job_spec_path_for_read(path) is None


def test_resolve_accepts_meta_job_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALHUB_MODE", "k8s")
    resolved = main._resolve_job_spec_path_for_read("/meta/job.json")
    assert resolved is not None
    assert resolved == Path("/meta/job.json").resolve()


def test_read_parameters_success(tmp_path: Path) -> None:
    job = tmp_path / "evalhub-jobs/job-1/0/provider/benchmark/meta/job.json"
    job.parent.mkdir(parents=True)
    job.write_text('{"parameters": {"limit": 3}}', encoding="utf-8")
    assert main._read_job_spec_parameters_from_path(str(job)) == {"limit": 3}


def test_read_parameters_missing_file_returns_empty(
    tmp_path: Path
) -> None:
    assert main._read_job_spec_parameters_from_path(str(tmp_path / "nope.json")) == {}


def test_read_parameters_invalid_json_returns_empty(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job = tmp_path / "job.json"
    job.write_text("{not-json", encoding="utf-8")
    assert main._read_job_spec_parameters_from_path(str(job)) == {}
    err = capsys.readouterr().err
    assert "invalid JSON" in err


def test_seed_offline_reads_local_job_spec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    job = tmp_path / "evalhub-jobs/job-1/meta/job.json"
    job.parent.mkdir(parents=True)
    job.write_text('{"parameters": {"tokenizer": "/test_data/tokenizer"}}')
    monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(job))
    seen = []
    monkeypatch.setattr(
        main, "_infer_auto_offline_from_local_test_data",
        lambda parameters: parameters == {"tokenizer": "/test_data/tokenizer"},
    )
    monkeypatch.setattr(main, "configure_hf_offline_environment", seen.append)
    main._seed_hf_offline_before_lm_eval_import()
    assert seen == ["/test_data"]
