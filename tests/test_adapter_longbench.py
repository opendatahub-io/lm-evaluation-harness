"""Long-context request budgets and nested-group reporting regressions."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import main
from lm_eval.api.instance import Instance
from lm_eval.models.openai_completions import LocalCompletionsAPI


def job(parameters, benchmark="longbench2"):
    return SimpleNamespace(
        benchmark_id=benchmark,
        model=SimpleNamespace(name="test/model", url="http://localhost:8000/v1"),
        parameters=parameters,
    )


@pytest.mark.parametrize("benchmark", ["longbenchv2", "longbench2"])
def test_longbench_explicit_budget_reaches_backend(benchmark):
    _, args, _ = main.build_lmeval_config(job({"max_length": 37568}, benchmark))
    assert args["loglikelihood_context_overflow"] == "middle"
    assert args["longbench_truncation_stats"] == {
        "truncated_example_ids": {},
        "truncated_requests": 0,
    }
    api = LocalCompletionsAPI(**{**args, "tokenizer_backend": None, "tokenizer": None})
    assert api.max_length == 37567  # The likelihood API reserves one output token.
    context = list(range(3000))
    inputs, lengths, _ = api.batch_loglikelihood_requests(
        [[(("prompt", " A"), context, [3001])]]
    )
    assert inputs == [context + [3001]]
    assert lengths == [3000]
    assert args["longbench_truncation_stats"] == {
        "truncated_example_ids": {},
        "truncated_requests": 0,
    }
    assert args["tokenized_requests"] is False


@pytest.mark.parametrize("benchmark", ["longbenchv2", "longbench2", "longbench2_code"])
def test_longbench_requires_explicit_served_budget(benchmark):
    with pytest.raises(ValueError, match="LongBench requires parameters.max_length"):
        main.build_lmeval_config(job({}, benchmark))


@pytest.mark.parametrize("value", [True, 0, 1, -1, "37568", 3.5, None])
def test_invalid_budget_is_rejected(value):
    with pytest.raises(ValueError, match="max_length must be an integer"):
        main.build_lmeval_config(job({"max_length": value}))


@pytest.mark.parametrize("mode", [None, True, "ignore"])
def test_invalid_overflow_mode_is_rejected(mode):
    with pytest.raises(ValueError, match="loglikelihood_context_overflow"):
        main.build_lmeval_config(
            job({"max_length": 20, "loglikelihood_context_overflow": mode})
        )


@pytest.mark.parametrize("concurrent", [1, 2])
def test_overflow_fails_before_any_model_call(monkeypatch, concurrent):
    _, args, _ = main.build_lmeval_config(
        job(
            {
                "max_length": 10,
                "num_concurrent": concurrent,
                "loglikelihood_context_overflow": "error",
            }
        )
    )
    api = LocalCompletionsAPI(**{**args, "tokenizer_backend": None, "tokenizer": None})
    api.tokenizer = (
        object()
    )  # Inputs below are already tokenized; no tokenizer download.
    call = MagicMock()
    monkeypatch.setattr(api, "model_call", call)
    async_call = MagicMock()
    monkeypatch.setattr(api, "get_batched_requests", async_call)
    with pytest.raises(ValueError, match="No truncated request was sent"):
        api._loglikelihood_tokens([(("prompt", " A"), list(range(9)), [10])])
    call.assert_not_called()
    async_call.assert_not_called()


def test_exact_budget_preserves_context_and_choice():
    api = LocalCompletionsAPI(
        tokenizer_backend=None, max_length=10, loglikelihood_context_overflow="error"
    )
    inputs, lengths, _ = api.batch_loglikelihood_requests(
        [[(("prompt", " A"), list(range(8)), [9])]]
    )
    assert inputs == [list(range(8)) + [9]]
    assert lengths == [8]


def test_explicit_truncation_and_other_task_default_are_preserved():
    _, args, _ = main.build_lmeval_config(
        job({"max_length": 10, "loglikelihood_context_overflow": "truncate"})
    )
    api = LocalCompletionsAPI(**{**args, "tokenizer_backend": None, "tokenizer": None})
    inputs, lengths, _ = api.batch_loglikelihood_requests(
        [[(("prompt", " A"), list(range(12)), [13])]]
    )
    assert inputs == [list(range(4, 12)) + [13]]
    assert lengths == [8]
    _, default_args, _ = main.build_lmeval_config(job({}, "arc_easy"))
    assert "max_length" not in default_args
    assert "loglikelihood_context_overflow" not in default_args


def test_middle_truncation_preserves_prompt_edges_and_complete_continuations():
    stats = {"truncated_example_ids": {}, "truncated_requests": 0}
    api = LocalCompletionsAPI(
        tokenizer_backend=None,
        max_length=10,
        loglikelihood_context_overflow="middle",
        longbench_truncation_stats=stats,
    )
    context = list(range(12))
    requests = [
        (("prompt", " A"), context, [91], ("longbench2_task", 4)),
        (("prompt", " B"), context, [92], ("longbench2_task", 4)),
        (("prompt 2", " A"), context, [93], ("longbench2_task", 5)),
        (("prompt 2", " B"), context, [94], ("longbench2_task", 5)),
    ]

    inputs, ctxlens, _ = api.batch_loglikelihood_requests([requests])

    expected_context = [0, 1, 2, 3, 8, 9, 10, 11]
    assert inputs == [expected_context + [91 + index] for index in range(4)]
    assert ctxlens == [8, 8, 8, 8]
    assert len(stats["truncated_example_ids"]) == 2
    assert stats["truncated_requests"] == 4


def test_middle_truncation_rejects_choice_that_cannot_preserve_both_prompt_edges():
    api = LocalCompletionsAPI(
        tokenizer_backend=None,
        max_length=4,
        loglikelihood_context_overflow="middle",
    )
    with pytest.raises(ValueError, match="cannot preserve both prompt edges"):
        api.batch_loglikelihood_requests(
            [[(("prompt", " answer"), [1, 2, 3], [4, 5])]]
        )


def test_left_truncation_is_included_in_protocol_counts():
    stats = {"truncated_example_ids": {}, "truncated_requests": 0}
    api = LocalCompletionsAPI(
        tokenizer_backend=None,
        max_length=10,
        loglikelihood_context_overflow="truncate",
        longbench_truncation_stats=stats,
    )
    requests = [
        (("prompt", " A"), list(range(12)), [91], ("longbench2_task", 4)),
        (("prompt", " B"), list(range(12)), [92], ("longbench2_task", 4)),
    ]

    inputs, ctxlens, _ = api.batch_loglikelihood_requests([requests])

    assert inputs == [list(range(4, 12)) + [91], list(range(4, 12)) + [92]]
    assert ctxlens == [8, 8]
    assert len(stats["truncated_example_ids"]) == 1
    assert stats["truncated_requests"] == 2


def test_string_request_retokenizes_decoded_text_before_budget_and_boundary():
    api = LocalCompletionsAPI(
        tokenizer_backend=None,
        tokenized_requests=False,
        max_length=4,
        loglikelihood_context_overflow="truncate",
    )
    api.tokenizer = object()
    api.decode_batch = MagicMock(
        side_effect=lambda tokens: [
            "old A" if tokens[0] == [1, 2, 3, 4] else "new A"
        ]
    )
    api.tok_encode = MagicMock(
        side_effect=lambda text: {
            "old A": [10, 11, 12, 13, 14],
            "new A": [20, 21, 22, 23],
            "new": [20, 21, 22],
            " A": [23],
        }[text]
    )
    api._encode_pair = MagicMock(return_value=([20, 21, 22], [23]))

    inputs, ctxlens, _ = api.batch_loglikelihood_requests(
        [[(("prompt", " A"), [1, 2, 3], [4])]]
    )

    assert inputs == [[20, 21, 22, 23]]
    assert ctxlens == [3]
    api._encode_pair.assert_called_once_with("new", " A")


def test_loglikelihood_retains_task_and_document_ids(monkeypatch):
    api = LocalCompletionsAPI(tokenizer_backend=None)
    monkeypatch.setattr(api, "_encode_pair", lambda context, continuation: ([1], [2]))
    dispatch = MagicMock(return_value=[])
    monkeypatch.setattr(api, "_loglikelihood_tokens", dispatch)
    request = Instance(
        request_type="loglikelihood",
        doc={},
        arguments=("prompt", " A"),
        idx=0,
        metadata=("longbench2_task", 17, 1),
    )

    api.loglikelihood([request])

    encoded_requests = dispatch.call_args.args[0]
    assert encoded_requests == [(("prompt", " A"), [1], [2], ("longbench2_task", 17))]


def group_results():
    return {
        "results": {"history": {"acc,none": 1.0}, "code": {"acc,none": 0.0}},
        "groups": {"longbench2": {"acc,none": 0.25}},
        "group_subtasks": {
            "longbench2": ["history_group", "code"],
            "history_group": ["history"],
        },
        "samples": {"history": [{}], "code": [{}, {}, {}]},
        "n-samples": {
            "history": {"original": 5, "effective": 1},
            "code": {"original": 10, "effective": 3},
        },
    }


@pytest.mark.parametrize(
    "benchmark",
    ["arc_easy", "truthfulqa_mc1", "ifeval", "gsm8k", "jsonschema_bench_easy"],
)
def test_other_benchmarks_keep_existing_context_and_generation_defaults(benchmark):
    _, args, gen_kwargs = main.build_lmeval_config(
        job({"parameters": {"max_gen_toks": 256}}, benchmark)
    )
    api = LocalCompletionsAPI(**{**args, "tokenizer_backend": None, "tokenizer": None})
    assert api.max_length == 2047
    assert api.loglikelihood_context_overflow == "truncate"
    assert gen_kwargs == "max_gen_toks=256"
    context = list(range(2050))
    inputs, lengths, _ = api.batch_loglikelihood_requests(
        [[(("prompt", " A"), context, [2051])]]
    )
    assert inputs == [context[4:] + [2051]]
    assert lengths == [2046]


@pytest.mark.parametrize("benchmark", ["longbenchv2", "longbench2"])
def test_nested_group_uses_weighted_score_and_actual_coverage(
    monkeypatch, tmp_path, benchmark
):
    import datasets

    monkeypatch.setattr(
        datasets.config, "HF_DATASETS_TRUST_REMOTE_CODE", False, raising=False
    )
    monkeypatch.setattr(main, "__file__", str(tmp_path / "main.py"))
    monkeypatch.setattr(
        main, "resolve_model_credentials", lambda: SimpleNamespace(api_key=None)
    )
    monkeypatch.setattr(main, "read_model_auth_key", lambda key: None)
    task_manager = MagicMock()
    task_manager._name_is_tag.return_value = False
    monkeypatch.setattr(main, "TaskManager", MagicMock(return_value=task_manager))
    monkeypatch.setattr(main, "_build_dataset_info", lambda *args: None)
    def fake_evaluate(**kwargs):
        stats = kwargs["model_args"]["longbench_truncation_stats"]
        stats["truncated_example_ids"]["[\"history\",0]"] = True
        stats["truncated_requests"] = 4
        return group_results()

    evaluate = MagicMock(side_effect=fake_evaluate)
    monkeypatch.setattr(main, "simple_evaluate", evaluate)
    config = job({"max_length": 37568}, benchmark)
    config.id = "00000000-0000-0000-0000-000000000001"
    config.benchmark_index = 0
    config.num_examples = 3
    config.exports = None
    config.model_dump = lambda: {}
    adapter = main.LMEvalAdapter.__new__(main.LMEvalAdapter)
    result = adapter.run_benchmark_job(config, MagicMock())
    assert evaluate.call_args.kwargs["tasks"] == ["longbench2"]
    assert result.benchmark_id == benchmark
    assert result.overall_score == 0.25
    assert result.num_examples_evaluated == 4
    assert adapter._run_info["num_samples_original"] == 15
    assert adapter._run_info["num_samples_effective"] == 4
    assert adapter._run_info["max_length"] == 37568
    assert adapter._run_info["configured_context_limit_tokens"] == 37568
    assert adapter._run_info["input_token_budget"] == 37567
    assert adapter._run_info["reserved_output_tokens"] == 1
    assert adapter._run_info["truncation_method"] == "middle"
    assert adapter._run_info["num_examples_evaluated"] == 4
    assert adapter._run_info["num_examples_truncated"] == 1
    assert adapter._run_info["num_truncated_loglikelihood_requests"] == 4
    assert (
        evaluate.call_args.kwargs["model_args"]["loglikelihood_context_overflow"]
        == "middle"
    )


def test_sample_counts_use_harness_counts_without_sample_logs():
    results = group_results()
    results.pop("samples")
    assert main._get_lmeval_evaluated_count(results, "longbench2") == 4


def test_duplicate_leaves_are_not_counted_twice():
    results = group_results()
    results["group_subtasks"]["longbench2"].append("history")
    assert main._get_lmeval_evaluated_count(results, "longbench2") == 4


def test_single_task_results_and_empty_sample_count_are_preserved():
    results = {"results": {"task": {"acc,none": 0.5}}, "samples": {"task": []}}
    assert main._get_lmeval_task_results(results, "task") == {"acc,none": 0.5}
    assert main._get_lmeval_evaluated_count(results, "task") == 0
