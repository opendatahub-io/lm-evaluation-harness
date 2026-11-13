# EvalHub LongBench v2 context configuration

The EvalHub catalog ID is `longbenchv2`. The adapter maps it to the harness
task/group ID `longbench2` and preserves `longbenchv2` in the returned job results.
No EvalHub catalog ID change is required. Its primary metric is `acc`.
This implementation uses multiple-choice loglikelihood scoring. It is not the
official LongBench runner's answer-generation protocol; scores should not be
assumed comparable to its leaderboard.

Set `max_length` in the benchmark's job parameters to the effective context limit
confirmed for the serving endpoint and gateway. Do not assume the model's
advertised maximum is the deployed limit. If the endpoint does not expose a
reliable served limit, obtain it from the serving deployment/configuration or
the endpoint owner and provide that value explicitly. The adapter validates the
value's type and enforces it locally; it cannot verify the gateway's actual
limit. The backend reserves one output token, so prompt plus answer-choice
tokens must fit within `max_length - 1`. Rendered chat-template tokens are part
of the prompt.

```json
{
  "id": "longbenchv2",
  "provider_id": "lm_evaluation_harness",
  "parameters": {
    "max_length": 37568,
    "loglikelihood_context_overflow": "middle",
    "num_few_shot": 0,
    "apply_chat_template": true,
    "tokenizer": "meta-llama/Llama-3.3-70B-Instruct"
  }
}
```

This is an evaluation-request benchmark fragment. The context value is an
example server configuration, not a general Llama or LongBench default. Use the
tokenizer matching the served model. This requires an adapter image containing
the context and chat-template changes; older images can ignore these settings.

For `longbenchv2` (resolved to `longbench2`), the direct harness ID `longbench2`,
and its leaf tasks, `max_length` is required and
`loglikelihood_context_overflow` defaults to `middle`. Oversized prompts retain
their beginning and end; answer-choice continuations are never truncated. If an
answer choice itself leaves no room for a prompt token, the request fails before
it is sent. `"error"` can be selected to reject oversized requests, while
`"truncate"` preserves the legacy left-truncation behavior. Other tasks retain
their existing default behavior unless explicitly configured.

The result's `additional_info` records the configured context limit, effective
input budget, one-token output reservation, truncation method, number of
evaluated examples, number of distinct examples truncated, and number of
answer-choice likelihood requests truncated. Examples are identified by task
and document ID, so the four choices for one multiple-choice example count once.

Middle truncation reduces prompt size but does not by itself establish that the
MaaS gateway accepts the request/response size. The previous
`response_payload_too_large` / HTTP 500 failure is a separate serving-path
issue; validate the gateway independently and do not treat a successful local
truncation test as proof that this infrastructure issue is fixed.

Group results use the harness's aggregate, preserving `weight_by_size`.
`num_examples_evaluated` counts samples across the selected leaf tasks, including
nested groups. When sample logs are unavailable, the harness's effective sample
counts are used. A request limit applies per leaf task, not to the group as a
whole.

Tests in `tests/test_adapter_longbench.py` exercise budgets above the old 2K
default, middle truncation boundaries and choice preservation, unique-example
accounting, catalog-ID resolution, preserved result identity, weighted group
results, and sample coverage. These are local regression tests; they do not
establish that a deployed image, serving gateway, or full dataset run is
validated.

Sources: [harness task definition](../lm_eval/tasks/longbench2/_longbench_common_yaml),
[group aggregation](../lm_eval/tasks/longbench2/_longbench2.yaml), and the
[official LongBench v2 evaluation instructions](https://github.com/THUDM/LongBench#evaluation).
