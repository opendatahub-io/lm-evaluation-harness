"""SGLang remote-code loading requires explicit opt-in, without a GPU."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lm_eval.api.registry import get_model
from lm_eval.models import sglang_causallms


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [
        ("", False),
        (",trust_remote_code=false", False),
        (",trust_remote_code=true", True),
    ],
    ids=["default-disabled", "explicit-disabled", "explicit-opt-in"],
)
def test_sglang_forwards_remote_code_opt_in(monkeypatch, extra_args, expected):
    engine = Mock(
        return_value=SimpleNamespace(
            tokenizer_manager=SimpleNamespace(tokenizer=object())
        )
    )
    monkeypatch.setattr(
        sglang_causallms, "sgl", SimpleNamespace(Engine=engine), raising=False
    )
    monkeypatch.setattr(sglang_causallms, "find_spec", lambda _name: object())

    backend = get_model("sglang").create_from_arg_string(
        "pretrained=test/model" + extra_args
    )

    engine.assert_called_once()
    assert engine.call_args.kwargs["trust_remote_code"] is expected
    assert backend.model_args["trust_remote_code"] is expected
