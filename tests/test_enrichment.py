from __future__ import annotations

import sys
import unittest

from mlflow_langchain_enrichment import (
    TraceContext,
    TraceEnrichmentCallback,
    build_invoke_config,
    default_request_preview,
    default_response_preview,
    invoke_with_enrichment,
)


class _FakeMlflow:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def update_current_trace(self, **kwargs):
        self.calls.append(kwargs)


class _FakeRunnable:
    def __init__(self, response):
        self.response = response
        self.last_config = None

    def invoke(self, inputs, config=None, **kwargs):
        self.last_config = config
        return self.response


class EnrichmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_mlflow = sys.modules.get("mlflow")
        self.fake_mlflow = _FakeMlflow()
        sys.modules["mlflow"] = self.fake_mlflow

    def tearDown(self) -> None:
        if self.original_mlflow is None:
            sys.modules.pop("mlflow", None)
        else:
            sys.modules["mlflow"] = self.original_mlflow

    def test_default_request_preview_prefers_question(self) -> None:
        preview = default_request_preview({"question": "How do I rotate credentials?"})
        self.assertEqual(preview, "How do I rotate credentials?")

    def test_default_response_preview_prefers_content(self) -> None:
        preview = default_response_preview({"content": "Rotate them in the admin console."})
        self.assertEqual(preview, "Rotate them in the admin console.")

    def test_callback_updates_trace_for_root_chain(self) -> None:
        callback = TraceEnrichmentCallback(
            TraceContext(
                user_id="user-1",
                session_id="session-1",
                tags={"app": "agent"},
                metadata={"app_version": "1.2.3"},
            )
        )

        callback.on_chain_start({}, {"question": "hello"}, run_id="root")
        callback.on_chain_end({"answer": "world"}, run_id="root")

        self.assertEqual(len(self.fake_mlflow.calls), 2)
        self.assertEqual(self.fake_mlflow.calls[0]["request_preview"], "hello")
        self.assertEqual(self.fake_mlflow.calls[0]["metadata"]["mlflow.trace.user"], "user-1")
        self.assertEqual(
            self.fake_mlflow.calls[0]["metadata"]["mlflow.trace.session"], "session-1"
        )
        self.assertEqual(self.fake_mlflow.calls[1]["response_preview"], "world")
        self.assertEqual(self.fake_mlflow.calls[1]["state"], "OK")

    def test_callback_ignores_nested_run(self) -> None:
        callback = TraceEnrichmentCallback(TraceContext())
        callback.on_chain_start({}, {"question": "root"}, run_id="root")
        callback.on_chain_start({}, {"question": "child"}, run_id="child", parent_run_id="root")
        callback.on_chain_end({"answer": "child"}, run_id="child")

        self.assertEqual(len(self.fake_mlflow.calls), 1)
        self.assertEqual(self.fake_mlflow.calls[0]["request_preview"], "root")

    def test_build_invoke_config_appends_callback_and_metadata(self) -> None:
        config = build_invoke_config(
            TraceContext(trace_name="support-chat", span_metadata={"tenant": "acme"}),
            {"metadata": {"route": "billing"}, "callbacks": ["existing"]},
        )

        self.assertEqual(config["metadata"]["route"], "billing")
        self.assertEqual(config["metadata"]["tenant"], "acme")
        self.assertEqual(config["run_name"], "support-chat")
        self.assertEqual(config["callbacks"][0], "existing")
        self.assertEqual(type(config["callbacks"][1]).__name__, "TraceEnrichmentCallback")

    def test_invoke_with_enrichment_passes_built_config(self) -> None:
        runnable = _FakeRunnable({"answer": "ok"})

        result = invoke_with_enrichment(
            runnable,
            {"question": "hello"},
            TraceContext(span_metadata={"tenant": "acme"}),
            config={"metadata": {"route": "support"}},
        )

        self.assertEqual(result, {"answer": "ok"})
        self.assertEqual(runnable.last_config["metadata"]["route"], "support")
        self.assertEqual(runnable.last_config["metadata"]["tenant"], "acme")
        self.assertEqual(type(runnable.last_config["callbacks"][0]).__name__, "TraceEnrichmentCallback")


if __name__ == "__main__":
    unittest.main()
