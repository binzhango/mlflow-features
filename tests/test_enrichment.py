from __future__ import annotations

import sys
import unittest

from mlflow_langchain_enrichment import (
    TraceContext,
    close_root_trace,
    close_trace_context,
    enable_mlflow_langchain_enrichment,
    get_current_trace_context,
    invoke_with_enrichment,
    open_root_trace,
    open_trace_context,
    using_root_trace,
)
from langchain_core.callbacks.manager import CallbackManager

from mlflow_langchain_enrichment.auto import (
    RootTraceHandle,
    disable_mlflow_langchain_enrichment,
    reset_current_trace_context,
    set_current_trace_context,
    using_trace_context,
)
from mlflow_langchain_enrichment.enrichment import (
    TraceEnrichmentCallback,
    build_invoke_config,
    default_request_preview,
    default_response_preview,
)


class _FakeMlflow:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.started_runs: list[dict] = []
        self._active_run = None
        self.started_spans: list[dict] = []

    def update_current_trace(self, **kwargs):
        self.calls.append(kwargs)

    def active_run(self):
        return self._active_run

    def start_run(self, **kwargs):
        self.started_runs.append(kwargs)
        self._active_run = object()
        fake_mlflow = self

        class _RunContext:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                fake_mlflow._active_run = None
                return False

        return _RunContext()

    def start_span(self, **kwargs):
        self.started_spans.append(kwargs)
        fake_mlflow = self

        class _SpanContext:
            def __enter__(self_inner):
                self_inner.inputs = None
                self_inner.outputs = None
                fake_mlflow.started_spans[-1]["context"] = self_inner
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                return False

            def set_inputs(self_inner, inputs):
                self_inner.inputs = inputs

            def set_outputs(self_inner, outputs):
                self_inner.outputs = outputs

        return _SpanContext()


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
        disable_mlflow_langchain_enrichment()
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

    def test_build_invoke_config_dedupes_existing_trace_callback(self) -> None:
        existing = TraceEnrichmentCallback(TraceContext())
        config = build_invoke_config(
            TraceContext(),
            {"callbacks": [existing]},
        )
        self.assertEqual(config["callbacks"], [existing])

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

    def test_auto_enrichment_injects_callback_from_contextvar(self) -> None:
        enable_mlflow_langchain_enrichment()
        token = set_current_trace_context(TraceContext(tags={"feature": "auto"}))
        try:
            manager = CallbackManager.configure()
        finally:
            reset_current_trace_context(token)

        self.assertTrue(
            any(isinstance(handler, TraceEnrichmentCallback) for handler in manager.handlers)
        )

    def test_auto_enrichment_uses_provider(self) -> None:
        enable_mlflow_langchain_enrichment(
            lambda: {
                "tags": {"feature": "provider"},
                "user_id": "user-7",
            }
        )
        manager = CallbackManager.configure()
        callback = next(
            handler for handler in manager.handlers if isinstance(handler, TraceEnrichmentCallback)
        )

        self.assertEqual(callback.trace_context.tags["feature"], "provider")
        self.assertEqual(callback.trace_context.user_id, "user-7")

    def test_using_trace_context_can_start_mlflow_run(self) -> None:
        with using_trace_context(
            ensure_run=True,
            mlflow_run_name="support-request-run",
            run_tags={"team": "support"},
            trace_name="trace-name",
        ):
            self.assertIsNotNone(self.fake_mlflow.active_run())

        self.assertEqual(len(self.fake_mlflow.started_runs), 1)
        self.assertEqual(self.fake_mlflow.started_runs[0]["run_name"], "support-request-run")
        self.assertEqual(self.fake_mlflow.started_runs[0]["tags"]["team"], "support")

    def test_using_trace_context_does_not_start_nested_mlflow_run(self) -> None:
        self.fake_mlflow._active_run = object()

        with using_trace_context(
            ensure_run=True,
            mlflow_run_name="ignored",
        ):
            self.assertIsNotNone(self.fake_mlflow.active_run())

        self.assertEqual(self.fake_mlflow.started_runs, [])

    def test_open_and_close_trace_context_manage_state(self) -> None:
        handle = open_trace_context(
            user_id="user-9",
            ensure_run=True,
            mlflow_run_name="manual-open-close",
        )
        try:
            self.assertEqual(handle.trace_context.user_id, "user-9")
            self.assertIsNotNone(self.fake_mlflow.active_run())
            self.assertEqual(get_current_trace_context().user_id, "user-9")
        finally:
            close_trace_context(handle)

        self.assertIsNone(self.fake_mlflow.active_run())
        self.assertIsNone(get_current_trace_context())

    def test_open_and_close_trace_context_restore_previous_context(self) -> None:
        outer = set_current_trace_context(TraceContext(user_id="outer"))
        try:
            handle = open_trace_context(user_id="inner")
            try:
                self.assertEqual(handle.trace_context.user_id, "inner")
                self.assertEqual(get_current_trace_context().user_id, "inner")
            finally:
                close_trace_context(handle)
            self.assertEqual(get_current_trace_context().user_id, "outer")
        finally:
            reset_current_trace_context(outer)

    def test_open_root_trace_starts_root_span(self) -> None:
        handle = open_root_trace(
            user_id="user-10",
            session_id="session-10",
            trace_name="manual-root-test",
        )
        try:
            self.assertIsInstance(handle, RootTraceHandle)
            self.assertEqual(self.fake_mlflow.started_spans[0]["name"], "manual-root-test")
            self.assertEqual(
                self.fake_mlflow.calls[0]["metadata"]["mlflow.trace.session"],
                "session-10",
            )
        finally:
            close_root_trace(handle)

    def test_using_root_trace_records_error_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "boom"):
            with using_root_trace(
                user_id="user-11",
                session_id="session-11",
            ):
                raise ValueError("boom")

        self.assertEqual(self.fake_mlflow.calls[-1]["state"], "ERROR")

    def test_using_root_trace_records_response_state(self) -> None:
        with using_root_trace(
            user_id="user-12",
            session_id="session-12",
        ) as handle:
            handle.request = {"question": "hello"}
            handle.response = {"answer": "done"}

        self.assertEqual(self.fake_mlflow.calls[-1]["state"], "OK")
        self.assertEqual(self.fake_mlflow.calls[-1]["response_preview"], "done")
        span = self.fake_mlflow.started_spans[-1]["context"]
        self.assertEqual(span.inputs, {"question": "hello"})
        self.assertEqual(span.outputs, {"answer": "done"})

    def test_using_root_trace_can_disable_root_span_io(self) -> None:
        with using_root_trace(
            user_id="user-13",
            session_id="session-13",
            capture_root_span_io=False,
        ) as handle:
            handle.request = {"question": "hello"}
            handle.response = {"answer": "done"}

        span = self.fake_mlflow.started_spans[-1]["context"]
        self.assertIsNone(span.inputs)
        self.assertIsNone(span.outputs)


if __name__ == "__main__":
    unittest.main()
