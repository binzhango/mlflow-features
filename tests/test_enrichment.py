from __future__ import annotations

import asyncio
import contextvars
import json
import sys
import unittest
from typing import Any

from deepagents import create_deep_agent
from deepagents.graph import resolve_model
from langchain.agents import create_agent
from mlflow_langchain_enrichment import (
    TraceContext,
    TraceSession,
    auto_trace_agent,
    auto_trace_llm,
    auto_trace_chain,
    close_root_trace,
    close_trace_context,
    enable_mlflow_langchain_enrichment,
    get_current_trace_context,
    invoke_with_enrichment,
    open_root_trace,
    open_trace_context,
    trace_llm,
    trace_llm_call,
    using_root_trace,
)
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

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
        self._span_stack: contextvars.ContextVar[tuple[dict[str, Any], ...]] = (
            contextvars.ContextVar("fake_mlflow_span_stack", default=())
        )
        self._next_span_id = 1
        self._next_trace_id = 1

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
        span_record = dict(kwargs)
        span_record["span_id"] = self._next_span_id
        self._next_span_id += 1
        current_stack = self._span_stack.get()
        parent = current_stack[-1] if current_stack else None
        span_record["parent_span_id"] = None if parent is None else parent["span_id"]
        if parent is None:
            span_record["trace_id"] = self._next_trace_id
            self._next_trace_id += 1
        else:
            span_record["trace_id"] = parent["trace_id"]
        self.started_spans.append(span_record)
        fake_mlflow = self

        class _SpanContext:
            def __enter__(self_inner):
                self_inner.inputs = None
                self_inner.outputs = None
                self_inner.attributes = {}
                fake_mlflow.started_spans[-1]["context"] = self_inner
                self_inner._span_stack_token = fake_mlflow._span_stack.set(
                    (*fake_mlflow._span_stack.get(), fake_mlflow.started_spans[-1])
                )
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                fake_mlflow._span_stack.reset(self_inner._span_stack_token)
                return False

            def set_inputs(self_inner, inputs):
                self_inner.inputs = inputs

            def set_outputs(self_inner, outputs):
                self_inner.outputs = outputs

            def set_attribute(self_inner, key, value):
                self_inner.attributes[key] = value

            def set_attributes(self_inner, attributes):
                self_inner.attributes.update(attributes)

            def set_span_type(self_inner, span_type):
                self_inner.attributes["mlflow.spanType"] = span_type

        return _SpanContext()


class _FakeRunnable:
    def __init__(self, response):
        self.response = response
        self.last_config = None
        self.last_kwargs = None
        self.model = "fake-llm"

    def invoke(self, inputs, config=None, **kwargs):
        self.last_config = config
        self.last_kwargs = kwargs
        return self.response

    async def ainvoke(self, inputs, config=None, **kwargs):
        self.last_config = config
        self.last_kwargs = kwargs
        return self.response

    def stream(self, inputs, config=None, **kwargs):
        self.last_config = config
        self.last_kwargs = kwargs
        for chunk in self.response:
            yield chunk

    async def astream(self, inputs, config=None, **kwargs):
        self.last_config = config
        self.last_kwargs = kwargs
        for chunk in self.response:
            yield chunk


class _ErrorRunnable(_FakeRunnable):
    def invoke(self, inputs, config=None, **kwargs):
        self.last_config = config
        raise ValueError("boom")


class _FakeChatModel(BaseChatModel):
    model_name: str = "fake-chat"

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="chat-ok",
                        response_metadata={"model": self.model_name},
                    )
                )
            ]
        )

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return _FakeRunnable(AIMessage(content="tool-ok"))


class _StaticChatModel(BaseChatModel):
    response_text: str = "ok"
    model_name: str = "static-chat"
    usage_metadata: dict[str, int] | None = None

    @property
    def _llm_type(self) -> str:
        return "static-chat"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=self.response_text,
                        response_metadata={"model": self.model_name},
                        usage_metadata=self.usage_metadata,
                    ),
                )
            ]
        )

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


class _SupervisorToolCallingChatModel(BaseChatModel):
    tool_name: str = "ask_child_agent"

    @property
    def _llm_type(self) -> str:
        return "supervisor-tool-chat"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        saw_tool_response = any(isinstance(message, ToolMessage) for message in messages)
        if not saw_tool_response:
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": self.tool_name,
                                    "args": {"question": "delegate this"},
                                    "id": "tool-call-1",
                                    "type": "tool_call",
                                }
                            ],
                        )
                    )
                ]
            )

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content="supervisor-final"),
                )
            ]
        )

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


class _DeepAgentTaskCallingChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "deepagent-task-chat"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        saw_tool_response = any(isinstance(message, ToolMessage) for message in messages)
        if not saw_tool_response:
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "task",
                                    "args": {
                                        "description": "Research the issue and return a short summary.",
                                        "subagent_type": "research-subagent",
                                    },
                                    "id": "task-call-1",
                                    "type": "tool_call",
                                }
                            ],
                        )
                    )
                ]
            )

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content="deepagent-final"),
                )
            ]
        )

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


class _FlakyChatModel(BaseChatModel):
    failures_before_success: int = 1
    attempts: int = 0
    model_name: str = "flaky-model"
    usage_metadata: dict[str, int] | None = None

    @property
    def _llm_type(self) -> str:
        return "flaky-chat"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.attempts += 1
        if self.attempts <= self.failures_before_success:
            raise ConnectionError(f"transient-{self.attempts}")
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="recovered",
                        response_metadata={"model": self.model_name},
                        usage_metadata=self.usage_metadata,
                    )
                )
            ]
        )

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


class ErgonomicApiTests(unittest.TestCase):
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

    def test_auto_trace_llm_wraps_raw_ainvoke(self) -> None:
        traced_llm = auto_trace_llm(
            _FakeRunnable({"answer": "ok"}),
            user_id="user-30",
            session_id="session-30",
            trace_name="auto-llm",
        )

        result = asyncio.run(traced_llm.ainvoke({"question": "hello"}))

        self.assertEqual(result, {"answer": "ok"})
        self.assertEqual(self.fake_mlflow.started_spans[0]["name"], "auto-llm")
        self.assertEqual(self.fake_mlflow.calls[-1]["metadata"]["mlflow.trace.user"], "user-30")

    def test_auto_trace_llm_can_add_dynamic_tags_and_metadata_without_request_model(self) -> None:
        traced_llm = auto_trace_llm(
            _FakeRunnable({"content": "ok"}),
            tags={"app": "support-bot"},
            metadata={"app_version": "0.1.0"},
            tags_builder=lambda runnable, inputs, response, error: {
                "model": runnable.model,
            },
            metadata_builder=lambda runnable, inputs, response, error: {
                "model_name": runnable.model,
                "response_type": type(response).__name__,
            },
        )

        result = asyncio.run(traced_llm.ainvoke({"question": "hello"}))

        self.assertEqual(result, {"content": "ok"})
        self.assertEqual(self.fake_mlflow.calls[-1]["tags"]["app"], "support-bot")
        self.assertEqual(self.fake_mlflow.calls[-1]["tags"]["model"], "fake-llm")
        self.assertEqual(self.fake_mlflow.calls[-1]["metadata"]["app_version"], "0.1.0")
        self.assertEqual(self.fake_mlflow.calls[-1]["metadata"]["model_name"], "fake-llm")
        self.assertEqual(self.fake_mlflow.calls[-1]["metadata"]["response_type"], "dict")

    def test_auto_trace_llm_preserves_base_chat_model_identity(self) -> None:
        traced_llm = auto_trace_llm(
            _FakeChatModel(),
            session_id="session-chat",
            trace_name="chat-trace",
        )

        self.assertIsInstance(traced_llm, BaseChatModel)
        self.assertIs(resolve_model(traced_llm), traced_llm)

        result = traced_llm.invoke("hello")

        self.assertEqual(result.content, "chat-ok")
        self.assertEqual(self.fake_mlflow.started_spans[0]["name"], "chat-trace")
        self.assertEqual(
            self.fake_mlflow.calls[-1]["metadata"]["mlflow.trace.session"],
            "session-chat",
        )

    def test_auto_trace_llm_preserves_chat_model_identity_for_bound_llm(self) -> None:
        traced_llm = auto_trace_llm(
            _FakeChatModel().with_config({"run_name": "bound-chat"}),
            session_id="session-bound-chat",
            trace_name="bound-chat-trace",
        )

        self.assertIsInstance(traced_llm, BaseChatModel)
        self.assertIs(resolve_model(traced_llm), traced_llm)

    def test_auto_trace_llm_sets_reserved_mlflow_token_usage_fields(self) -> None:
        traced_llm = auto_trace_llm(
            _FakeRunnable(
                AIMessage(
                    content="ok",
                    usage_metadata={
                        "input_tokens": 5,
                        "output_tokens": 7,
                        "total_tokens": 12,
                    },
                )
            ),
            session_id="session-token-usage",
            trace_name="token-usage-trace",
        )

        result = asyncio.run(traced_llm.ainvoke({"question": "hello"}))

        self.assertEqual(result.content, "ok")
        self.assertEqual(
            self.fake_mlflow.calls[-1]["metadata"]["mlflow.trace.tokenUsage"],
            '{"input_tokens":5,"output_tokens":7,"total_tokens":12}',
        )
        self.assertEqual(self.fake_mlflow.calls[-1]["metadata"]["input_tokens"], "5")
        self.assertEqual(self.fake_mlflow.calls[-1]["metadata"]["output_tokens"], "7")
        self.assertEqual(self.fake_mlflow.calls[-1]["metadata"]["total_tokens"], "12")
        span = self.fake_mlflow.started_spans[-1]["context"]
        self.assertEqual(
            span.attributes["mlflow.chat.tokenUsage"],
            {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12},
        )

    def test_auto_trace_llm_records_generate_path_token_usage(self) -> None:
        traced_llm = auto_trace_llm(
            _StaticChatModel(
                response_text="generated",
                model_name="generate-model",
                usage_metadata={
                    "input_tokens": 8,
                    "output_tokens": 9,
                    "total_tokens": 17,
                },
            ),
            session_id="session-generate-token-usage",
            trace_name="generate-token-usage",
        )

        result = traced_llm.generate([[HumanMessage(content="hello")]])

        self.assertEqual(result.generations[0][0].message.content, "generated")
        self.assertEqual(
            self.fake_mlflow.calls[-1]["metadata"]["mlflow.trace.tokenUsage"],
            '{"input_tokens":8,"output_tokens":9,"total_tokens":17}',
        )
        span = self.fake_mlflow.started_spans[-1]["context"]
        self.assertEqual(span.attributes["mlflow.spanType"], "CHAT_MODEL")
        self.assertEqual(
            span.attributes["mlflow.chat.tokenUsage"],
            {"input_tokens": 8, "output_tokens": 9, "total_tokens": 17},
        )

    def test_auto_trace_chain_wraps_raw_ainvoke(self) -> None:
        traced_chain = auto_trace_chain(
            _FakeRunnable({"answer": "ok"}),
            user_id="user-chain",
            session_id="session-chain",
            trace_name="auto-chain",
        )

        result = asyncio.run(traced_chain.ainvoke({"question": "hello"}))

        self.assertEqual(result, {"answer": "ok"})
        self.assertEqual(self.fake_mlflow.started_spans[0]["name"], "auto-chain")
        self.assertEqual(
            self.fake_mlflow.calls[-1]["metadata"]["mlflow.trace.session"],
            "session-chain",
        )

    def test_auto_trace_chain_sets_reserved_mlflow_token_usage_fields(self) -> None:
        traced_chain = auto_trace_chain(
            _FakeRunnable(
                {
                    "answer": "ok",
                    "usage_metadata": {
                        "input_tokens": 3,
                        "output_tokens": 4,
                        "total_tokens": 7,
                    },
                }
            ),
            session_id="session-chain-token-usage",
            trace_name="chain-token-usage",
        )

        result = asyncio.run(traced_chain.ainvoke({"question": "hello"}))

        self.assertEqual(result["answer"], "ok")
        self.assertEqual(
            self.fake_mlflow.calls[-1]["metadata"]["mlflow.trace.tokenUsage"],
            '{"input_tokens":3,"output_tokens":4,"total_tokens":7}',
        )
        span = self.fake_mlflow.started_spans[-1]["context"]
        self.assertEqual(
            span.attributes["mlflow.chat.tokenUsage"],
            {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
        )

    def test_trace_llm_supports_async_context_manager(self) -> None:
        async def _run():
            with_trace = trace_llm(
                user_id="user-31",
                session_id="session-31",
                trace_name="context-trace",
            )
            self.assertIsInstance(with_trace, TraceSession)
            async with with_trace as trace:
                trace.set_request({"question": "hello"})
                response = await _FakeRunnable({"answer": "ok"}).ainvoke({"question": "hello"})
                trace.set_response(response)
                return response

        result = asyncio.run(_run())

        self.assertEqual(result, {"answer": "ok"})
        self.assertEqual(self.fake_mlflow.started_spans[0]["name"], "context-trace")
        self.assertEqual(self.fake_mlflow.calls[-1]["response_preview"], "ok")
        span = self.fake_mlflow.started_spans[-1]["context"]
        self.assertEqual(span.inputs, {"question": "hello"})
        self.assertEqual(span.outputs, {"answer": "ok"})

    def test_trace_llm_can_add_tags_and_metadata_after_llm_call(self) -> None:
        llm = _FakeRunnable({"content": "ok"})

        async def _run():
            async with trace_llm(
                user_id="user-31",
                session_id="session-31",
                tags={"app": "support-bot"},
                metadata={"app_version": "0.1.0"},
                trace_name="context-trace-update",
            ) as trace:
                trace.set_request({"question": "hello"})
                response = await llm.ainvoke({"question": "hello"})
                trace.add_tags({"model": llm.model})
                trace.add_metadata(
                    {
                        "model_name": llm.model,
                        "response_type": type(response).__name__,
                    }
                )
                trace.set_response(response)
                return response

        result = asyncio.run(_run())

        self.assertEqual(result, {"content": "ok"})
        self.assertEqual(self.fake_mlflow.calls[-1]["tags"]["app"], "support-bot")
        self.assertEqual(self.fake_mlflow.calls[-1]["tags"]["model"], "fake-llm")
        self.assertEqual(self.fake_mlflow.calls[-1]["metadata"]["app_version"], "0.1.0")
        self.assertEqual(self.fake_mlflow.calls[-1]["metadata"]["model_name"], "fake-llm")
        self.assertEqual(self.fake_mlflow.calls[-1]["metadata"]["response_type"], "dict")

    def test_trace_llm_call_wraps_async_function(self) -> None:
        runnable = _FakeRunnable({"answer": "ok"})

        @trace_llm_call(
            user_id="user-32",
            session_id="session-32",
            trace_name="decorated-trace",
        )
        async def agent(llm, input):
            return await llm.ainvoke(input)

        result = asyncio.run(agent(runnable, {"question": "hello"}))

        self.assertEqual(result, {"answer": "ok"})
        self.assertEqual(self.fake_mlflow.started_spans[0]["name"], "decorated-trace")
        self.assertEqual(self.fake_mlflow.calls[-1]["request_preview"], "hello")
        self.assertEqual(self.fake_mlflow.calls[-1]["response_preview"], "ok")

    def test_trace_llm_call_supports_custom_request_resolver(self) -> None:
        runnable = _FakeRunnable({"answer": "ok"})

        @trace_llm_call(
            user_id="user-33",
            request_resolver=lambda llm, payload, extra=None: payload["actual_input"],
        )
        async def agent(llm, payload, extra=None):
            return await llm.ainvoke(payload["actual_input"])

        result = asyncio.run(
            agent(
                runnable,
                {"actual_input": {"question": "hello"}},
                extra="ignored",
            )
        )

        self.assertEqual(result, {"answer": "ok"})
        self.assertEqual(self.fake_mlflow.calls[-1]["request_preview"], "hello")

    def test_auto_trace_agent_nests_manual_subagent_and_llm_spans_in_one_trace(self) -> None:
        session_id = "session-agent"
        user_id = "user-agent"

        child_model = auto_trace_llm(
            _StaticChatModel(
                response_text="child-answer",
                model_name="child-test-model",
                usage_metadata={
                    "input_tokens": 4,
                    "output_tokens": 6,
                    "total_tokens": 10,
                },
            ),
            session_id=session_id,
            user_id=user_id,
            trace_name="child-model",
        )
        child_agent = auto_trace_agent(
            create_agent(model=child_model, tools=[], name="billing-subagent"),
            session_id=session_id,
            user_id=user_id,
        )

        async def ask_child_agent(question: str) -> str:
            """Delegate the question to the child agent."""
            result = await child_agent.ainvoke(
                {"messages": [{"role": "user", "content": question}]}
            )
            return result["messages"][-1].content

        supervisor_model = auto_trace_llm(
            _SupervisorToolCallingChatModel(tool_name="ask_child_agent"),
            session_id=session_id,
            user_id=user_id,
            trace_name="supervisor-model",
        )
        supervisor_agent = auto_trace_agent(
            create_agent(
                model=supervisor_model,
                tools=[ask_child_agent],
                name="supervisor-agent",
            ),
            session_id=session_id,
            user_id=user_id,
        )

        result = asyncio.run(
            supervisor_agent.ainvoke(
                {"messages": [{"role": "user", "content": "Please delegate this"}]}
            )
        )

        self.assertEqual(result["messages"][-1].content, "supervisor-final")
        span_names = [span["name"] for span in self.fake_mlflow.started_spans]
        self.assertIn("supervisor-agent", span_names)
        self.assertIn("billing-subagent", span_names)
        self.assertIn("supervisor-model", span_names)
        self.assertIn("child-model", span_names)
        self.assertEqual({span["trace_id"] for span in self.fake_mlflow.started_spans}, {1})

        supervisor_span = next(
            span for span in self.fake_mlflow.started_spans if span["name"] == "supervisor-agent"
        )
        subagent_span = next(
            span for span in self.fake_mlflow.started_spans if span["name"] == "billing-subagent"
        )
        self.assertEqual(subagent_span["parent_span_id"], supervisor_span["span_id"])
        self.assertEqual(subagent_span["context"].attributes["agent_name"], "billing-subagent")
        self.assertEqual(subagent_span["context"].attributes["agent_type"], "subagent")
        self.assertEqual(subagent_span["context"].attributes["mlflow.spanType"], "AGENT")
        self.assertEqual(
            subagent_span["context"].attributes["parent_agent_name"],
            "supervisor-agent",
        )
        self.assertEqual(subagent_span["context"].attributes["model_name"], "child-test-model")
        self.assertEqual(
            subagent_span["context"].attributes["mlflow.llm.model"],
            "child-test-model",
        )
        child_model_span = next(
            span for span in self.fake_mlflow.started_spans if span["name"] == "child-model"
        )
        self.assertEqual(child_model_span["context"].attributes["mlflow.spanType"], "CHAT_MODEL")
        self.assertEqual(
            subagent_span["context"].attributes["mlflow.chat.tokenUsage"],
            {"input_tokens": 4, "output_tokens": 6, "total_tokens": 10},
        )
        self.assertEqual(subagent_span["context"].attributes["input_tokens"], 4)
        self.assertEqual(subagent_span["context"].attributes["output_tokens"], 6)
        self.assertEqual(subagent_span["context"].attributes["total_tokens"], 10)
        self.assertEqual(
            subagent_span["context"].outputs["messages"][-1]["content"],
            "child-answer",
        )
        self.assertEqual(subagent_span["context"].outputs["content"], "child-answer")
        self.assertEqual(
            subagent_span["context"].outputs["response_metadata"]["model"],
            "child-test-model",
        )
        self.assertEqual(
            subagent_span["context"].outputs["usage_metadata"],
            {"input_tokens": 4, "output_tokens": 6, "total_tokens": 10},
        )
        self.assertEqual(self.fake_mlflow.calls[-1]["metadata"]["agent_name"], "supervisor-agent")

    def test_auto_trace_agent_keeps_parallel_child_calls_in_same_trace(self) -> None:
        session_id = "session-parallel"
        user_id = "user-parallel"
        child_agent = auto_trace_agent(
            create_agent(
                model=auto_trace_llm(
                    _StaticChatModel(
                        response_text="parallel-child",
                        model_name="parallel-child-model-name",
                        usage_metadata={
                            "input_tokens": 2,
                            "output_tokens": 3,
                            "total_tokens": 5,
                        },
                    ),
                    session_id=session_id,
                    user_id=user_id,
                    trace_name="parallel-child-model",
                ),
                tools=[],
                name="billing-subagent",
            ),
            session_id=session_id,
            user_id=user_id,
        )

        async def _run() -> Any:
            with using_root_trace(
                session_id=session_id,
                user_id=user_id,
                trace_name="supervisor-root",
            ) as trace:
                trace.request = {"messages": [{"role": "user", "content": "run in parallel"}]}
                result = await asyncio.gather(
                    child_agent.ainvoke(
                        {"messages": [{"role": "user", "content": "first"}]}
                    ),
                    child_agent.ainvoke(
                        {"messages": [{"role": "user", "content": "second"}]}
                    ),
                )
                trace.response = result
                return result

        results = asyncio.run(_run())

        self.assertEqual(len(results), 2)
        root_span = next(
            span for span in self.fake_mlflow.started_spans if span["name"] == "supervisor-root"
        )
        child_spans = [
            span for span in self.fake_mlflow.started_spans if span["name"] == "billing-subagent"
        ]
        self.assertEqual(len(child_spans), 2)
        self.assertTrue(all(span["trace_id"] == root_span["trace_id"] for span in child_spans))
        self.assertTrue(all(span["parent_span_id"] == root_span["span_id"] for span in child_spans))
        self.assertEqual(
            root_span["context"].attributes["mlflow.chat.tokenUsage"],
            {"input_tokens": 4, "output_tokens": 6, "total_tokens": 10},
        )
        self.assertEqual(root_span["context"].attributes["input_tokens"], 4)
        self.assertEqual(root_span["context"].attributes["output_tokens"], 6)
        self.assertEqual(root_span["context"].attributes["total_tokens"], 10)
        self.assertEqual(
            self.fake_mlflow.calls[-1]["metadata"]["mlflow.trace.tokenUsage"],
            '{"input_tokens":4,"output_tokens":6,"total_tokens":10}',
        )
        ledger = self.fake_mlflow.calls[-1]["metadata"]["mlflow_langchain_enrichment.tokenUsageByCall"]
        parsed_ledger = json.loads(ledger)
        self.assertEqual(len(parsed_ledger), 2)
        self.assertTrue(all(key.startswith("call_") for key in parsed_ledger))
        self.assertTrue(
            all(
                entry == {
                    "input_tokens": 2,
                    "output_tokens": 3,
                    "total_tokens": 5,
                    "span_name": "parallel-child-model",
                    "model_name": "parallel-child-model-name",
                }
                for entry in parsed_ledger.values()
            )
        )
        self.assertEqual(self.fake_mlflow.calls[-1]["metadata"]["input_tokens"], "4")
        self.assertEqual(self.fake_mlflow.calls[-1]["metadata"]["output_tokens"], "6")
        self.assertEqual(self.fake_mlflow.calls[-1]["metadata"]["total_tokens"], "10")


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

    def test_auto_trace_agent_retrofits_deepagents_subagent_graphs_and_model_retries(self) -> None:
        deep_agent = create_deep_agent(
            model=_StaticChatModel(response_text="supervisor"),
            tools=[],
            subagents=[
                {
                    "name": "research-subagent",
                    "description": "Research worker",
                    "system_prompt": "Do research",
                    "model": _StaticChatModel(response_text="research"),
                    "tools": [],
                }
            ],
            name="deep-supervisor",
        )

        traced_agent = auto_trace_agent(
            deep_agent,
            session_id="session-deep",
            user_id="user-deep",
        )

        self.assertEqual(
            traced_agent.runnable.builder.nodes["model"].retry_policy.max_attempts,
            3,
        )

        task_tool = traced_agent.runnable.builder.nodes["tools"].runnable._tools_by_name["task"]
        subagent_graphs = None
        for func in (getattr(task_tool.func, "__wrapped__", task_tool.func), getattr(task_tool.coroutine, "__wrapped__", task_tool.coroutine)):
            for name, cell in zip(func.__code__.co_freevars, func.__closure__ or [], strict=False):
                if name == "subagent_graphs":
                    subagent_graphs = cell.cell_contents
                    break
            if subagent_graphs is not None:
                break

        self.assertIsNotNone(subagent_graphs)
        wrapped_subagent = subagent_graphs["research-subagent"]
        self.assertEqual(type(wrapped_subagent).__name__, "TracedAgentRunnable")
        self.assertEqual(wrapped_subagent.agent_name, "research-subagent")
        self.assertEqual(
            wrapped_subagent.runnable.builder.nodes["model"].retry_policy.max_attempts,
            3,
        )

    def test_auto_trace_agent_deepagents_run_emits_named_subagent_child_span(self) -> None:
        traced_agent = auto_trace_agent(
            create_deep_agent(
                model=auto_trace_llm(
                    _DeepAgentTaskCallingChatModel(),
                    session_id="session-deep-run",
                    user_id="user-deep-run",
                    trace_name="deep-supervisor-model",
                ),
                tools=[],
                subagents=[
                    {
                        "name": "research-subagent",
                        "description": "Research worker",
                        "system_prompt": "Do research",
                        "model": auto_trace_llm(
                            _StaticChatModel(
                                response_text="research-result",
                                model_name="research-model",
                                usage_metadata={
                                    "input_tokens": 3,
                                    "output_tokens": 4,
                                    "total_tokens": 7,
                                },
                            ),
                            session_id="session-deep-run",
                            user_id="user-deep-run",
                            trace_name="research-model",
                        ),
                        "tools": [],
                    }
                ],
                name="deep-supervisor",
            ),
            session_id="session-deep-run",
            user_id="user-deep-run",
        )

        result = asyncio.run(
            traced_agent.ainvoke(
                {"messages": [{"role": "user", "content": "Please research this"}]}
            )
        )

        self.assertEqual(result["messages"][-1].content, "deepagent-final")
        supervisor_span = next(
            span for span in self.fake_mlflow.started_spans if span["name"] == "deep-supervisor"
        )
        subagent_spans = [
            span for span in self.fake_mlflow.started_spans if span["name"] == "research-subagent"
        ]
        self.assertEqual(len(subagent_spans), 2)
        outer_subagent_span = next(
            span for span in subagent_spans if span["parent_span_id"] == supervisor_span["span_id"]
        )
        inner_subagent_span = next(
            span for span in subagent_spans if span["parent_span_id"] == outer_subagent_span["span_id"]
        )
        self.assertEqual(outer_subagent_span["trace_id"], supervisor_span["trace_id"])
        self.assertEqual(outer_subagent_span["context"].attributes["agent_name"], "research-subagent")
        self.assertEqual(outer_subagent_span["context"].attributes["agent_type"], "subagent")
        self.assertEqual(outer_subagent_span["context"].attributes["mlflow.spanType"], "AGENT")
        self.assertEqual(inner_subagent_span["context"].attributes["mlflow.spanType"], "CHAT_MODEL")
        self.assertEqual(inner_subagent_span["context"].outputs["content"], "research-result")
        supervisor_model_spans = [
            span for span in self.fake_mlflow.started_spans if span["name"] == "deep-supervisor-model"
        ]
        self.assertGreaterEqual(len(supervisor_model_spans), 2)
        self.assertEqual(supervisor_model_spans[0]["context"].attributes["mlflow.spanType"], "TOOL")
        self.assertEqual(
            supervisor_model_spans[-1]["context"].attributes["mlflow.spanType"],
            "CHAT_MODEL",
        )

    def test_auto_trace_agent_retry_attempts_create_multiple_model_spans_but_one_usage_entry(self) -> None:
        flaky_model = _FlakyChatModel(
            failures_before_success=1,
            usage_metadata={
                "input_tokens": 11,
                "output_tokens": 13,
                "total_tokens": 24,
            },
        )
        traced_agent = auto_trace_agent(
            create_agent(
                model=auto_trace_llm(
                    flaky_model,
                    session_id="session-retry",
                    user_id="user-retry",
                    trace_name="flaky-model-span",
                ),
                tools=[],
                name="retry-agent",
            ),
            session_id="session-retry",
            user_id="user-retry",
            model_max_retries=3,
        )

        result = asyncio.run(
            traced_agent.ainvoke(
                {"messages": [{"role": "user", "content": "retry please"}]}
            )
        )

        self.assertEqual(result["messages"][-1].content, "recovered")
        self.assertEqual(flaky_model.attempts, 2)
        model_spans = [
            span for span in self.fake_mlflow.started_spans if span["name"] == "flaky-model-span"
        ]
        self.assertEqual(len(model_spans), 2)
        self.assertIn("error", model_spans[0]["context"].outputs)
        self.assertEqual(model_spans[1]["context"].outputs["content"], "recovered")

        ledger = json.loads(
            self.fake_mlflow.calls[-1]["metadata"]["mlflow_langchain_enrichment.tokenUsageByCall"]
        )
        self.assertEqual(len(ledger), 1)
        self.assertEqual(
            next(iter(ledger.values())),
            {
                "input_tokens": 11,
                "output_tokens": 13,
                "total_tokens": 24,
                "span_name": "flaky-model-span",
                "model_name": "flaky-model",
            },
        )


if __name__ == "__main__":
    unittest.main()
