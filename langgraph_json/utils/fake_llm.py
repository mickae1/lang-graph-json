"""Scripted chat model (provider type ``fake``).

Used by the test-suite and for offline smoke tests of the packaged binary.
It supports ``bind_tools`` and token streaming, so the whole pipeline
(supervisor, tools, confirmation, WebSocket events) can be exercised without
any real LLM.

A script entry is either:

* a string — plain text answer;
* a dict ``{"content": str, "tool_calls": [{"name": ..., "args": {...}}]}``;
* a callable ``(messages, tool_names) -> AIMessage | dict | str``.

When the script is exhausted the model answers ``"echo: <last user text>"``.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Callable, Iterator

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


class FakeChatModel(BaseChatModel):
    responses: list[Any] = Field(default_factory=list)
    model_name: str = "fake-model"
    # Record of every call: {"messages": [...], "tools": [names]}
    calls: list[dict] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    @property
    def _identifying_params(self) -> dict:
        return {"model_name": self.model_name}

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        names = [convert_to_openai_tool(t)["function"]["name"] for t in tools]
        return self.bind(tool_names=names, **kwargs)

    # ------------------------------------------------------------------ #
    def _next_message(self, messages: list[BaseMessage], tool_names: list[str]) -> AIMessage:
        self.calls.append({"messages": list(messages), "tools": list(tool_names)})
        if self.responses:
            item = self.responses.pop(0)
        else:
            last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
            item = f"echo: {_text(last_human.content) if last_human else ''}"
        if callable(item):
            item = item(messages, tool_names)
        if isinstance(item, AIMessage):
            msg = item
        elif isinstance(item, str):
            msg = AIMessage(content=item)
        else:
            calls = [
                {"name": c["name"], "args": c.get("args", {}), "id": c.get("id") or f"call_{uuid.uuid4().hex[:12]}", "type": "tool_call"}
                for c in item.get("tool_calls", [])
            ]
            msg = AIMessage(content=item.get("content", ""), tool_calls=calls)
        n_in = sum(len(_text(m.content).split()) for m in messages)
        n_out = max(1, len(_text(msg.content).split())) + 5 * len(msg.tool_calls)
        msg.usage_metadata = {"input_tokens": n_in, "output_tokens": n_out, "total_tokens": n_in + n_out}
        msg.response_metadata = {"model_name": self.model_name}
        return msg

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        tool_names: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        msg = self._next_message(messages, tool_names or [])
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        tool_names: list[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        msg = self._next_message(messages, tool_names or [])
        text = _text(msg.content)
        words = text.split(" ") if text else []
        for i, w in enumerate(words):
            token = w if i == len(words) - 1 else w + " "
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token, id=msg.id))
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                id=msg.id,
                tool_call_chunks=[
                    {"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i, "type": "tool_call_chunk"}
                    for i, c in enumerate(msg.tool_calls)
                ],
                usage_metadata=msg.usage_metadata,
                response_metadata=msg.response_metadata,
                chunk_position="last",
            )
        )


ScriptItem = str | dict | Callable[[list[BaseMessage], list[str]], Any]
