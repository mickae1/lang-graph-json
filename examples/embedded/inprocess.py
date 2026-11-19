"""Use lang-graph-json as a library, in-process (no HTTP server, no JSON files).

    pip install lang-graph-json
    ANTHROPIC_API_KEY=... python examples/embedded/inprocess.py

Without ANTHROPIC_API_KEY a scripted fake model is used, so the example
always runs.
"""

import asyncio
import os

from langgraph_json import AI_METADATA, FakeChatModel, Runtime


# A Python tool: plain function + metadata (or just a docstring).
@AI_METADATA({
    "name": "get_weather",
    "description": "Returns the current weather of a city.",
    "read_only": True,
})
def get_weather(city: str) -> dict:
    return {"city": city, "temperature_c": 21, "sky": "clear"}


def add(a: float, b: float) -> float:
    """Adds two numbers."""
    return a + b


AGENTS = {
    "supervisor": {"system_prompt": "Route weather questions to weather, arithmetic to calculator."},
    "agents": {
        "weather": {
            "description": "Answers questions about the weather.",
            "system_prompt": "You answer weather questions using your tool.",
            "allowed_tools": ["get_weather"],
        },
        "calculator": {
            "description": "Does arithmetic.",
            "system_prompt": "You compute with your tool and give the result.",
            "allowed_tools": ["add"],
            "requires_confirmation": ["add"],  # human-in-the-loop works the same in-process
        },
    },
}


def make_llm():
    if os.environ.get("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=os.environ.get("LGJ_MODEL", "claude-haiku-5-5"))
    return FakeChatModel(responses=[
        {"tool_calls": [{"name": "route", "args": {"next": "weather"}}]},
        {"tool_calls": [{"name": "get_weather", "args": {"city": "Paris"}}]},
        "It is 21 °C and clear in Paris.",
        {"tool_calls": [{"name": "route", "args": {"next": "FINISH"}}]},
        {"tool_calls": [{"name": "route", "args": {"next": "calculator"}}]},
        {"tool_calls": [{"name": "add", "args": {"a": 2, "b": 3}}]},
        "2 + 3 = 5",
        {"tool_calls": [{"name": "route", "args": {"next": "FINISH"}}]},
    ])


async def main() -> None:
    async with Runtime(agents=AGENTS, functions=[get_weather, add], llm=make_llm()) as rt:
        r = await rt.chat("What's the weather in Paris?", session_id="demo")
        print(f"[{r['agent']}] {r['response']}")

        r = await rt.chat("What is 2 + 3?", session_id="demo")
        while r["status"] == "confirmation_needed":
            c = r["confirmation"]
            print(f"approve {c['tool_name']}({c['tool_args']})? -> yes")
            r = await rt.confirm("demo", approved=True)
        print(f"[{r['agent']}] {r['response']}")


if __name__ == "__main__":
    asyncio.run(main())
