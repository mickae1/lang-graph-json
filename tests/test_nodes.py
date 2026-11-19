from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langgraph_json.utils.nodes import (
    answered_this_turn,
    history_window,
    message_text,
    render_for_agent,
    strip_thinking,
    transcript,
)


def tc(name, cid, **args):
    return {"name": name, "args": args, "id": cid, "type": "tool_call"}


CONV = [
    HumanMessage(content="first question", id="h1"),
    AIMessage(content="", name="a", tool_calls=[tc("t1", "c1", x=1)], id="a1"),
    ToolMessage(content="result-1", tool_call_id="c1", name="t1", id="t1"),
    AIMessage(content="answer from a", name="a", id="a2"),
    HumanMessage(content="second question", id="h2"),
    AIMessage(content="", name="b", tool_calls=[tc("t2", "c2")], id="b1"),
    ToolMessage(content="result-2", tool_call_id="c2", name="t2", id="t2"),
    AIMessage(content="answer from b", name="b", id="b2"),
]


def ids(msgs):
    return [m.id for m in msgs]


def test_history_window_keeps_current_turn_and_starts_on_user():
    assert ids(history_window(CONV, 0)) == ["h2", "b1", "t2", "b2"]
    # 2 prior messages would start on a ToolMessage/AIMessage: trimmed to the next user message
    assert ids(history_window(CONV, 2)) == ["h2", "b1", "t2", "b2"]
    assert ids(history_window(CONV, 4)) == ids(CONV)
    assert ids(history_window(CONV, 100)) == ids(CONV)


def test_render_own_messages_native_others_as_text():
    out = render_for_agent(CONV, "b", 40)
    # agent b sees its own tool call natively
    assert any(isinstance(m, AIMessage) and m.id == "b1" for m in out)
    assert any(isinstance(m, ToolMessage) and m.tool_call_id == "c2" for m in out)
    # agent a's tool call/result become text
    assert not any(isinstance(m, ToolMessage) and m.tool_call_id == "c1" for m in out)
    texts = [m.content for m in out if isinstance(m, HumanMessage)]
    assert any("[a] (called tool t1" in t for t in texts)
    assert any("tool t1 returned: result-1" in t for t in texts)
    assert any(t == "[a] answer from a" for t in texts)
    # ends with a user message (b's own final answer is last -> continue prompt)
    assert isinstance(out[-1], HumanMessage) and "Continue" in out[-1].content


def test_render_for_other_agent_ends_with_user_message():
    out = render_for_agent(CONV, "c", 0)
    assert all(isinstance(m, HumanMessage) for m in out)
    assert out[-1].content == "[b] answer from b"


def test_transcript():
    text = transcript(CONV, 40)
    assert "User: first question" in text and "[b]: answer from b" in text
    assert "[a] called t1" in text and "-> t2 result: result-2" in text


def test_answered_this_turn():
    assert answered_this_turn(CONV)
    assert not answered_this_turn(CONV[:7])
    assert not answered_this_turn([HumanMessage(content="x")])


def test_message_text_handles_blocks():
    assert message_text([{"type": "text", "text": "a"}, {"type": "tool_use"}, "b"]) == "ab"
    assert message_text(None) == ""


THINK = {"type": "thinking", "thinking": "", "signature": "sig"}


def test_thinking_blocks_only_replayed_in_the_tool_loop_in_progress():
    old = AIMessage(content=[THINK, {"type": "text", "text": "done before"}], name="a", id="old")
    only_thinking = AIMessage(content=[{"type": "redacted_thinking", "data": "x"}], name="a", id="empty")
    loop = AIMessage(content=[THINK], name="a", tool_calls=[tc("t1", "c9")], id="loop")
    msgs = [
        HumanMessage(content="q1", id="h1"), old, only_thinking,
        HumanMessage(content="q2", id="h2"), loop,
        ToolMessage(content="r", tool_call_id="c9", name="t1", id="r"),
    ]
    out = render_for_agent(msgs, "a", 10)
    assert ids(out) == ["h1", "old", "h2", "loop", "r"]  # the reasoning-only message is dropped
    assert out[1].content == [{"type": "text", "text": "done before"}]
    assert out[3].content == [THINK]  # in-progress loop: kept unchanged


def test_strip_thinking_keeps_tool_calls_and_plain_messages():
    m = AIMessage(content=[THINK], name="a", tool_calls=[tc("t1", "c1")])
    assert strip_thinking(m).content == [] and strip_thinking(m).tool_calls
    plain = AIMessage(content="hi")
    assert strip_thinking(plain) is plain
