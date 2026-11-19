"""End-to-end graph behaviour with a scripted LLM (no network)."""

import json

import pytest
from conftest import agent, call, fake_llm, route
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from langgraph_json.gateway.runtime import Conflict


async def test_supervisor_routes_tool_runs_and_finishes(make_runtime, backend):
    rt = await make_runtime(
        responses=[
            route("hmi", "open document 42"),
            call("open_item", "Opening.", itemType="Document", itemId="42"),
            "Document 42 is open.",
            route("FINISH"),
        ]
    )
    r = await rt.chat("open document 42", "s1")
    assert r["status"] == "completed"
    assert r["response"] == "Document 42 is open." and r["agent"] == "hmi"
    assert r["route"] == ["hmi"]
    assert r["tool_calls"][0]["tool_name"] == "open_item" and r["tool_calls"][0]["status"] == "ok"
    assert backend.calls() == [("POST", "/api/items/open")]
    llm = fake_llm(rt)
    # supervisor got the route tool; the agent got only its whitelist; instructions passed on
    assert llm.calls[0]["tools"] == ["route"]
    assert llm.calls[1]["tools"] == ["open_item"]
    system = llm.calls[1]["messages"][0]
    assert isinstance(system, SystemMessage) and "open document 42" in system.content
    assert r["usage"]["tokens_used"] > 0


async def test_single_agent_needs_no_routing_call(make_runtime):
    rt = await make_runtime(agents={"solo": agent("solo", ["open_item"])}, responses=["hello from solo"])
    r = await rt.chat("hi")
    assert r["response"] == "hello from solo" and r["route"] == ["solo"]
    assert len(fake_llm(rt).calls) == 1


async def test_supervisor_answers_directly(make_runtime):
    rt = await make_runtime(responses=[{"content": "Hello! How can I help?", "tool_calls": [{"name": "route", "args": {"next": "FINISH"}}]}])
    r = await rt.chat("hello")
    assert r["response"] == "Hello! How can I help?" and r["agent"] == "supervisor" and r["route"] == []


async def test_supervisor_answer_field_wins_over_empty_text(make_runtime):
    # models that think between tool calls may return no visible text: the reply is in `answer`
    rt = await make_runtime(responses=[{"content": "", "tool_calls": [{"name": "route", "args": {"next": "FINISH", "answer": "PINEAPPLE"}}]}])
    r = await rt.chat("what was the code word?")
    assert r["response"] == "PINEAPPLE" and r["agent"] == "supervisor"


async def test_supervisor_unknown_route_finishes(make_runtime):
    rt = await make_runtime(responses=[route("ghost")])
    r = await rt.chat("hello")
    assert r["status"] == "completed" and r["route"] == []


async def test_multi_agent_chain_via_supervisor(make_runtime, backend):
    rt = await make_runtime(
        responses=[
            route("hmi"),
            "hmi done",
            route("jobs"),
            call("get_job_status", jobId="J1"),
            "job J1 is done",
            route("FINISH"),
        ]
    )
    r = await rt.chat("open then check job")
    assert r["route"] == ["hmi", "jobs"]
    assert r["response"] == "job J1 is done"
    assert backend.calls() == [("GET", "/api/jobs/J1")]
    # the jobs agent saw the hmi answer as text, not as its own message
    jobs_call = fake_llm(rt).calls[3]
    assert any(isinstance(m, HumanMessage) and m.content == "[hmi] hmi done" for m in jobs_call["messages"])


async def test_tool_not_in_whitelist_is_refused(make_runtime, backend):
    rt = await make_runtime(responses=[route("hmi"), call("run_job", jobId="J1"), "sorry", route("FINISH")])
    r = await rt.chat("run job")
    assert backend.calls() == []
    tool_msgs = [m for m in r["messages"] if m["type"] == "tool"]
    assert tool_msgs[0]["status"] == "error" and "not permitted" in tool_msgs[0]["content"]


async def test_invalid_tool_arguments_are_returned_to_llm(make_runtime, backend):
    rt = await make_runtime(responses=[route("hmi"), call("open_item", itemType="Bogus", itemId="1"), "fixed", route("FINISH")])
    r = await rt.chat("open")
    assert backend.calls() == []
    tool = next(m for m in r["messages"] if m["type"] == "tool")
    assert tool["status"] == "error" and "Error" in tool["content"]
    assert r["response"] == "fixed"


async def test_backend_error_is_returned_to_llm(make_runtime, backend):
    import httpx

    backend.responses[("GET", "/api/jobs/J1")] = httpx.Response(500, text="boom")
    rt = await make_runtime(responses=[route("jobs"), call("get_job_status", jobId="J1"), "the backend failed", route("FINISH")])
    r = await rt.chat("status")
    assert r["tool_calls"][0]["status"] == "error"
    assert r["response"] == "the backend failed"


# --------------------------------------------------------------------------- #
# Human-in-the-loop
# --------------------------------------------------------------------------- #
async def test_confirmation_approved(make_runtime, backend):
    rt = await make_runtime(responses=[route("jobs"), call("run_job", "Starting.", jobId="J7"), "Job J7 started.", route("FINISH")])
    r = await rt.chat("run job J7", "s")
    assert r["status"] == "confirmation_needed"
    c = r["confirmation"]
    assert c["tool_name"] == "run_job" and c["tool_args"] == {"jobId": "J7"} and c["agent_name"] == "jobs"
    assert backend.calls() == []

    with pytest.raises(Conflict):
        await rt.chat("another message", "s")

    r2 = await rt.confirm("s", approved=True)
    assert r2["status"] == "completed" and r2["response"] == "Job J7 started."
    assert backend.calls() == [("POST", "/api/jobs/J7/run")]
    with pytest.raises(Conflict):
        await rt.confirm("s", approved=True)


async def test_confirmation_denied(make_runtime, backend):
    rt = await make_runtime(responses=[route("jobs"), call("run_job", jobId="J7"), "OK, not started.", route("FINISH")])
    await rt.chat("run", "s")
    r = await rt.confirm("s", approved=False, reason="too expensive")
    assert backend.calls() == []
    tool = next(m for m in r["messages"] if m["type"] == "tool")
    assert "denied" in tool["content"] and "too expensive" in tool["content"]
    assert r["response"] == "OK, not started."


async def test_two_confirmations_in_one_step(make_runtime, backend):
    rt = await make_runtime(
        responses=[
            route("jobs"),
            {"tool_calls": [{"name": "run_job", "args": {"jobId": "A"}}, {"name": "get_job_status", "args": {"jobId": "A"}}, {"name": "run_job", "args": {"jobId": "B"}}]},
            "done",
            route("FINISH"),
        ]
    )
    r = await rt.chat("run A and B", "s")
    assert r["confirmation"]["tool_args"] == {"jobId": "A"}
    r = await rt.confirm("s", True)
    assert r["status"] == "confirmation_needed" and r["confirmation"]["tool_args"] == {"jobId": "B"}
    assert backend.calls() == []  # nothing runs before every decision is known
    r = await rt.confirm("s", False)
    assert r["status"] == "completed"
    assert backend.calls() == [("POST", "/api/jobs/A/run"), ("GET", "/api/jobs/A")]


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
async def test_max_tool_calls_per_turn(make_runtime, backend):
    loop = [call("get_job_status", jobId=str(i)) for i in range(10)]
    rt = await make_runtime(server={"max_tool_calls_per_turn": 3}, responses=[route("jobs"), *loop])
    r = await rt.chat("loop forever")
    assert len(backend.requests) == 3
    assert "limit of 3 tool calls" in r["response"]


async def test_token_budget(make_runtime):
    rt = await make_runtime(server={"token_budget_per_session": 30}, agents={"solo": agent("solo", ["open_item"])},
                            responses=["word " * 40, "second answer"])
    r1 = await rt.chat("first", "s")
    assert r1["status"] == "completed" and r1["usage"]["tokens_used"] >= 30
    r2 = await rt.chat("second", "s")
    assert r2["status"] == "budget_exceeded"
    assert len(fake_llm(rt).calls) == 1


async def test_supervisor_max_steps(make_runtime):
    script = []
    for _ in range(5):
        script += [route("hmi"), "again"]
    rt = await make_runtime(responses=script, supervisor={"max_steps": 2})
    r = await rt.chat("loop")
    assert r["route"] == ["hmi"]  # route list is de-duplicated when consecutive
    hmi_answers = [m for m in r["messages"] if m["type"] == "ai" and m["name"] == "hmi"]
    assert len(hmi_answers) == 2


# --------------------------------------------------------------------------- #
# History, handoff, direct mode, sub-graphs
# --------------------------------------------------------------------------- #
async def test_history_is_kept_per_session_and_stateless_agent(make_runtime):
    agents = {"mem": agent("mem", ["open_item"]), "stateless": agent("stateless", ["open_item"], max_history_messages=0)}
    rt = await make_runtime(agents=agents, responses=[route("mem"), "one", route("FINISH"), route("mem"), "two", route("FINISH"),
                                                      route("stateless"), "three", route("FINISH")])
    await rt.chat("first", "s")
    await rt.chat("second", "s")
    calls = fake_llm(rt).calls
    mem_second = [m.content for m in calls[4]["messages"]]
    assert "first" in mem_second and "second" in mem_second
    await rt.chat("third", "s")
    stateless = [m.content for m in calls[7]["messages"] if isinstance(m, HumanMessage)]
    assert stateless == ["third"]
    # other sessions are isolated
    rt2_hist = await rt.history("s")
    assert len([m for m in rt2_hist if m["type"] == "human"]) == 3


async def test_handoff_between_agents(make_runtime, backend):
    agents = {
        "hmi": agent("hmi", ["open_item"], can_handoff_to=["jobs"]),
        "jobs": agent("jobs", ["get_job_status"]),
    }
    rt = await make_runtime(agents=agents, responses=[route("hmi"), call("handoff_to_jobs", reason="needs job status"),
                                                      call("get_job_status", jobId="J"), "status ok", route("FINISH")])
    r = await rt.chat("x")
    assert r["route"] == ["hmi", "jobs"]
    assert r["response"] == "status ok" and r["agent"] == "jobs"
    calls = fake_llm(rt).calls
    assert calls[1]["tools"] == ["open_item", "handoff_to_jobs"]
    assert calls[2]["tools"] == ["get_job_status"]
    # no supervisor call between the handoff and the jobs agent
    assert len(calls) == 5


async def test_direct_agent_mode_bypasses_supervisor(make_runtime):
    rt = await make_runtime(responses=["direct answer"])
    r = await rt.chat("hi", agent="jobs")
    assert r["response"] == "direct answer" and r["route"] == ["jobs"]
    assert fake_llm(rt).calls[0]["tools"] == ["get_job_status", "run_job"]
    with pytest.raises(ValueError):
        await rt.chat("hi", agent="ghost")


async def test_sub_graph_agent(make_runtime, backend):
    agents = {
        "team": {"type": "sub_graph", "description": "team lead", "system_prompt": "coordinate", "sub_agents": ["worker_a", "worker_b"]},
        "worker_a": agent("worker a", ["open_item"], routable=False),
        "worker_b": agent("worker b", ["get_job_status"], routable=False),
        "other": agent("other", ["open_item"]),
    }
    rt = await make_runtime(
        agents=agents,
        responses=[
            route("team"),  # top supervisor
            route("worker_a"),  # team supervisor
            call("open_item", itemType="Project", itemId="P"),
            "a done",
            route("worker_b"),
            call("get_job_status", jobId="J"),
            "b done",
            route("FINISH"),  # team supervisor
            route("FINISH"),  # top supervisor
        ],
    )
    assert rt.info.top_level == ["team", "other"]
    r = await rt.chat("do the team thing", "s")
    assert r["status"] == "completed" and r["response"] == "b done"
    assert backend.calls() == [("POST", "/api/items/open"), ("GET", "/api/jobs/J")]
    assert r["route"][0] == "team" and {"worker_a", "worker_b"} <= set(r["route"])
    tokens_after = r["usage"]["tokens_used"]
    total = sum(1 for _ in fake_llm(rt).calls)
    assert total == 9 and tokens_after > 0


async def test_sub_graph_with_confirmation(make_runtime, backend):
    agents = {
        "team": {"type": "sub_graph", "description": "team", "system_prompt": "coordinate", "sub_agents": ["runner"]},
        "runner": agent("runner", ["run_job"], requires_confirmation=["run_job"], routable=False),
        "other": agent("other", ["open_item"]),
    }
    rt = await make_runtime(agents=agents, responses=[route("team"), call("run_job", jobId="Z"), "ran Z", route("FINISH")])
    r = await rt.chat("run Z", "s")
    assert r["status"] == "confirmation_needed" and r["confirmation"]["agent_name"] == "runner"
    r = await rt.confirm("s", True)
    assert r["status"] == "completed" and r["response"] == "ran Z"
    assert backend.calls() == [("POST", "/api/jobs/Z/run")]


async def test_graph_state_survives_hot_reload(make_runtime):
    rt = await make_runtime(agents={"solo": agent("solo", ["open_item"])}, responses=["one", "two"])
    await rt.chat("first", "s")
    cfg = rt.config_dir / "agents.json"
    data = json.loads(cfg.read_text())
    data["agents"]["solo"]["system_prompt"] = "You are reloaded."
    cfg.write_text(json.dumps(data))
    rt.reload()
    await rt.chat("second", "s")
    last = fake_llm(rt).calls[-1]["messages"]
    assert last[0].content.startswith("You are reloaded.")
    assert [m.content for m in last if isinstance(m, HumanMessage)] == ["first", "second"]


async def test_tool_messages_follow_tool_calls(make_runtime):
    """Every ToolMessage sent to the LLM answers a tool call of the same agent."""
    rt = await make_runtime(responses=[route("jobs"), call("get_job_status", jobId="1"), "ok", route("FINISH")])
    await rt.chat("x")
    for c in fake_llm(rt).calls:
        own_ids = set()
        for m in c["messages"]:
            if getattr(m, "tool_calls", None):
                own_ids |= {tc["id"] for tc in m.tool_calls}
            if isinstance(m, ToolMessage):
                assert m.tool_call_id in own_ids
