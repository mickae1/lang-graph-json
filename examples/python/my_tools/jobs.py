"""A Python tool orchestrating several REST calls and returning a compact
summary (the LLM sees one small result instead of several big ones)."""

from langgraph_json.metadata import AI_METADATA, ToolContext


@AI_METADATA({
    "name": "jobs_overview",
    "domain": "jobs",
    "read_only": True,
    "description": "Returns the state of several jobs in one call.",
    "parameters": {
        "job_ids": {"type": "array", "items": {"type": "string"}, "required": True},
    },
})
async def jobs_overview(job_ids: list[str], ctx: ToolContext) -> dict:
    states: dict[str, str] = {}
    for job_id in job_ids:
        status = await ctx.client.request("GET", f"/api/jobs/{job_id}")
        states[job_id] = status.get("state", "unknown") if isinstance(status, dict) else str(status)
    done = sum(1 for s in states.values() if s == "done")
    return {"total": len(states), "done": done, "states": states}
