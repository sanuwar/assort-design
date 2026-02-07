Next step: wire the real LangGraph pipeline (routing → generate → evaluate → revise → persist) and replace the scaffold in graph.py.

That will also require:

Implementing real OpenAI calls + mock fallback in llm.py.
Using prompts from agent_profiles.yaml (no hardcoded prompts).
Persisting tags/questions to DB on success.
If that sounds right, I’ll start with the pipeline implementation first.


Why we only see response logs in OpenAI dashboard (not traces)
We use the OpenAI Responses API directly inside llm.py. The Responses API returns response IDs and usage, but it does not emit Trace records unless you use the Agents SDK or Realtime API with tracing enabled. So the OpenAI dashboard can show logs/usage, but Trace timelines are not available in our current integration.

For node‑level visibility in LangGraph, the correct tool is LangSmith tracing.