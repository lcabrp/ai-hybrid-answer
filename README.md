# ai-hybrid-answer
Testing an idea from a comment of a XDA article.

## Local llama.cpp + web tools agent

`local_rag_agent.py` is a Python orchestrator for a local `llama-server` endpoint
(`http://localhost:8080/v1`) that:

- injects the current UTC date/time into the system prompt,
- exposes DuckDuckGo search and URL fetch as tool-calls,
- runs a tool-calling loop until the model returns a final answer.

## MCP-backed variant

This repo also includes an MCP implementation inspired by the same idea:

- `mcp_web_tools_server.py` exposes `web_search` and `web_fetch` as MCP tools.
- `local_rag_agent_mcp.py` starts that MCP server over stdio, discovers tool
	schemas through MCP `list_tools`, converts them to OpenAI-compatible JSON
	tool schemas, then runs a llama.cpp tool-calling loop.
- The system prompt still injects the current UTC date for recency grounding.

### Setup

```bash
uv sync
```

### Usage

```bash
uv run python local_rag_agent.py "What are the latest AI safety news updates today?"
```

### Usage (MCP)

```bash
uv run python local_rag_agent_mcp.py "What are the latest AI safety news updates today?"
```
