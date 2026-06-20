from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from openai import OpenAI


def build_system_prompt(now: datetime | None = None) -> str:
    """Build a system prompt that always contains the exact current date/time."""
    current = now or datetime.now(timezone.utc)
    return (
        "You are a local assistant powered by Gemma running behind llama.cpp. "
        f"Current date and time (UTC): {current.isoformat()}.\n"
        "You can use tools for web_search and web_fetch. "
        "If the user asks about current events, recent news, or anything uncertain, "
        "call tools first, then answer with grounded facts and cite URLs you used."
    )


def _safe_parse_json(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _extract_tool_list(list_tools_result: Any) -> list[Any]:
    if hasattr(list_tools_result, "tools") and isinstance(list_tools_result.tools, list):
        return list_tools_result.tools
    if isinstance(list_tools_result, dict):
        tools = list_tools_result.get("tools", [])
        if isinstance(tools, list):
            return tools
    return []


def _tool_name(tool: Any) -> str:
    if isinstance(tool, dict):
        return str(tool.get("name", ""))
    return str(getattr(tool, "name", ""))


def _tool_description(tool: Any) -> str:
    if isinstance(tool, dict):
        return str(tool.get("description", ""))
    return str(getattr(tool, "description", ""))


def _tool_schema(tool: Any) -> dict[str, Any]:
    if isinstance(tool, dict):
        schema = tool.get("inputSchema") or tool.get("input_schema")
    else:
        schema = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None)

    # The OpenAI-compatible tool schema must expose JSON Schema object parameters.
    # If the MCP tool does not provide one, we use a minimal empty object schema.
    if isinstance(schema, dict) and schema.get("type") == "object":
        return schema
    return {"type": "object", "properties": {}}


def build_tools_schema_from_mcp_list(list_tools_result: Any) -> list[dict[str, Any]]:
    """
    Convert MCP list_tools output into OpenAI/llama.cpp tool-call schema.

    JSON schema mapping notes:
    - MCP `tool.inputSchema` becomes OpenAI `function.parameters`.
    - OpenAI requires a wrapper shape: {"type": "function", "function": {...}}.
    - We preserve each tool's `name` and `description` for model tool selection.
    - Required/default/min/max constraints are carried through from MCP schema.
    """
    openai_tools: list[dict[str, Any]] = []
    for tool in _extract_tool_list(list_tools_result):
        name = _tool_name(tool)
        if not name:
            continue
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": _tool_description(tool),
                    "parameters": _tool_schema(tool),
                },
            }
        )
    return openai_tools


async def _call_mcp_tool(
    session: ClientSession,
    function_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    result = await session.call_tool(function_name, arguments)

    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured

    parts: list[str] = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if isinstance(text, str) and text:
            parts.append(text)

    if len(parts) == 1:
        parsed = _safe_parse_json(parts[0])
        if parsed:
            return parsed

    payload: dict[str, Any] = {"content": parts}
    is_error = getattr(result, "isError", None)
    if isinstance(is_error, bool):
        payload["is_error"] = is_error
    return payload


async def run_agent_async(
    user_input: str,
    client: OpenAI,
    model: str,
    max_rounds: int = 6,
    now: datetime | None = None,
    server_script: str = "mcp_web_tools_server.py",
) -> str:
    """
    Run tool-calling loop using tools discovered from an MCP server.

    Flow:
    1. Spawn MCP stdio server process.
    2. Initialize MCP session and discover tools via `list_tools`.
    3. Convert MCP input schemas to OpenAI tool schemas.
    4. Run chat/tool loop where tool calls are executed via MCP `call_tool`.
    """
    script_path = Path(server_script)
    if not script_path.is_absolute():
        script_path = Path(__file__).parent / script_path

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(script_path)],
        cwd=str(script_path.parent),
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            mcp_tools = await session.list_tools()
            tools = build_tools_schema_from_mcp_list(mcp_tools)

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": build_system_prompt(now=now)},
                {"role": "user", "content": user_input},
            ]

            for _ in range(max_rounds):
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                )
                choice = response.choices[0].message

                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": choice.content or "",
                }
                tool_calls = getattr(choice, "tool_calls", None) or []
                if tool_calls:
                    assistant_message["tool_calls"] = []
                    for call in tool_calls:
                        assistant_message["tool_calls"].append(
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.function.name,
                                    "arguments": call.function.arguments or "{}",
                                },
                            }
                        )
                messages.append(assistant_message)

                if not tool_calls:
                    return choice.content or ""

                for call in tool_calls:
                    function_name = call.function.name
                    arguments = _safe_parse_json(call.function.arguments)
                    try:
                        result = await _call_mcp_tool(session, function_name, arguments)
                    except Exception as exc:  # noqa: BLE001
                        result = {
                            "error": str(exc),
                            "tool": function_name,
                            "arguments": arguments,
                            "traceback": traceback.format_exc(),
                        }

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": function_name,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )

    raise RuntimeError(
        f"Maximum tool-calling rounds exceeded ({max_rounds}). "
        "The model did not produce a final answer."
    )


def run_agent(
    user_input: str,
    client: OpenAI,
    model: str,
    max_rounds: int = 6,
    now: datetime | None = None,
    server_script: str = "mcp_web_tools_server.py",
) -> str:
    return asyncio.run(
        run_agent_async(
            user_input=user_input,
            client=client,
            model=model,
            max_rounds=max_rounds,
            now=now,
            server_script=server_script,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local llama.cpp + MCP web tools agent"
    )
    parser.add_argument("prompt", help="User question for the assistant")
    parser.add_argument("--model", default="gemma", help="Model name served by llama.cpp")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080/v1",
        help="llama-server OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--server-script",
        default="mcp_web_tools_server.py",
        help="Path to MCP server script that exposes tools",
    )
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="local")
    answer = run_agent(
        user_input=args.prompt,
        client=client,
        model=args.model,
        server_script=args.server_script,
    )
    print(answer)


if __name__ == "__main__":
    main()
