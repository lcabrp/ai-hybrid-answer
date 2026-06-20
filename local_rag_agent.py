from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from typing import Any

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
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


def build_tools_schema() -> list[dict[str, Any]]:
    """
    Return OpenAI-compatible tool JSON schemas for llama.cpp tool calling.

    llama-server exposes an OpenAI-compatible endpoint, so we provide tools
    as `type=function` objects with JSON-schema parameter definitions.

    Schema layout used by both OpenAI and llama.cpp-compatible servers:
    - `type`: must be `function` for function/tool calls.
    - `function.name`: the Python-dispatch key used in `_execute_tool`.
    - `function.description`: guidance to help the model choose a tool.
    - `function.parameters`: JSON Schema object used by the model to build
      validated arguments (properties/required/default/min/max, etc.).

    The model emits arguments as a JSON string in `tool_calls[*].function.arguments`.
    We then parse and execute those arguments locally in Python.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for recent information using DuckDuckGo.",
                # JSON Schema: object with one required property (`query`)
                # and one optional bounded property (`max_results`).
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query.",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of search results to return.",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_fetch",
                "description": "Fetch and extract readable text from a URL.",
                # JSON Schema: object with required URL and optional output cap.
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to download and extract text from.",
                        },
                        "max_chars": {
                            "type": "integer",
                            "description": "Max number of characters to return.",
                            "default": 8000,
                            "minimum": 500,
                            "maximum": 30000,
                        },
                    },
                    "required": ["url"],
                },
            },
        },
    ]


def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    max_results = max(1, min(max_results, 10))
    results: list[dict[str, str]] = []
    with DDGS() as ddgs:
        for row in ddgs.text(query, max_results=max_results):
            results.append(
                {
                    "title": row.get("title", ""),
                    "url": row.get("href", ""),
                    "snippet": row.get("body", ""),
                }
            )
    return {"query": query, "results": results}


def web_fetch(url: str, max_chars: int = 8000) -> dict[str, Any]:
    max_chars = max(500, min(max_chars, 30000))
    response = requests.get(
        url,
        timeout=15,
        headers={"User-Agent": "local-rag-agent/1.0"},
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    text = " ".join(soup.get_text(separator=" ").split())
    return {"url": url, "content": text[:max_chars]}


def _execute_tool(function_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if function_name == "web_search":
        return web_search(
            query=arguments.get("query", ""),
            max_results=int(arguments.get("max_results", 5)),
        )
    if function_name == "web_fetch":
        return web_fetch(
            url=arguments.get("url", ""),
            max_chars=int(arguments.get("max_chars", 8000)),
        )
    raise ValueError(f"Unknown tool requested: {function_name}")


def _safe_parse_json(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def run_agent(
    user_input: str,
    client: OpenAI,
    model: str,
    max_rounds: int = 6,
    now: datetime | None = None,
) -> str:
    """
    Core agent loop:
    1) Ask llama.cpp
    2) If it returns tool calls, execute them
    3) Append tool outputs and ask again
    4) Return final grounded answer
    """
    tools = build_tools_schema()
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

        # Tool callback handling:
        # Each model tool call is parsed, executed in Python, then written back
        # as a `tool` message with matching `tool_call_id` for the next round.
        for call in tool_calls:
            function_name = call.function.name
            arguments = _safe_parse_json(call.function.arguments)
            try:
                result = _execute_tool(function_name, arguments)
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
        f"Last message role={messages[-1].get('role')!r}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local llama.cpp + DuckDuckGo tool-calling agent"
    )
    parser.add_argument("prompt", help="User question for the assistant")
    parser.add_argument("--model", default="gemma", help="Model name served by llama.cpp")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080/v1",
        help="llama-server OpenAI-compatible base URL",
    )
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="local")
    answer = run_agent(user_input=args.prompt, client=client, model=args.model)
    print(answer)


if __name__ == "__main__":
    main()
