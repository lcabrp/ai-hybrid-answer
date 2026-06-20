from __future__ import annotations

from typing import Any

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Web Tools MCP")


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search the web for recent information using DuckDuckGo."""
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


@mcp.tool()
def web_fetch(url: str, max_chars: int = 8000) -> dict[str, Any]:
    """Fetch and extract readable text from a URL."""
    max_chars = max(500, min(max_chars, 30000))
    response = requests.get(
        url,
        timeout=15,
        headers={"User-Agent": "local-rag-agent-mcp/1.0"},
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    text = " ".join(soup.get_text(separator=" ").split())
    return {"url": url, "content": text[:max_chars]}


if __name__ == "__main__":
    import asyncio

    asyncio.run(mcp.run_stdio_async())
