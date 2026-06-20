import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from local_rag_agent import build_system_prompt, build_tools_schema, run_agent


def _tool_call(call_id: str, name: str, arguments: str):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _response(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeCompletions:
    def __init__(self, responses):
        self._responses = responses
        self.calls = 0

    def create(self, **kwargs):
        response = self._responses[self.calls]
        self.calls += 1
        return response


class FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


class LocalRagAgentTests(unittest.TestCase):
    def test_system_prompt_contains_explicit_datetime(self):
        now = datetime(2026, 6, 20, 12, 0, 0, tzinfo=timezone.utc)
        prompt = build_system_prompt(now=now)
        self.assertIn("2026-06-20T12:00:00+00:00", prompt)
        self.assertIn("Current date and time (UTC):", prompt)

    def test_tools_schema_contains_search_and_fetch(self):
        tools = build_tools_schema()
        names = [t["function"]["name"] for t in tools]
        self.assertEqual(names, ["web_search", "web_fetch"])

    def test_agent_loop_executes_tool_and_returns_final_answer(self):
        tool_request = SimpleNamespace(
            content="",
            tool_calls=[_tool_call("call_1", "web_search", '{"query":"latest ai news"}')],
        )
        final_answer = SimpleNamespace(
            content="Here are the latest AI headlines from the web results.",
            tool_calls=None,
        )
        fake_client = FakeClient([_response(tool_request), _response(final_answer)])

        import local_rag_agent

        original_search = local_rag_agent.web_search
        try:
            local_rag_agent.web_search = lambda query, max_results=5: {
                "query": query,
                "results": [{"title": "AI", "url": "https://example.com", "snippet": "news"}],
            }
            result = run_agent(
                user_input="What happened in AI today?",
                client=fake_client,
                model="gemma",
            )
        finally:
            local_rag_agent.web_search = original_search

        self.assertEqual(
            result, "Here are the latest AI headlines from the web results."
        )
        self.assertEqual(fake_client.chat.completions.calls, 2)


if __name__ == "__main__":
    unittest.main()
