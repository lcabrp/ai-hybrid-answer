import unittest

from local_rag_agent_mcp import build_tools_schema_from_mcp_list


class LocalRagAgentMcpTests(unittest.TestCase):
    def test_build_tools_schema_from_dict_list(self):
        list_tools_result = {
            "tools": [
                {
                    "name": "web_search",
                    "description": "Search",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                        "required": ["query"],
                    },
                }
            ]
        }

        tools = build_tools_schema_from_mcp_list(list_tools_result)
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["function"]["name"], "web_search")
        self.assertEqual(tools[0]["function"]["parameters"]["required"], ["query"])

    def test_missing_schema_falls_back_to_empty_object(self):
        list_tools_result = {
            "tools": [
                {
                    "name": "web_fetch",
                    "description": "Fetch",
                }
            ]
        }

        tools = build_tools_schema_from_mcp_list(list_tools_result)
        self.assertEqual(tools[0]["function"]["parameters"], {"type": "object", "properties": {}})


if __name__ == "__main__":
    unittest.main()
