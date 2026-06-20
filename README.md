# ai-hybrid-answer
Testing an idea from a comment of a XDA article.

## Local llama.cpp + web tools agent

`local_rag_agent.py` is a Python orchestrator for a local `llama-server` endpoint
(`http://localhost:8080/v1`) that:

- injects the current UTC date/time into the system prompt,
- exposes DuckDuckGo search and URL fetch as tool-calls,
- runs a tool-calling loop until the model returns a final answer.

### Setup

```bash
pip install -r requirements.txt
```

### Usage

```bash
python local_rag_agent.py "What are the latest AI safety news updates today?"
```
