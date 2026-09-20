---
title: Enterprise Multi-Agent AI Assistant
emoji: 🧩
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Multi-agent RAG assistant with citations and a human approval gate
---

# Enterprise Multi-Agent AI Assistant

A multi-agent retrieval-augmented assistant for enterprise policy questions and
business workflows. Four agents coordinate through a LangGraph state machine:

| Agent | Responsibility |
| --- | --- |
| **Planner** | Classifies intent and writes the retrieval queries |
| **Retrieval** | Hybrid BM25 + vector search over the policy corpus |
| **Reasoning** | Answers using only the retrieved passages, with citations |
| **Workflow** | Prepares a validated business action - never executes one |
| **Verification** | Checks groundedness, citations and policy conflicts |

Anything that would write to a business system stops at a **human approval
gate** before it runs.

This Space runs with no API key in deterministic mode: answers are extracted
verbatim from the cited documents, so the demo is free and reproducible.
Adding an `ANTHROPIC_API_KEY` secret switches the same graph to generative
reasoning with Claude.

**Source:** https://github.com/pardhu0201/enterprise-multi-agent-assistant
