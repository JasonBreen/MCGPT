# MCGPT

MCGPT is an experimental framework for building autonomous Minecraft NPCs that
combine Large Language Model (LLM) planning with classic game AI techniques.
The repository currently focuses on the Python-side agent loop and ships with a
stub plugin so you can prototype planning logic before wiring it to an actual
Minecraft server mod or plugin.

## Repository Layout

| Path | Description |
| --- | --- |
| `minecraft_ai_agent_framework.py` | Core agent abstractions (observation, actions, memory, planners) and a stub integration. |
| `codex_overview.md` | Architectural notes that explain how the Python agent will pair with a Fabric/Paper integration layer. |

## Quick Start

You can experiment with the in-memory stub agent without launching Minecraft:

```bash
python minecraft_ai_agent_framework.py --ticks 3 --planner simple
```

This command executes the agent loop three times using the bundled
`SimpleHeuristicPlanner`, which demonstrates how the agent updates its
inventory, moves, and records chat messages.

To see how the framework can call into an LLM (or a mock), run:

```bash
python minecraft_ai_agent_framework.py --ticks 2 --planner dummy-llm
```

## Next Steps

The framework is intentionally lightweight. Recommended follow-up work
includes:

- Implementing a Fabric mod or Paper plugin that forwards observations to the
  Python agent and executes returned actions.
- Replacing the dummy LLM client with your preferred hosted or local model.
- Extending `Memory` with retrieval mechanisms (vector stores, summaries, etc.).
- Introducing persistence for agent state so NPCs can learn across sessions.

