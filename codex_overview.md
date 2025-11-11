# Overview: Minecraft AI Agent Framework

This document provides a high-level overview of the planned architecture for autonomous NPCs in Minecraft using an AI agent. It is intended to assist language-model based tools like Codex in understanding the project context and generating useful code.

## Python Agent Layer

- The agent’s reasoning, planning, and memory reside in Python.
- Key components:
  - **Observation**: Captures the NPC’s current state, including position, environment, and inventory.
  - **Action**: Represents a single high-level action (e.g., move, mine, craft, speak).
  - **Memory**: Stores past observations, actions, and outcomes to enable learning and context awareness.
  - **Planner Interface**: Abstract class defining the `plan()` method that turns observations and memory into a sequence of actions.
  - **LLMPlanner**: Implementation that prompts a language model to generate plans. It uses a system prompt to define the NPC’s personality and goals.
  - **MinecraftAgent**: Core loop that gathers observations, calls the planner, executes actions, and updates memory.
  - **Stub Plugin**: Demonstration class for testing the agent outside of Minecraft.

This code lives in `minecraft_ai_agent_framework.py`.

## Java/Fabric Integration Layer

The Python agent cannot directly control entities inside Minecraft. A separate mod/plugin written in Java (or Kotlin) must:

1. **Spawn and manage NPC entities** within the game world.
2. **Collect observations** (position, surroundings, inventory, chat).
3. **Send observations to the Python agent** via IPC (e.g., WebSockets or HTTP).
4. **Receive planned actions** from the agent and translate them into in-game behaviors (movement, mining, crafting, combat).
5. **Persist state** if needed.

For deep control over AI behaviors, a Fabric mod is recommended. It allows custom entity classes and AI tasks. A Paper plugin is an alternative for simpler deployments but provides less control over pathfinding and entity brains.

## Communication Protocol

Define a JSON-based protocol for messages between the Java integration layer and the Python agent. For example:

- **Observation message** from Java to Python:

```json
{
  "type": "observation",
  "position": [x, y, z],
  "nearbyEntities": [...],
  "inventory": {...},
  "chat": [...]
}
```

- **Action message** from Python to Java:

```json
{
  "type": "action",
  "verb": "move",
  "target": [x, y, z],
  "args": {}
}
```

The Java side should interpret the action and apply it using Minecraft’s server API.

## Next Steps

- Choose the server environment (Fabric vs. Paper).
- Scaffold a Java project that implements the integration layer. Include:
  - A custom NPC entity class.
  - AI tasks for idle, look, and movement.
  - A WebSocket or HTTP client to communicate with the Python agent.
  - Commands to spawn/test the NPC.

Once the integration layer is in place, the Python agent can be connected and tested in-game.

---

This document should give Codex enough context to generate Java classes and handle the agent bridge.
