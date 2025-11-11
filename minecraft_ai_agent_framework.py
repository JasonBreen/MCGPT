"""
Minecraft AI Agent Framework
---------------------------

This module provides a skeleton for building AI-driven Non‑Player Characters
(NPCs) in Minecraft. It separates the core agent logic (perception,
reasoning and action) from the Minecraft integration layer (e.g., Fabric or
Spigot plugins).

The framework is inspired by modern AI‑agent research such as Voyager, which
integrates large language models (LLMs) to continuously explore and learn
in Minecraft【845755156785356†L16-L33】.  It also incorporates ideas from
state‑of‑the‑art agent frameworks like LangGraph, OpenAI Agents SDK,
AutoGen, CrewAI, Google ADK and Dify【219298744872631†L139-L159】【219298744872631†L174-L189】【219298744872631†L199-L220】【219298744872631†L228-L243】【219298744872631†L245-L266】【219298744872631†L269-L287】.

The goal of this skeleton is to offer a starting point for implementing
lifelong learning NPCs that can observe the game world, plan using an
LLM, execute actions in Minecraft, and remember past experiences.

Usage:
   1. Implement a Minecraft plugin/mod that connects the in‑game NPC
      entity to this framework.  The plugin should forward world state
      to the agent and translate the agent’s actions into Minecraft
      commands.
   2. Extend `MinecraftAgent` to customize perception, memory, planning
      and action execution.
   3. Supply your own LLM or agentic framework (LangGraph, AutoGen,
      etc.) in the `Planner` class.

Note: This code is a conceptual template and does not run on its own.
You will need to provide concrete implementations for the abstract
methods, and integrate with Minecraft’s server API (e.g., Fabric or
Spigot) to send/receive events.
"""

from __future__ import annotations

import abc
import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Core Data Structures
# -----------------------------------------------------------------------------

@dataclass
class Observation:
    """Represents a snapshot of the game state relevant to the agent."""
    time: float
    position: Tuple[float, float, float]
    inventory: Dict[str, int]
    nearby_entities: List[Dict[str, Any]]
    chat_history: List[str]
    # Add more fields as needed (biome, weather, objectives, etc.)

@dataclass
class Action:
    """Represents an action the agent wants to perform in the game."""
    verb: str
    target: Optional[Any] = None
    args: Optional[Dict[str, Any]] = None

@dataclass
class MemoryEntry:
    """Stores past interactions or events for long‑term memory."""
    observation: Observation
    action_taken: Action
    result: str  # Free‑text summary or structured outcome

# -----------------------------------------------------------------------------
# Memory Component
# -----------------------------------------------------------------------------

class Memory:
    """Simple memory module that stores observation/action pairs."""
    def __init__(self):
        self.entries: List[MemoryEntry] = []

    def store(self, observation: Observation, action: Action, result: str) -> None:
        entry = MemoryEntry(observation, action, result)
        self.entries.append(entry)

    def recall_recent(self, n: int = 10) -> List[MemoryEntry]:
        return self.entries[-n:]

    def search(self, query: str) -> List[MemoryEntry]:
        """Very naive text search over memory entries."""
        return [e for e in self.entries if query.lower() in e.result.lower()]

# -----------------------------------------------------------------------------
# Planner Interface
# -----------------------------------------------------------------------------

class Planner(abc.ABC):
    """Abstract planner that turns observations into planned actions."""

    @abc.abstractmethod
    def plan(self, observation: Observation, memory: Memory) -> List[Action]:
        """Given the current observation and memory, produce a plan (sequence of actions)."""
        raise NotImplementedError

# Example implementation using a language model

class LLMPlanner(Planner):
    """Planner that queries a Large Language Model or agent framework."""

    def __init__(self, llm_client: Any, system_prompt: str, agent_framework: str = "AutoGen") -> None:
        """
        Parameters
        ----------
        llm_client: object
            An object exposing a `complete(prompt: str) -> str` method, such as
            an OpenAI API client, Anthropic client, or a wrapper around a local LLM.
        system_prompt: str
            A description of the agent’s identity, personality and high‑level goals.
        agent_framework: str
            Name of the agentic framework used (e.g., "LangGraph", "AutoGen", "CrewAI").
        """
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.agent_framework = agent_framework

    def plan(self, observation: Observation, memory: Memory) -> List[Action]:
        # Construct a prompt summarizing the current observation and recent memory
        memory_summaries = "\n".join(
            [f"At {time.ctime(entry.observation.time)} you did {entry.action_taken.verb} and observed {entry.result}."
             for entry in memory.recall_recent(5)]
        )
        prompt = f"""
{self.system_prompt}

Current Observation:
- Position: {observation.position}
- Inventory: {json.dumps(observation.inventory)}
- Nearby entities: {json.dumps(observation.nearby_entities)}
- Chat: {observation.chat_history[-3:] if observation.chat_history else []}

Recent Memory:
{memory_summaries if memory_summaries else 'None'}

Based on the above, think step‑by‑step and propose a sequence of high‑level actions to achieve your goals.  
Respond with a JSON list where each element has fields: "verb", "target", and optional "args".
"""
        
        # Ask the LLM for a plan. This call can be routed through a framework
        # like LangGraph or AutoGen to leverage their tracing, multi‑agent
        # orchestration, and error handling features【219298744872631†L139-L159】【219298744872631†L199-L220】.
        response = self.llm_client.complete(prompt)

        try:
            actions_data = json.loads(response)
            actions: List[Action] = [Action(verb=a.get("verb"), target=a.get("target"), args=a.get("args"))
                                     for a in actions_data]
            return actions
        except Exception:
            # If parsing fails, fall back to no action
            return []


class SimpleHeuristicPlanner(Planner):
    """Deterministic planner useful for local testing without an LLM."""

    def __init__(self, desired_wood: int = 3, exploration_step: Tuple[float, float, float] = (1.0, 0.0, 1.0)) -> None:
        self.desired_wood = desired_wood
        self.exploration_step = exploration_step

    def plan(self, observation: Observation, memory: Memory) -> List[Action]:
        plan: List[Action] = []

        wood_count = observation.inventory.get("wood", 0)
        if wood_count < self.desired_wood:
            plan.append(Action(verb="mine", target="wood", args={"reason": "gather_resources"}))
        else:
            # Look up the last move target to avoid oscillating.
            last_move_target: Optional[Tuple[float, float, float]] = None
            for entry in reversed(memory.recall_recent(5)):
                if entry.action_taken.verb == "move" and isinstance(entry.action_taken.target, (list, tuple)):
                    last_move_target = tuple(entry.action_taken.target)
                    break

            next_target = tuple(
                observation.position[i] + self.exploration_step[i]
                for i in range(3)
            )
            if last_move_target == next_target:
                # Take a different direction if we already moved there recently
                next_target = (
                    observation.position[0] + self.exploration_step[2],
                    observation.position[1],
                    observation.position[2] + self.exploration_step[0],
                )

            plan.append(Action(verb="move", target=list(next_target), args={"style": "explore"}))
            plan.append(Action(verb="say", target="Exploring the surroundings!", args={"channel": "npc"}))

        return plan

# -----------------------------------------------------------------------------
# Minecraft Agent
# -----------------------------------------------------------------------------

class MinecraftAgent:
    """
    Encapsulates the AI NPC’s lifecycle: observe, plan, act, and remember.

    This class should be instantiated by the Minecraft plugin/mod.  The plugin
    should provide implementations for the abstract methods
    `_get_current_observation` and `_execute_action`.
    """

    def __init__(self, planner: Planner, memory: Optional[Memory] = None) -> None:
        self.planner = planner
        self.memory = memory or Memory()

    # ---------------------- Abstract Integration Methods ----------------------
    def _get_current_observation(self) -> Observation:
        """Fetch the latest game state from the Minecraft server.  
        Must be implemented in the plugin/mod layer."""
        raise NotImplementedError

    def _execute_action(self, action: Action) -> str:
        """Send the action to the server (e.g., move, mine, chat).  
        Must return a textual result of what happened, for memory storage."""
        raise NotImplementedError

    # ----------------------- Agent Lifecycle Loop -----------------------------
    def tick(self) -> None:
        """One iteration of the agent’s perception→reasoning→action loop."""
        observation = self._get_current_observation()

        plan = self.planner.plan(observation, self.memory)
        if not plan:
            return  # No actions; perhaps idle

        for action in plan:
            result = self._execute_action(action)
            # Store to memory for future recall
            self.memory.store(observation, action, result)
            # Update observation after each action (optional but useful)
            observation = self._get_current_observation()

    def run_forever(self, interval: float = 2.0) -> None:
        """Continuously run the agent loop with a delay between iterations."""
        while True:
            self.tick()
            time.sleep(interval)

# -----------------------------------------------------------------------------
# Example stub implementations for integration
# -----------------------------------------------------------------------------

class StubMinecraftPlugin(MinecraftAgent):
    """
    This stub simulates the plugin interface for demonstration purposes.  In a real
    plugin, these methods would interface with the server API (e.g., Fabric or
    Spigot) to get the NPC’s state and issue commands.  This stub simply
    returns fixed data and echoes actions.
    """

    def __init__(self, planner: Planner):
        super().__init__(planner)
        self.position = (0.0, 64.0, 0.0)
        self.inventory = {"wood": 0, "stone": 0}
        self.chat_history = []

    def _get_current_observation(self) -> Observation:
        # In reality, gather data from Minecraft’s world and the NPC entity
        obs = Observation(
            time=time.time(),
            position=self.position,
            inventory=self.inventory.copy(),
            nearby_entities=[],
            chat_history=self.chat_history.copy(),
        )
        return obs

    def _execute_action(self, action: Action) -> str:
        # In reality, convert the Action into game commands.  Here we just
        # update local state and print to console for demonstration.
        args_display = action.args if action.args is not None else {}
        result_summary = f"Executed {action.verb} with target={action.target} and args={args_display}"
        # Example: update inventory if action is 'mine'
        if action.verb == "mine" and action.target == "wood":
            self.inventory["wood"] += 1
            result_summary += "; gained wood"
        elif action.verb == "move" and isinstance(action.target, (list, tuple)):
            self.position = tuple(action.target)
            result_summary += f"; new position={self.position}"
        elif action.verb == "say" and isinstance(action.target, str):
            # Append to chat history
            self.chat_history.append(action.target)
            result_summary += "; message sent"
        print(result_summary)
        return result_summary

# -----------------------------------------------------------------------------
# Usage Example (Pseudo‑Code)
# -----------------------------------------------------------------------------

def _build_planner(name: str) -> Planner:
    """Factory helper for the CLI demo."""

    if name == "simple":
        return SimpleHeuristicPlanner()

    if name == "dummy-llm":
        class DummyLLMClient:
            def complete(self, prompt: str) -> str:
                # For demonstration, always return a simple plan
                return json.dumps([
                    {"verb": "mine", "target": "wood", "args": {}},
                    {"verb": "move", "target": [1.0, 64.0, 1.0], "args": {}},
                    {"verb": "say", "target": "Hello there!", "args": {}}
                ])

        system_prompt = (
            "You are an autonomous Minecraft NPC. Your goal is to survive, gather resources, "
            "build shelters, and help nearby players. You must follow Minecraft’s rules and "
            "never cheat. Provide detailed plans using high-level actions like 'move', 'mine', "
            "'craft', and 'say'."
        )
        return LLMPlanner(DummyLLMClient(), system_prompt, agent_framework="LangGraph")

    raise ValueError(f"Unknown planner '{name}'. Choose from 'simple' or 'dummy-llm'.")


def _run_demo(planner_name: str, ticks: int, interval: float) -> None:
    agent = StubMinecraftPlugin(_build_planner(planner_name))
    for _ in range(ticks):
        agent.tick()
        if interval:
            time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Minecraft agent stub loop.")
    parser.add_argument("--planner", choices=["simple", "dummy-llm"], default="simple",
                        help="Planner implementation to use for the demo run.")
    parser.add_argument("--ticks", type=int, default=2,
                        help="Number of agent ticks to execute before exiting.")
    parser.add_argument("--interval", type=float, default=0.0,
                        help="Optional sleep duration (seconds) between ticks.")
    args = parser.parse_args()

    _run_demo(args.planner, max(1, args.ticks), max(0.0, args.interval))
