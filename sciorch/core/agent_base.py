from __future__ import annotations

from abc import ABC, abstractmethod


class AgentBase(ABC):
    """Base interface for orchestrator agents."""

    @abstractmethod
    async def step(self, *args, **kwargs):
        """Execute one decision step."""

    @abstractmethod
    async def run(self, *args, **kwargs):
        """Run the agent loop."""
