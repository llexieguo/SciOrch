from __future__ import annotations

from dataclasses import dataclass

from sciorch.core.subagent_reasoning import SubAgentReasoning
from sciorch.types import DelegateRequest, DelegateResult


@dataclass
class DelegateTaskTool:
    sub_agent: SubAgentReasoning

    async def __call__(self, request: DelegateRequest) -> DelegateResult:
        return await self.sub_agent.run(request)
