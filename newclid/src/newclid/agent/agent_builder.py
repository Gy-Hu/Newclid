from typing import Any

from newclid.agent._index import AgentName
from newclid.agent.agents_interface import DeductiveAgent
from newclid.agent.ddarn import DDARN
from newclid.agent.follow_deductions import FollowDeductions
from newclid.agent.human_agent import HumanAgent
from newclid.agent.llm_agent import LLMAgent
from py_yuclid.yuclid_adapter import YuclidAdapter


def make_agent(agent_name: str | AgentName, **kwargs: Any) -> DeductiveAgent:
    match AgentName(agent_name):
        case AgentName.DDARN:
            return DDARN()
        case AgentName.HUMAN_AGENT:
            return HumanAgent()
        case AgentName.FOLLOW_DEDUCTIONS:
            provider = kwargs.get("deductions_provider") or YuclidAdapter()
            return FollowDeductions(deductions_provider=provider)
        case AgentName.LLM_AGENT:
            provider = kwargs.get("deductions_provider") or YuclidAdapter()
            return LLMAgent(deductions_provider=provider, **kwargs)
