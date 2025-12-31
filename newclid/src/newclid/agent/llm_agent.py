from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Literal, Sequence

from pydantic import BaseModel

from newclid.agent.agents_interface import DeductiveAgent
from newclid.agent.follow_deductions import (
    ARPremiseConstruction,
    CachedARDeduction,
    CachedDeduction,
    CachedNumericalCheckDeduction,
    CachedReflexivityDeduction,
    CachedRuleDeduction,
    DeductionProvider,
    _check_premises_of_deduction,
    _deps_from_conclusions_of_deduction,
    _validate_premises_of_deduction,
    DeductionType,
)
from newclid.problem import ProblemSetup
from py_yuclid.yuclid_adapter import YuclidAdapter

LOGGER = logging.getLogger(__name__)


class LLMAgentStats(BaseModel):
    """Statistics for the LLM-guided agent."""

    agent_type: Literal["llm_agent"] = "llm_agent"
    n_deductions_stored: int
    n_deductions_followed: int
    n_llm_calls: int
    n_llm_failures: int


def _deduction_to_str(deduction: CachedDeduction) -> str:
    match deduction.deduction_type:
        case DeductionType.RULE:
            conclusions = ", ".join(str(c) for c in deduction.conclusions)
            return f"RULE {deduction.rule.fullname}: {conclusions}"
        case DeductionType.AR:
            conclusions = ", ".join(str(c) for c in deduction.conclusions)
            return f"AR {deduction.ar_reason}: {conclusions}"
        case DeductionType.NUM:
            conclusions = ", ".join(str(c) for c in deduction.conclusions)
            return f"NUM: {conclusions}"
        case DeductionType.REFLEXIVITY:
            conclusions = ", ".join(str(c) for c in deduction.conclusions)
            return f"REFL: {conclusions}"
    return "UNKNOWN"


class LLMAgent(DeductiveAgent):
    """
    A simple deductive agent that asks an LLM来排序预计算的推导，并按建议执行。

    设计目标是最小侵入：复用 `FollowDeductions` 的推导收集与校验逻辑，
    仅把“下一步选哪条推导”交给 LLM。如果 LLM 调用失败，会回退到 FIFO 顺序。
    """

    def __init__(
        self,
        deductions_provider: DeductionProvider | None = None,
        *,
        api_key: str | None = None,
        base_url: str = "https://yunwu.ai/v1",
        model: str = "deepseek-v3.2",
        max_options: int = 6,
        timeout: int = 10,
    ) -> None:
        self._deductions_provider = deductions_provider or YuclidAdapter()
        self._deductions: list[CachedDeduction] = []
        self.premises_of_deduction: dict[CachedDeduction, list] = {}
        self.deps_of_deduction: dict[CachedDeduction, list] = {}
        self._has_gathered_deductions = False
        self._api_key = api_key or os.getenv("YUNWU_API_KEY")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._max_options = max_options
        self._timeout = timeout
        self._stats = LLMAgentStats(
            n_deductions_stored=0,
            n_deductions_followed=0,
            n_llm_calls=0,
            n_llm_failures=0,
        )

    def get_stats(self) -> LLMAgentStats:
        return self._stats.model_copy()

    def step(self, proof, rules) -> bool:
        if not self._has_gathered_deductions:
            self._gather_deductions(proof)

        if not self._deductions:
            return False

        # 选项列表（截断到 max_options）
        options = self._deductions[: self._max_options]
        choice_idx = self._choose_with_llm(
            options=options, goals=proof.goals, problem=proof.problem
        )
        if choice_idx is None or not (0 <= choice_idx < len(options)):
            choice_idx = 0  # 回退策略

        next_deduction = options[choice_idx]
        # 从原列表移除该 deduction
        self._deductions.remove(next_deduction)

        precomputation_input_str = self._deductions_provider.precomputation_input_str
        premises_predicates = self.premises_of_deduction[next_deduction]
        _check_premises_of_deduction(
            next_deduction,
            precomputation_input_str=precomputation_input_str,
            premises_predicates=premises_predicates,
            pred_graph=proof.graph,
        )
        for new_dep in self.deps_of_deduction[next_deduction]:
            LOGGER.debug(f"Proved by LLM-guided deduction: {new_dep}")
            _success = proof.apply(new_dep)
            self._stats.n_deductions_followed += 1

        return True

    def _gather_deductions(self, proof_state) -> None:
        self._deductions = self._deductions_provider.ordered_deductions_for_problem(
            proof_state.problem
        ).copy()
        self._stats.n_deductions_stored = len(self._deductions)
        for deduction in self._deductions:
            self.premises_of_deduction[deduction] = _validate_premises_of_deduction(
                deduction,
                precomputation_input_str=self._deductions_provider.precomputation_input_str,
                proof_state=proof_state,
            )
            self.deps_of_deduction[deduction] = _deps_from_conclusions_of_deduction(
                deduction,
                precomputation_input_str=self._deductions_provider.precomputation_input_str,
                premises_predicates=self.premises_of_deduction[deduction],
                proof_state=proof_state,
            )
        self._has_gathered_deductions = True

    # ----- LLM helpers -------------------------------------------------- #
    def _choose_with_llm(
        self,
        *,
        options: Sequence[CachedDeduction],
        goals,
        problem: ProblemSetup,
    ) -> int | None:
        """
        调用 LLM 让它返回 0..len(options)-1 的整数。失败则返回 None。
        """
        if self._api_key is None:
            LOGGER.warning("YUNWU_API_KEY not set; falling back to FIFO.")
            self._stats.n_llm_failures += 1
            return None

        goals_text = ", ".join(str(g) for g in goals)
        options_text = "\n".join(
            f"{idx}: {_deduction_to_str(opt)}" for idx, opt in enumerate(options)
        )
        prompt = (
            "你是一个几何自动证明的决策助手。下面是当前目标和可用的推导步骤，"
            "请选择一个最可能快速完成证明的选项编号（0 开始），只输出数字：\n"
            f"Goals: {goals_text}\n"
            f"Options:\n{options_text}"
        )

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a concise geometry proof planner. Reply with a single integer index.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }
        try:
            self._stats.n_llm_calls += 1
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url=f"{self._base_url}/chat/completions",
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"].strip()
            # 只取首个整数
            for token in content.split():
                try:
                    return int(token)
                except ValueError:
                    continue
        except Exception as exc:  # pragma: no cover - 网络/解析异常
            LOGGER.warning(f"LLM call failed, fallback to FIFO: {exc}")
            self._stats.n_llm_failures += 1
            return None
        self._stats.n_llm_failures += 1
        return None

    def reset(self) -> None:
        self._deductions = []
        self.premises_of_deduction = {}
        self.deps_of_deduction = {}
        self._has_gathered_deductions = False
        self._stats = LLMAgentStats(
            n_deductions_stored=0,
            n_deductions_followed=0,
            n_llm_calls=0,
            n_llm_failures=0,
        )
