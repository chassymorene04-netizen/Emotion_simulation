from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock
from typing import Callable, Dict, List, Optional

import numpy as np

from agent import ACTION_TYPE_TO_CODE, Agent
from prompts import (
    build_batch_action_decision_prompt,
    build_batch_perception_prompt,
)
from config import LLMConfig, SimulationConfig
from generation import (
    contains_prompt_leak,
    scenario_generation_spec,
    seed_generation_prompt,
)
from scoring import (
    STRATEGY_PARAMETER_MAPPING_VERSION,
    STRATEGY_SCORE_FIELDS,
    STRATEGY_SCORING_PROMPT_VERSION,
    aggregate_strategy_parameters,
    score_seed_content,
)
from models import (
    ActionDecision,
    ActionEvent,
    ContentItem,
    FeedBundle,
    HIGH_INTERACTION_FOLLOWER_PERSONA_KEY,
    LOW_REACTIVITY_PERSONA_KEY,
    NORMAL_PERSONA_KEYS,
    PerceptionResult,
    ScenarioConfig,
    SILENT_OBSERVER_PERSONA_KEY,
    EMOTIONAL_SENSITIVE_PERSONA_KEY,
    sample_persona,
)
from formulas import (
    clip_01,
    compute_amplification,
    compute_duration,
    compute_group_activation_rate,
    compute_group_dispersion,
    compute_group_exposure_rate,
    compute_high_negative_ratio,
    compute_interaction_rate,
    compute_mean_step_emotion_increment,
    compute_peak,
    compute_peak_average_rise_speed,
    compute_recovery_gap,
    group_mean_negative_emotion,
    interaction_feedback,
)
from llm_client import LLMClient
from experimental_data import (
    SourceEvent,
    default_source_events,
    experimental_stimulus_event_id,
    load_experimental_stimuli,
)
from platform_mechanisms import build_agent_feed, build_network


@dataclass
class SimulationResult:
    """单次仿真的结果容器。"""

    seed_nodes: List[int]
    scenario_type: str
    active_source_event_id: str
    active_source_event_summary: str
    group_negative_emotion: List[float]
    high_negative_ratio: List[float]
    group_exposure_rate: List[float]
    group_activation_rate: List[float]
    group_dispersion: List[float]
    peak: float
    amplification: float
    recovery_gap: float
    mean_step_emotion_increment: float
    peak_average_rise_speed: float
    duration: float
    exposure_counts: List[int]
    interaction_counts: List[int]
    active_counts: List[int]
    active_rate: List[float]
    time_series: List[Dict[str, float]]
    action_counts: Dict[str, int]
    participation_counts: Dict[str, int]
    participation_action_counts: Dict[str, int]
    interaction_cases: List[Dict[str, object]]
    sampled_agent_state_rows: List[Dict[str, object]]
    runtime_llm_request_count: int
    runtime_llm_failure_count: int
    runtime_llm_batch_request_count: int
    runtime_llm_inference_item_count: int
    runtime_llm_valid_item_count: int
    runtime_llm_invalid_item_count: int
    runtime_llm_fallback_item_count: int
    runtime_llm_input_token_count: int
    runtime_llm_output_token_count: int
    runtime_llm_estimated_cost_rmb: float
    runtime_llm_diagnostics: Dict[str, int | float]
    runtime_llm_model: str
    risk_boost: float
    memory_boost: float
    share_boost: float
    strategy_parameter_source: str

    def summary_dict(self) -> Dict[str, float | str | int]:
        """整理为单行摘要结果。"""

        initial_group_negative_emotion = self.group_negative_emotion[0] if self.group_negative_emotion else 0.0
        final_group_negative_emotion = self.group_negative_emotion[-1] if self.group_negative_emotion else 0.0
        summary: Dict[str, float | str | int] = {
            "scenario_type": self.scenario_type,
            "active_source_event_id": self.active_source_event_id,
            "active_source_event_summary": self.active_source_event_summary,
            "seed_node_count": len(self.seed_nodes),
            "initial_group_negative_emotion": initial_group_negative_emotion,
            "peak": self.peak,
            "amplification": self.amplification,
            "recovery_gap": self.recovery_gap,
            "mean_step_emotion_increment": self.mean_step_emotion_increment,
            "peak_average_rise_speed": self.peak_average_rise_speed,
            "duration": self.duration,
            "final_group_negative_emotion": final_group_negative_emotion,
            "final_high_negative_ratio": self.high_negative_ratio[-1] if self.high_negative_ratio else 0.0,
            "final_group_dispersion": self.group_dispersion[-1] if self.group_dispersion else 0.0,
            "runtime_llm_request_count": self.runtime_llm_request_count,
            "runtime_llm_failure_count": self.runtime_llm_failure_count,
            "runtime_llm_batch_request_count": self.runtime_llm_batch_request_count,
            "runtime_llm_inference_item_count": self.runtime_llm_inference_item_count,
            "runtime_llm_valid_item_count": self.runtime_llm_valid_item_count,
            "runtime_llm_invalid_item_count": self.runtime_llm_invalid_item_count,
            "runtime_llm_fallback_item_count": self.runtime_llm_fallback_item_count,
            "runtime_llm_input_token_count": self.runtime_llm_input_token_count,
            "runtime_llm_output_token_count": self.runtime_llm_output_token_count,
            "runtime_llm_estimated_cost_rmb": self.runtime_llm_estimated_cost_rmb,
            "runtime_llm_model": self.runtime_llm_model,
            "risk_boost": self.risk_boost,
            "memory_boost": self.memory_boost,
            "share_boost": self.share_boost,
            "strategy_parameter_source": self.strategy_parameter_source,
        }
        return summary


@dataclass
class AgentStepResult:
    """保存单个 agent 在某一轮的完整计算结果。"""

    node_id: int
    updated_agent: Agent
    perception: PerceptionResult
    decision: ActionDecision
    generated_text: str
    action_strength: float
    generated_content: Optional[ContentItem]
    action_event: Optional[ActionEvent]
    target_content_id: Optional[str]
    feed_misinfo_exposed: bool
    active: bool
    used_runtime_llm: bool = False


class EmotionDiffusionSimulation:
    """LLM-agent 驱动的社交扩散仿真主类。"""

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        scenario_type: str = "risk_exaggeration",
        scenario_override: Optional[ScenarioConfig] = None,
        llm_config: Optional[LLMConfig] = None,
        progress_callback: Optional[Callable[[str, Dict[str, object]], None]] = None,
    ) -> None:
        self.config = config or SimulationConfig.from_env()
        self.config.ensure_directories()
        self.scenario_type = scenario_type
        self.progress_callback = progress_callback

        self._scenario_override = scenario_override
        self.scenario = scenario_override or ScenarioConfig(
            scene_name=scenario_type,
            memory_boost=0.0,
            share_boost=0.0,
            risk_boost=0.0,
        )
        self.strategy_parameter_source = (
            "experiment_override" if scenario_override is not None else "pending_seed_scoring"
        )

        self.llm_config = llm_config or LLMConfig.from_env()
        self.llm_client = LLMClient(
            api_key=self.llm_config.api_key,
            base_url=self.llm_config.base_url,
            model=self.llm_config.model,
            timeout=self.llm_config.timeout,
        )
        self.rng = np.random.default_rng(self.config.random_seed)
        self.graph = build_network(
            network_type=self.config.network_type,
            num_agents=self.config.num_agents,
            edge_probability=self.config.graph_edge_probability,
            random_seed=self.config.random_seed,
        )

        self.content_counter = 0
        self._content_id_lock = Lock()
        self.content_pool: List[ContentItem] = []
        self.content_index: Dict[str, ContentItem] = {}
        self.recent_actions: List[ActionEvent] = []
        self.source_events = default_source_events()
        self.active_source_event = self._select_active_source_event()
        self.experimental_stimuli = load_experimental_stimuli(
            self.config.experimental_stimuli_path
        )
        self._use_experimental_stimulus_event()
        self.persona_nodes: Dict[str, List[int]] = {}

        self.agents = self._initialize_agents()
        self.initial_misinfo_exposed_nodes = self._select_initial_misinfo_exposed_nodes()
        self.initial_content_assignments: Dict[int, str] = {}
        self.seed_nodes = self._initialize_seed_nodes()
        self.runtime_llm_diagnostics: Dict[str, int | float] = {
            "active_agent_steps": 0,
            "llm_qualified_items": 0,
            "llm_disqualified_no_misinfo_items": 0,
            "llm_disqualified_no_focus_items": 0,
            "llm_eligible_before_cap_items": 0,
            "llm_admitted_after_cap_items": 0,
            "llm_deferred_by_cap_items": 0,
            "perception_refresh_items": 0,
            "perception_request_items": 0,
            "perception_retry_items": 0,
            "perception_rule_fallback_items": 0,
            "decision_refresh_items": 0,
            "decision_request_items": 0,
            "decision_retry_items": 0,
            "decision_rule_execution_items": 0,
            "decision_blocked_by_invalid_perception_items": 0,
            "decision_rule_fallback_items": 0,
        }

    def _emit_progress(self, event: str, **payload: object) -> None:
        """向外发送仿真进度事件。"""

        if self.progress_callback is None:
            return
        self.progress_callback(
            event,
            {
                "scenario_type": self.scenario_type,
                "progress_detail": self.config.progress_detail,
                **payload,
            },
        )

    def _select_active_source_event(self) -> SourceEvent:
        """选定当前实验使用的中性事件母本。"""

        if not self.source_events:
            raise ValueError("系统中没有可用的中性事件母本。")
        index = int(self.rng.integers(0, len(self.source_events)))
        return self.source_events[index]

    def _use_experimental_stimulus_event(self) -> None:
        """Use the event background associated with the published stimuli."""

        event_id = experimental_stimulus_event_id(
            self.experimental_stimuli,
            self.scenario_type,
        )
        if not event_id:
            raise ValueError(f"实验刺激数据缺少场景：{self.scenario_type}")
        for event in self.source_events:
            if event.source_event_id == event_id:
                self.active_source_event = event
                return
        raise ValueError(f"实验刺激数据引用了未知事件母本：{event_id}")

    def _next_content_id(self) -> str:
        """生成递增内容 ID。"""

        with self._content_id_lock:
            self.content_counter += 1
            return f"content_{self.content_counter:05d}"

    def _register_content(self, item: ContentItem) -> None:
        """登记内容对象，供传播池和上下文查询共用。"""

        self.content_index[item.content_id] = item

    def _normal_persona_ratios(self) -> Dict[str, float]:
        return {
            SILENT_OBSERVER_PERSONA_KEY: max(0.0, self.config.silent_observer_ratio),
            EMOTIONAL_SENSITIVE_PERSONA_KEY: max(0.0, self.config.emotional_sensitive_ratio),
            LOW_REACTIVITY_PERSONA_KEY: max(0.0, self.config.low_reactivity_user_ratio),
            HIGH_INTERACTION_FOLLOWER_PERSONA_KEY: max(0.0, self.config.high_interaction_follower_ratio),
        }

    def _initial_exposure_persona_ratios(self) -> Dict[str, float]:
        return {
            SILENT_OBSERVER_PERSONA_KEY: max(0.0, self.config.initial_silent_observer_ratio),
            EMOTIONAL_SENSITIVE_PERSONA_KEY: max(0.0, self.config.initial_emotional_sensitive_ratio),
            LOW_REACTIVITY_PERSONA_KEY: max(0.0, self.config.initial_low_reactivity_user_ratio),
            HIGH_INTERACTION_FOLLOWER_PERSONA_KEY: max(0.0, self.config.initial_high_interaction_follower_ratio),
        }

    def _allocate_counts_by_ratio(self, total_count: int, ratios: Dict[str, float]) -> Dict[str, int]:
        if total_count <= 0:
            return {key: 0 for key in ratios}
        ratio_sum = sum(ratios.values())
        if ratio_sum <= 0.0:
            ratios = {key: 1.0 for key in ratios}
            ratio_sum = float(len(ratios))
        raw_counts = {key: total_count * value / ratio_sum for key, value in ratios.items()}
        counts = {key: int(np.floor(value)) for key, value in raw_counts.items()}
        remaining = total_count - sum(counts.values())
        order = sorted(ratios.keys(), key=lambda key: raw_counts[key] - counts[key], reverse=True)
        for index in range(remaining):
            counts[order[index % len(order)]] += 1
        return counts

    def _select_initial_misinfo_exposed_nodes(self) -> set[int]:
        normal_nodes = list(self.agents.keys())
        if not normal_nodes:
            return set()
        target_count = int(round(len(normal_nodes) * clip_01(self.config.initial_misinfo_exposure_ratio)))
        target_count = max(1, min(len(normal_nodes), target_count))

        nodes_by_persona: Dict[str, List[int]] = {key: [] for key in NORMAL_PERSONA_KEYS}
        for node_id in normal_nodes:
            persona_key = self.agents[node_id].persona.persona_key
            if persona_key in nodes_by_persona:
                nodes_by_persona[persona_key].append(node_id)

        target_counts = self._allocate_counts_by_ratio(target_count, self._initial_exposure_persona_ratios())
        selected: set[int] = set()
        for persona_key in NORMAL_PERSONA_KEYS:
            pool = nodes_by_persona.get(persona_key, [])
            draw_count = min(target_counts.get(persona_key, 0), len(pool))
            if draw_count > 0:
                selected.update(int(node_id) for node_id in self.rng.choice(pool, size=draw_count, replace=False))

        remaining_count = target_count - len(selected)
        if remaining_count > 0:
            remaining_pool = [node_id for node_id in normal_nodes if node_id not in selected]
            selected.update(int(node_id) for node_id in self.rng.choice(remaining_pool, size=remaining_count, replace=False))
        return selected

    def _allocate_normal_persona_keys(self, normal_nodes: List[int]) -> Dict[int, str]:
        if not normal_nodes:
            return {}
        if not self.config.use_persona_ratio_allocation:
            return {
                node_id: NORMAL_PERSONA_KEYS[int(self.rng.integers(0, len(NORMAL_PERSONA_KEYS)))]
                for node_id in normal_nodes
            }
        ratios = self._normal_persona_ratios()
        counts = self._allocate_counts_by_ratio(len(normal_nodes), ratios)
        persona_keys: List[str] = []
        for key in NORMAL_PERSONA_KEYS:
            persona_keys.extend([key] * counts[key])
        self.rng.shuffle(persona_keys)
        return {node_id: persona_keys[index] for index, node_id in enumerate(normal_nodes)}

    def _initialize_agents(self) -> Dict[int, Agent]:
        """初始化异质化 agent。"""

        nodes = [int(node) for node in self.graph.nodes()]
        normal_nodes = list(nodes)
        persona_key_by_node = self._allocate_normal_persona_keys(normal_nodes)
        self.persona_nodes = {key: [] for key in NORMAL_PERSONA_KEYS}

        agents: Dict[int, Agent] = {}
        for node_id in nodes:
            node_id = int(node_id)
            persona_key = persona_key_by_node[node_id]
            persona = sample_persona(
                self.rng,
                role_family="normal_user",
                scenario_type=self.scenario_type,
                forced_persona_key=persona_key,
            )
            participation_mode = persona.persona_key
            self.persona_nodes[persona.persona_key].append(node_id)
            initial_emotion = float(self.rng.uniform(*self.config.initial_emotion_range))
            agents[node_id] = Agent(
                node_id=node_id,
                persona=persona,
                emotion=initial_emotion,
                sensitivity=float(self.rng.uniform(*self.config.sensitivity_range)),
                memory_window=int(
                    self.rng.integers(
                        self.config.memory_window_range[0],
                        self.config.memory_window_range[1] + 1,
                    )
                ),
                interaction_tendency=float(self.rng.uniform(*persona.interaction_tendency_range)),
                participation_mode=participation_mode,
                emotion_history=[initial_emotion],
            )
        return agents

    def _event_by_id(self, source_event_id: Optional[str]) -> SourceEvent:
        """根据事件 ID 查询事件母本。"""

        if source_event_id:
            for event in self.source_events:
                if event.source_event_id == source_event_id:
                    return event
        return self.active_source_event

    def _create_content_from_generated_payload(
        self,
        payload: Dict[str, object],
        source_agent_id: int,
        source_role: str,
        step: int,
        parent_content_id: Optional[str] = None,
    ) -> ContentItem:
        """将生成结果转换为平台内容对象。"""

        scenario_type = self.scenario_type
        source_event = self._event_by_id(str(payload.get("source_event_id", self.active_source_event.source_event_id)))
        title = str(payload.get("title", "")).strip()
        text = str(payload.get("text", "")).strip()
        full_text = f"{title}。{text}".strip("。") if title else text

        arousal_score = clip_01(float(payload.get("arousal_score", 0.0)))
        misleading_ratio = clip_01(float(payload["misleading_ratio"]))

        return ContentItem(
            content_id=self._next_content_id(),
            source_agent_id=source_agent_id,
            source_role=source_role,
            content_type="post",
            text=full_text,
            scenario_type=scenario_type,
            source_event_id=source_event.source_event_id,
            arousal_score=arousal_score,
            misleading_ratio=misleading_ratio,
            hotness_score=clip_01(0.26 + 0.22 * arousal_score + self.scenario.share_boost),
            created_step=step,
            parent_content_id=parent_content_id,
            thread_root_content_id=None,
            thread_depth=0,
            metadata={
                "title": title,
                "source_event_summary": source_event.source_event_summary,
                "generated_by": "system_misinformation_source",
                "scenario_generation_strategy": str(payload.get("scenario_generation_strategy", "")),
                "text_generated_by": str(payload.get("text_generated_by", "rule")),
                "scores_generated_by": str(payload.get("scores_generated_by", "rule")),
                "strategy_scores_generated_by": str(
                    payload.get("strategy_scores_generated_by", "rule")
                ),
            },
        )

    def _resolve_thread_context(self, target_content: Optional[ContentItem]) -> tuple[Optional[str], int]:
        """根据目标内容推断新内容的线程位置。"""

        if target_content is None:
            return None, 0
        return target_content.root_id(), target_content.thread_depth + 1

    @staticmethod
    def _contains_prompt_leak(text: str) -> bool:
        return contains_prompt_leak(text)

    def _scenario_generation_spec(self) -> Dict[str, object]:
        return scenario_generation_spec(self.scenario_type)

    def _build_seed_generation_prompt(
        self,
        spec: Dict[str, object],
        event: SourceEvent,
    ) -> str:
        return seed_generation_prompt(
            spec,
            event.source_event_id,
            event.source_event_summary,
        )

    @staticmethod
    def _is_valid_experimental_stimulus(payload: Dict[str, object]) -> bool:
        """Validate that a published stimulus contains text and all scores."""

        if (
            payload.get("text_generated_by") != "llm"
            or payload.get("scores_generated_by") != "llm"
            or payload.get("strategy_scores_generated_by") != "llm"
        ):
            return False
        if not str(payload.get("title", "")).strip() or not str(payload.get("text", "")).strip():
            return False
        try:
            arousal = float(payload["arousal_score"])
            misleading = float(payload["misleading_ratio"])
            strategy_scores = [float(payload[field]) for field in STRATEGY_SCORE_FIELDS]
        except (KeyError, TypeError, ValueError):
            return False
        return all(
            0.0 <= value <= 1.0
            for value in [arousal, misleading, *strategy_scores]
        )

    def _experimental_stimulus(self, seed_index: int) -> Dict[str, object]:
        rows = self.experimental_stimuli.get(self.scenario_type, [])
        if not rows:
            raise ValueError(f"实验刺激数据缺少场景：{self.scenario_type}")
        payload = dict(rows[seed_index % len(rows)])
        if not self._is_valid_experimental_stimulus(payload):
            raise ValueError(
                f"实验刺激第 {seed_index + 1} 条不完整，"
                "必须同时包含 LLM 生成文本和 LLM 评分。"
            )
        return payload

    def _required_seed_content_count(self) -> int:
        return max(1, int(round(self.config.num_agents * self.config.seed_fraction)))

    def _generate_system_misinformation_payload(self, seed_index: int) -> Dict[str, object]:
        event = self.active_source_event
        spec = self._scenario_generation_spec()

        title = str(spec["title"])
        text = str(spec["text_template"]).format(summary=event.source_event_summary)
        payload: Dict[str, object] = {
            "title": title,
            "text": f"{text}（种子内容{seed_index + 1}）",
            "source_event_id": event.source_event_id,
            "scenario_type": self.scenario_type,
            "scenario_generation_strategy": spec["strategy"],
            "generation_prompt_version": "strategy_expression_range_v2",
            "text_generated_by": "rule",
            "scores_generated_by": "rule",
            "strategy_scores_generated_by": "rule",
            "strategy_scoring_prompt_version": STRATEGY_SCORING_PROMPT_VERSION,
            "strategy_parameter_mapping_version": STRATEGY_PARAMETER_MAPPING_VERSION,
        }
        if not (self.config.runtime_llm_enabled and self.llm_client.has_api_key()):
            raise RuntimeError(
                "在线生成初始化材料需要有效的运行时 LLM 配置。"
            )

        generation_prompt = self._build_seed_generation_prompt(spec, event)
        generated_payload: Optional[Dict[str, object]] = None
        for attempt in range(2):
            attempt_prompt = generation_prompt
            if attempt == 1:
                attempt_prompt += (
                    "\n\n【纠错要求】上一次输出未通过格式或内容检查。"
                    "本次仅返回包含 title 和 text 的 JSON，不得复述任何任务要求、策略名称或强度信息。"
                )
            generated = self.llm_client.generate_json(
                attempt_prompt,
                temperature=self.llm_config.content_temperature,
            )
            if not isinstance(generated, dict):
                continue

            generated_title = str(generated.get("title", "")).strip()
            generated_text = str(generated.get("text", "")).strip()
            if not generated_title or not generated_text:
                continue
            if self._contains_prompt_leak(generated_title) or self._contains_prompt_leak(generated_text):
                continue

            generated_payload = {
                "title": generated_title,
                "text": generated_text,
            }
            break

        if generated_payload is not None:
            payload.update(generated_payload)
            payload["text_generated_by"] = "llm"

        payload.update(
            score_seed_content(
                llm_client=self.llm_client,
                source_event_id=event.source_event_id,
                source_event_summary=event.source_event_summary,
                title=str(payload["title"]),
                text=str(payload["text"]),
            )
        )
        return payload

    def _build_system_misinformation_payload(self, seed_index: int) -> Dict[str, object]:
        """Read one versioned experimental stimulus."""

        return self._experimental_stimulus(seed_index)

    def _calibrate_scenario_from_seed_payloads(
        self,
        payloads: List[Dict[str, object]],
    ) -> None:
        """Use the batch mean of LLM-scored seed texts as scenario parameters."""

        if self._scenario_override is not None:
            self.strategy_parameter_source = "experiment_override"
            return

        parameters = aggregate_strategy_parameters(payloads)
        self.scenario = ScenarioConfig(
            scene_name=self.scenario_type,
            memory_boost=parameters["memory_boost"],
            share_boost=parameters["share_boost"],
            risk_boost=parameters["risk_boost"],
        )
        self.strategy_parameter_source = "published_stimulus_batch_mean"

    def _initialize_seed_nodes(self) -> List[int]:
        seed_content_count = self._required_seed_content_count()
        seed_payloads = [
            self._build_system_misinformation_payload(seed_index)
            for seed_index in range(seed_content_count)
        ]
        self._calibrate_scenario_from_seed_payloads(seed_payloads)

        seed_content_ids: List[str] = []
        for payload in seed_payloads:
            seed_content = self._create_content_from_generated_payload(
                payload=payload,
                source_agent_id=-1,
                source_role="system_misinformation_source",
                step=0,
            )
            seed_content.metadata["initial_seed_content"] = True
            self.content_pool.append(seed_content)
            self._register_content(seed_content)
            seed_content_ids.append(seed_content.content_id)

        for index, node_id in enumerate(sorted(self.initial_misinfo_exposed_nodes)):
            self.initial_content_assignments[int(node_id)] = seed_content_ids[index % len(seed_content_ids)]

        return sorted(int(node_id) for node_id in self.initial_misinfo_exposed_nodes)

    def _find_content(self, content_id: Optional[str]) -> Optional[ContentItem]:
        """根据内容 ID 查询内容对象。"""

        if not content_id:
            return None
        return self.content_index.get(content_id)

    def _collect_target_adjustment(
        self,
        adjustment_pool: Dict[str, float],
        target_content_id: Optional[str],
        action_strength: float,
    ) -> None:
        """记录本轮对目标内容的热度增量，轮末统一提交。"""

        if not target_content_id or action_strength <= 0.0:
            return
        adjustment_pool[target_content_id] = adjustment_pool.get(target_content_id, 0.0) + 0.36 * action_strength

    def _apply_target_adjustments(self, target_adjustments: Dict[str, float]) -> None:
        """把本轮累计的目标内容更新统一应用到内容池。"""

        for content_id, adjustment in target_adjustments.items():
            target_content = self._find_content(content_id)
            if target_content is None:
                continue
            target_content.hotness_score = clip_01(target_content.hotness_score + adjustment)

    @staticmethod
    def _misleading_retention_rate(action_type: str) -> float:
        return 0.90 if action_type == "repost" else 0.80

    def _create_action_generated_content(
        self,
        agent: Agent,
        decision: ActionDecision,
        target_content: Optional[ContentItem],
        generated_text: str,
        step: int,
        feed: FeedBundle,
    ) -> Optional[ContentItem]:
        """根据行为结果生成新的平台内容。"""

        if decision.action_type not in {"comment", "reply", "repost"}:
            return None
        if not generated_text.strip():
            return None

        if target_content is None:
            return None

        retention_applied = target_content.is_misinformation(self.config.misinformation_threshold)
        retention_rate = 0.0
        if retention_applied:
            retention_rate = self._misleading_retention_rate(decision.action_type)
            misleading_ratio = target_content.misleading_ratio * retention_rate
            scenario_type = target_content.scenario_type
        else:
            misleading_ratio = 0.08
            scenario_type = "normal"
        thread_root_content_id, thread_depth = self._resolve_thread_context(target_content)

        return ContentItem(
            content_id=self._next_content_id(),
            source_agent_id=agent.node_id,
            source_role=agent.persona.role_name,
            content_type=decision.action_type,
            text=generated_text.strip(),
            scenario_type=scenario_type,
            source_event_id=target_content.source_event_id,
            arousal_score=clip_01(0.55 * target_content.arousal_score + 0.45 * agent.emotion),
            misleading_ratio=clip_01(misleading_ratio),
            hotness_score=clip_01(0.18 + 0.34 * decision.action_intensity + self.scenario.share_boost),
            created_step=step,
            parent_content_id=target_content.content_id,
            thread_root_content_id=thread_root_content_id,
            thread_depth=thread_depth,
            metadata={
                "derived_from": target_content.content_id,
                "source_action_type": decision.action_type,
                "retention_applied": retention_applied,
                "retention_rate": retention_rate,
            },
        )

    def _decay_and_prune_content(self, step: int) -> None:
        """衰减内容热度并清理过时内容。"""

        kept: List[ContentItem] = []
        for item in self.content_pool:
            item.hotness_score = clip_01(item.hotness_score * 0.90)
            age = step - item.created_step
            if age <= self.config.max_content_age:
                kept.append(item)
        self.content_pool = kept

    def _eligible_content_for_agent(
        self,
        node_id: int,
        step: int,
        visible_content: List[ContentItem],
    ) -> tuple[List[ContentItem], set[str]]:
        """按首轮分配或有向网络边确定内容传播资格。"""

        if step == 0:
            assigned_content_id = self.initial_content_assignments.get(node_id)
            if assigned_content_id is None:
                return [], set()
            eligible = [item for item in visible_content if item.content_id == assigned_content_id]
            return eligible, {assigned_content_id}

        eligible = [
            item
            for item in visible_content
            if item.source_agent_id >= 0 and self.graph.has_edge(item.source_agent_id, node_id)
        ]
        return eligible, set()

    def _build_step_feeds(self, step: int) -> tuple[Dict[int, FeedBundle], Dict[str, object]]:
        """先判断传播资格，再为每个 agent 构建实际暴露内容。"""

        recent_actions_by_content = self._index_recent_actions_by_content()
        visible_content = [
            item
            for item in self.content_pool
            if 0 <= step - item.created_step <= self.config.max_content_age
        ]
        feeds: Dict[int, FeedBundle] = {}
        candidate_content_ids: set[str] = set()
        candidate_agent_content_pairs = 0
        for node_id, agent in self.agents.items():
            eligible_content, forced_content_ids = self._eligible_content_for_agent(
                int(node_id), step, visible_content
            )
            candidate_content_ids.update(item.content_id for item in eligible_content)
            candidate_agent_content_pairs += len(eligible_content)
            if step > 0:
                for item in eligible_content:
                    assert item.source_agent_id >= 0, "Post-initial content must have an agent publisher."
                    assert self.graph.has_edge(item.source_agent_id, int(node_id)), (
                        "Eligible receiver is not a successor of the content publisher."
                    )
            feeds[int(node_id)] = build_agent_feed(
                agent,
                self.graph,
                eligible_content,
                recent_actions_by_content,
                self.config,
                step,
                forced_content_ids=forced_content_ids,
                content_lookup=self.content_index,
            )
        exposed_content_ids = {
            item.content_id
            for feed in feeds.values()
            for item in feed.items
        }
        return feeds, {
            "content_pool_size": len(self.content_pool),
            "visible_content_count": len(visible_content),
            "candidate_content_count": len(candidate_content_ids),
            "candidate_agent_content_pairs": candidate_agent_content_pairs,
            "exposed_content_ids": exposed_content_ids,
        }

    @staticmethod
    def _content_score_summary(items: List[ContentItem]) -> Dict[str, float]:
        if not items:
            return {
                "mean_misleading_ratio": 0.0,
                "mean_arousal_score": 0.0,
                "max_misleading_ratio": 0.0,
                "max_arousal_score": 0.0,
            }
        misleading = np.asarray([item.misleading_ratio for item in items], dtype=float)
        arousal = np.asarray([item.arousal_score for item in items], dtype=float)
        return {
            "mean_misleading_ratio": float(np.mean(misleading)),
            "mean_arousal_score": float(np.mean(arousal)),
            "max_misleading_ratio": float(np.max(misleading)),
            "max_arousal_score": float(np.max(arousal)),
        }

    @staticmethod
    def _emotion_distribution(emotions: List[float]) -> Dict[str, float]:
        if not emotions:
            return {
                "emotion_min": 0.0,
                "emotion_max": 0.0,
                "emotion_mean": 0.0,
                "emotion_median": 0.0,
                "emotion_p75": 0.0,
                "emotion_p90": 0.0,
            }
        values = np.asarray(emotions, dtype=float)
        return {
            "emotion_min": float(np.min(values)),
            "emotion_max": float(np.max(values)),
            "emotion_mean": float(np.mean(values)),
            "emotion_median": float(np.median(values)),
            "emotion_p75": float(np.quantile(values, 0.75)),
            "emotion_p90": float(np.quantile(values, 0.90)),
        }

    @staticmethod
    def _effective_retention(items: List[ContentItem]) -> tuple[float, int]:
        applied_rates = [
            float(item.metadata.get("retention_rate", 0.0))
            for item in items
            if bool(item.metadata.get("retention_applied", False))
            and item.content_type in {"comment", "reply", "repost"}
        ]
        if not applied_rates:
            return 0.0, 0
        return float(np.mean(np.asarray(applied_rates, dtype=float))), len(applied_rates)

    def _index_recent_actions_by_content(self) -> Dict[str, List[ActionEvent]]:
        index: Dict[str, List[ActionEvent]] = {}
        for action in self.recent_actions:
            if action.target_content_id:
                index.setdefault(action.target_content_id, []).append(action)
            if action.generated_content_id:
                index.setdefault(action.generated_content_id, []).append(action)
        return index

    @staticmethod
    def _active_nodes_from_feeds(feeds: Dict[int, FeedBundle]) -> set[int]:
        """实际暴露即激活。"""

        return {int(node_id) for node_id, feed in feeds.items() if feed.items}

    def _runtime_llm_enabled(self) -> bool:
        """运行时 LLM 开启后，所有实际获得 Feed 的 Agent 均参与批量推理。"""

        return bool(self.config.runtime_llm_enabled and self.llm_client.has_api_key())

    @staticmethod
    def _content_prompt_text(item: Optional[ContentItem], limit: int = 220) -> str:
        if item is None:
            return "无"
        text = " ".join(item.text.split())
        return text if len(text) <= limit else text[:limit] + "..."

    def _build_perception_batch_record(
        self,
        node_id: int,
        feed: FeedBundle,
    ) -> Dict[str, object]:
        agent = self.agents[node_id]
        focus = agent._find_focus_content(feed, None)
        base_social_pressure = clip_01(self.scenario.share_boost)
        return {
            "agent_id": node_id,
            "user_profile": agent.persona.short_profile(),
            "current_emotion": round(float(agent.emotion), 4),
            "sensitivity": round(float(agent.sensitivity), 4),
            "interaction_tendency": round(float(agent.interaction_tendency), 4),
            "root_post": self._content_prompt_text(feed.root_post_item or focus),
            "focus_content": self._content_prompt_text(focus),
            "interaction_context": self._content_prompt_text(feed.interaction_context_item),
            "base_social_pressure": round(float(base_social_pressure), 4),
        }

    def _build_decision_batch_record(
        self,
        node_id: int,
        feed: FeedBundle,
        perception: PerceptionResult,
    ) -> Dict[str, object]:
        agent = self.agents[node_id]
        focus = agent._find_focus_content(feed, perception.focus_content_id)
        candidates = agent._candidate_actions_for_focus(
            focus,
            self.config.misinformation_threshold,
        )
        activation_score = clip_01(
            0.50 * agent.emotion
            + 0.30 * agent.interaction_tendency
            + 0.20 * perception.social_pressure
        )
        return {
            "agent_id": node_id,
            "user_profile": agent.persona.short_profile(),
            "updated_emotion": round(float(agent.emotion), 4),
            "interaction_tendency": round(float(agent.interaction_tendency), 4),
            "social_pressure": round(float(perception.social_pressure), 4),
            "activation_score": round(float(activation_score), 4),
            "drive_threshold": round(float(agent.persona.drive_threshold), 4),
            "emotion_distribution": {
                key: round(float(value), 4)
                for key, value in perception.emotion_distribution.items()
            },
            "focus_content": self._content_prompt_text(focus),
            "candidate_actions": [
                ACTION_TYPE_TO_CODE[action]
                for action in candidates
            ],
        }

    @staticmethod
    def _chunk_records(
        records: List[Dict[str, object]],
        batch_size: int,
    ) -> List[List[Dict[str, object]]]:
        return [
            records[index : index + batch_size]
            for index in range(0, len(records), batch_size)
        ]

    def _request_batched_agent_payloads(
        self,
        records: List[Dict[str, object]],
        prompt_builder: Callable[[List[Dict[str, object]]], str],
        temperature: float,
    ) -> Dict[int, object]:
        """并行请求一轮批次；条目级校验与重试由具体阶段负责。"""

        if not records or not self._runtime_llm_enabled():
            return {}

        batch_size = max(1, int(self.config.llm_agent_batch_size))

        def request_round(
            pending_records: List[Dict[str, object]],
        ) -> Dict[int, object]:
            batches = self._chunk_records(pending_records, batch_size)
            round_results: Dict[int, object] = {}
            worker_count = max(1, min(self.config.parallel_workers, len(batches)))
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(
                        self.llm_client.generate_json_batch,
                        prompt_builder(batch),
                        [int(row["agent_id"]) for row in batch],
                        temperature,
                    ): batch
                    for batch in batches
                }
                for future in as_completed(futures):
                    round_results.update(future.result())
            return round_results

        return request_round(records)

    @staticmethod
    def _sample_agent_state_rows(
        agent_state_rows: List[Dict[str, object]],
        max_agents: int = 5,
    ) -> List[Dict[str, object]]:
        if not agent_state_rows:
            return []

        per_agent: Dict[int, List[Dict[str, object]]] = {}
        for row in agent_state_rows:
            agent_id = int(row["agent_id"])
            per_agent.setdefault(agent_id, []).append(row)

        summaries: List[tuple[int, int, int, int]] = []
        for agent_id, rows in per_agent.items():
            active_rows = [row for row in rows if bool(row["active"])]
            if not active_rows:
                continue
            first_active_step = min(int(row["step"]) for row in active_rows)
            active_steps = sum(1 for row in rows if bool(row["active"]))
            interaction_steps = sum(1 for row in rows if float(row["action_strength"]) > 0.0)
            summaries.append((agent_id, first_active_step, active_steps, interaction_steps))

        summaries.sort(key=lambda item: (item[1], -item[2], -item[3], item[0]))
        selected_agent_ids = [agent_id for agent_id, _, _, _ in summaries[:max_agents]]

        selected_rows: List[Dict[str, object]] = []
        for agent_id in selected_agent_ids:
            rows = sorted(per_agent[agent_id], key=lambda row: int(row["step"]))
            first_active_step = min(int(row["step"]) for row in rows if bool(row["active"]))
            selected_rows.extend(row for row in rows if int(row["step"]) >= first_active_step)
        return selected_rows

    def _compute_inactive_agent_step(self, step: int, node_id: int, agent: Agent) -> AgentStepResult:
        working_agent = agent
        working_agent.decay_perceived_emotion_distribution(
            self.config.emotion_structure_decay_rate
        )
        working_agent.last_interaction_strength = 0.0
        working_agent.last_stimulus = 0.0
        working_agent.last_feedback = 0.0
        working_agent.last_action_type = "inactive"

        perception = PerceptionResult(
            focus_content_id=None,
            negative_emotion_delta=0.0,
            social_pressure=0.0,
            emotion_structure_source="natural_decay",
            reason="本轮未进入平台信息处理流程。",
            impression_text="",
        )
        decision = ActionDecision(
            action_type="inactive",
            target_content_id=None,
            action_intensity=0.0,
            reason="本轮未激活。",
        )
        return AgentStepResult(
            node_id=node_id,
            updated_agent=working_agent,
            perception=perception,
            decision=decision,
            generated_text="",
            action_strength=0.0,
            generated_content=None,
            action_event=None,
            target_content_id=None,
            feed_misinfo_exposed=False,
            active=False,
            used_runtime_llm=False,
        )

    def _finalize_active_agent_step(
        self,
        step: int,
        node_id: int,
        agent: Agent,
        feed: FeedBundle,
        perception: PerceptionResult,
        decision: ActionDecision,
        generated_text: str,
        used_runtime_llm: bool,
    ) -> AgentStepResult:
        """把已完成感知与决策的 Agent 结果转换为传播事件。"""

        working_agent = agent
        target_content = self._find_content(decision.target_content_id)

        action_strength = 0.0
        generated_content = None
        action_event = None
        if working_agent.should_interact(decision):
            action_weight = self.config.action_weights.get(decision.action_type, 0.0)
            action_strength = clip_01(action_weight * decision.action_intensity)
            working_agent.last_interaction_strength = action_strength
            generated_content = self._create_action_generated_content(
                agent=working_agent,
                decision=decision,
                target_content=target_content,
                generated_text=generated_text,
                step=step,
                feed=feed,
            )
            action_event = ActionEvent(
                actor_id=node_id,
                actor_role=working_agent.persona.role_name,
                action_type=decision.action_type,
                target_content_id=decision.target_content_id,
                generated_content_id=generated_content.content_id if generated_content is not None else None,
                action_intensity=action_strength,
                step=step,
            )
        else:
            working_agent.last_interaction_strength = 0.0

        return AgentStepResult(
            node_id=node_id,
            updated_agent=working_agent,
            perception=perception,
            decision=decision,
            generated_text=generated_text,
            action_strength=action_strength,
            generated_content=generated_content,
            action_event=action_event,
            target_content_id=decision.target_content_id,
            feed_misinfo_exposed=feed.misinfo_exposed,
            active=True,
            used_runtime_llm=used_runtime_llm,
        )

    def _compute_batched_active_agent_steps(
        self,
        step: int,
        active_nodes: set[int],
        feeds: Dict[int, FeedBundle],
        previous_emotions: Dict[int, float],
        previous_interaction_strengths: Dict[int, float],
    ) -> List[AgentStepResult]:
        """按感知、公式更新、决策三个阶段同步计算全部活跃 Agent。"""

        if not active_nodes:
            return []

        ordered_nodes = sorted(active_nodes)
        runtime_llm = self._runtime_llm_enabled()
        self.runtime_llm_diagnostics["active_agent_steps"] += len(ordered_nodes)
        agent_rngs = {
            node_id: np.random.default_rng(
                self.config.random_seed + (step + 1) * 100_000 + node_id
            )
            for node_id in ordered_nodes
        }

        llm_eligible_nodes: set[int] = set()
        llm_qualified_nodes: set[int] = set()
        if runtime_llm:
            for node_id in ordered_nodes:
                agent = self.agents[node_id]
                feed = feeds[node_id]
                if not feed.misinfo_exposed:
                    self.runtime_llm_diagnostics[
                        "llm_disqualified_no_misinfo_items"
                    ] += 1
                    continue
                focus = agent._find_focus_content(feed, None)
                if focus is None:
                    self.runtime_llm_diagnostics[
                        "llm_disqualified_no_focus_items"
                    ] += 1
                    continue
                llm_eligible_nodes.add(node_id)

        if runtime_llm and llm_eligible_nodes:
            def capacity_priority(node_id: int) -> tuple[int, int]:
                deterministic_rank = (
                    (node_id + 1) * 1_103_515_245
                    + (self.config.random_seed + 1) * 12_345
                ) % 2_147_483_647
                return step, deterministic_rank

            ordered_eligible = sorted(llm_eligible_nodes, key=capacity_priority)
            cap = max(0, int(self.config.llm_agent_cap_per_step))
            admitted = ordered_eligible if cap == 0 else ordered_eligible[:cap]
            llm_qualified_nodes = set(admitted)
            deferred_count = len(llm_eligible_nodes) - len(llm_qualified_nodes)
            self.runtime_llm_diagnostics["llm_eligible_before_cap_items"] += len(
                llm_eligible_nodes
            )
            self.runtime_llm_diagnostics["llm_admitted_after_cap_items"] += len(
                llm_qualified_nodes
            )
            self.runtime_llm_diagnostics["llm_deferred_by_cap_items"] += deferred_count
            self.runtime_llm_diagnostics["llm_qualified_items"] += len(
                llm_qualified_nodes
            )

        for node_id in ordered_nodes:
            self.agents[node_id].record_feed(feeds[node_id])

        perception_records = [
            self._build_perception_batch_record(node_id, feeds[node_id])
            for node_id in ordered_nodes
            if node_id in llm_qualified_nodes
        ]
        self.runtime_llm_diagnostics["perception_request_items"] += len(
            perception_records
        )
        perception_payloads = self._request_batched_agent_payloads(
            perception_records,
            build_batch_perception_prompt,
            self.llm_config.perception_temperature,
        )
        invalid_perception_records = []
        if runtime_llm:
            for record in perception_records:
                node_id = int(record["agent_id"])
                agent = self.agents[node_id]
                base_social_pressure = clip_01(self.scenario.share_boost)
                if agent.parse_llm_perception_payload(
                    perception_payloads.get(node_id),
                    base_social_pressure,
                ) is None:
                    invalid_perception_records.append(record)
        if invalid_perception_records:
            self.runtime_llm_diagnostics["perception_retry_items"] += len(
                invalid_perception_records
            )
            perception_payloads.update(
                self._request_batched_agent_payloads(
                    invalid_perception_records,
                    build_batch_perception_prompt,
                    self.llm_config.perception_temperature,
                )
            )

        perceptions: Dict[int, PerceptionResult] = {}
        focus_contents: Dict[int, Optional[ContentItem]] = {}
        used_llm_perception: Dict[int, bool] = {}
        perception_sources: Dict[int, str] = {}
        perception_valid_for_cycle: set[int] = set()
        perception_fallback_count = 0
        for node_id in ordered_nodes:
            agent = self.agents[node_id]
            feed = feeds[node_id]
            base_social_pressure = clip_01(self.scenario.share_boost)
            valid_payload = None
            if runtime_llm and node_id in llm_qualified_nodes:
                payload = perception_payloads.get(node_id)
                if (
                    agent.parse_llm_perception_payload(
                        payload,
                        base_social_pressure,
                    )
                    is not None
                ):
                    valid_payload = payload
                    perception_valid_for_cycle.add(node_id)
                    perception_sources[node_id] = "llm_refresh"
                    self.runtime_llm_diagnostics["perception_refresh_items"] += 1
                else:
                    perception_sources[node_id] = "pending_or_retained"
                    perception_fallback_count += 1
                    self.runtime_llm_diagnostics["perception_rule_fallback_items"] += 1
            else:
                perception_sources[node_id] = (
                    "natural_decay"
                    if not feed.misinfo_exposed
                    else "pending_or_retained"
                )
            used_llm = bool(runtime_llm and valid_payload is not None)
            perception = agent.perceive_feed(
                feed,
                self.llm_client,
                self.config,
                self.scenario,
                use_llm_perception=used_llm,
                llm_payload=valid_payload,
                emotion_structure_source=perception_sources[node_id],
            )
            network_feedback = interaction_feedback(
                neighbor_ids=[
                    int(source) for source in self.graph.predecessors(node_id)
                ],
                previous_interaction_strengths=previous_interaction_strengths,
                previous_emotions=previous_emotions,
            )
            focus_contents[node_id] = agent.apply_perception_update(
                feed,
                perception,
                self.scenario,
                network_feedback,
                self.config,
            )
            perceptions[node_id] = perception
            used_llm_perception[node_id] = used_llm

        decision_records = [
            self._build_decision_batch_record(
                node_id,
                feeds[node_id],
                perceptions[node_id],
            )
            for node_id in ordered_nodes
            if node_id in llm_qualified_nodes
            and node_id in perception_valid_for_cycle
        ]
        self.runtime_llm_diagnostics["decision_request_items"] += len(
            decision_records
        )
        if runtime_llm:
            self.runtime_llm_diagnostics[
                "decision_blocked_by_invalid_perception_items"
            ] += len(llm_qualified_nodes - perception_valid_for_cycle)
        decision_payloads = self._request_batched_agent_payloads(
            decision_records,
            build_batch_action_decision_prompt,
            self.llm_config.perception_temperature,
        )
        invalid_decision_records = []
        if runtime_llm:
            for record in decision_records:
                node_id = int(record["agent_id"])
                if self.agents[node_id].parse_llm_action_payload(
                    feeds[node_id],
                    perceptions[node_id],
                    self.config,
                    decision_payloads.get(node_id),
                ) is None:
                    invalid_decision_records.append(record)
        if invalid_decision_records:
            self.runtime_llm_diagnostics["decision_retry_items"] += len(
                invalid_decision_records
            )
            decision_payloads.update(
                self._request_batched_agent_payloads(
                    invalid_decision_records,
                    build_batch_action_decision_prompt,
                    self.llm_config.perception_temperature,
                )
            )

        results: List[AgentStepResult] = []
        decision_fallback_count = 0
        for node_id in ordered_nodes:
            agent = self.agents[node_id]
            valid_payload = None
            if (
                runtime_llm
                and node_id in llm_qualified_nodes
                and node_id in perception_valid_for_cycle
            ):
                payload = decision_payloads.get(node_id)
                if (
                    agent.parse_llm_action_payload(
                        feeds[node_id],
                        perceptions[node_id],
                        self.config,
                        payload,
                    )
                    is not None
                ):
                    valid_payload = payload
                    self.runtime_llm_diagnostics["decision_refresh_items"] += 1
                else:
                    decision_fallback_count += 1
                    self.runtime_llm_diagnostics["decision_rule_fallback_items"] += 1
            elif runtime_llm:
                self.runtime_llm_diagnostics["decision_rule_execution_items"] += 1
            decision, used_llm_decision = agent.decide_action(
                feeds[node_id],
                perceptions[node_id],
                self.config,
                self.scenario,
                agent_rngs[node_id],
                self.llm_client,
                use_runtime_llm=bool(runtime_llm and valid_payload is not None),
                llm_payload=valid_payload,
            )
            generated_text = agent.finalize_decision(
                feeds[node_id],
                focus_contents[node_id],
                decision,
                self.llm_client,
                self.config,
                self.llm_config,
                step,
                agent_rngs[node_id],
            )
            results.append(
                self._finalize_active_agent_step(
                    step,
                    node_id,
                    agent,
                    feeds[node_id],
                    perceptions[node_id],
                    decision,
                    generated_text,
                    used_runtime_llm=(
                        used_llm_perception[node_id] and used_llm_decision
                    ),
                )
            )

        if runtime_llm:
            self.llm_client.record_fallback_items(
                perception_fallback_count + decision_fallback_count
            )
        return results

    def run(self) -> SimulationResult:
        """执行完整仿真。"""

        group_negative_emotion: List[float] = []
        high_negative_ratio: List[float] = []
        group_exposure_rate: List[float] = []
        group_activation_rate: List[float] = []
        group_dispersion: List[float] = []
        exposure_counts: List[int] = []
        interaction_counts: List[int] = []
        active_counts: List[int] = []
        active_rate: List[float] = []
        time_series: List[Dict[str, float]] = []
        action_types = ["inactive", "ignore", "like", "comment", "reply", "repost"]
        action_counts = {action_type: 0 for action_type in action_types}
        participation_keys = list(NORMAL_PERSONA_KEYS)
        participation_counts = {key: 0 for key in participation_keys}
        participation_action_counts = {key: 0 for key in participation_keys}
        max_interaction_case_records = 120
        interaction_cases: List[Dict[str, object]] = []
        agent_state_rows: List[Dict[str, object]] = []
        initial_seed_contents = [
            item
            for item in self.content_pool
            if item.is_misinformation(self.config.misinformation_threshold) and bool(item.metadata.get("initial_seed_content"))
        ][:3]
        for item in initial_seed_contents:
            interaction_cases.append(
                {
                    "record_type": "initial_misinformation",
                    "scenario_type": self.scenario_type,
                    "source_event_id": item.source_event_id,
                    "step": 0,
                    "content_id": item.content_id,
                    "content_type": item.content_type,
                    "content_text": item.text,
                    "arousal_score": item.arousal_score,
                    "misleading_ratio": item.misleading_ratio,
                    "text_generated_by": item.metadata.get("text_generated_by", "rule"),
                    "scores_generated_by": item.metadata.get("scores_generated_by", "rule"),
                }
            )
        for agent in self.agents.values():
            mode = agent.participation_mode if agent.participation_mode in participation_counts else agent.persona.persona_key
            participation_counts[mode] += 1

        previous_emotions = {node_id: agent.emotion for node_id, agent in self.agents.items()}
        previous_interaction_strengths = {
            node_id: agent.last_interaction_strength for node_id, agent in self.agents.items()
        }

        self._emit_progress(
            "simulation_start",
            num_agents=self.config.num_agents,
            num_steps=self.config.num_steps,
            seed_nodes=list(self.seed_nodes),
            source_event_id=self.active_source_event.source_event_id,
        )

        for step in range(self.config.num_steps):
            self._emit_progress("step_start", step=step, total_steps=self.config.num_steps)

            # 轮初冻结 feed，避免同轮状态串扰。
            feeds, feed_diagnostics = self._build_step_feeds(step)
            active_nodes = self._active_nodes_from_feeds(feeds)
            self._emit_progress("step_interaction", step=step, total_steps=self.config.num_steps)

            step_results = [
                self._compute_inactive_agent_step(step, int(node_id), agent)
                for node_id, agent in self.agents.items()
                if int(node_id) not in active_nodes
            ]
            target_adjustments: Dict[str, float] = {}
            try:
                step_results.extend(
                    self._compute_batched_active_agent_steps(
                        step,
                        active_nodes,
                        feeds,
                        previous_emotions,
                        previous_interaction_strengths,
                    )
                )
            except Exception as error:
                raise RuntimeError(
                    f"第 {step + 1} 轮批量 Agent 计算失败：{error}"
                ) from error
            self._emit_progress(
                "agent_done",
                step=step,
                total_steps=self.config.num_steps,
                finished_agents=len(active_nodes),
                total_agents=len(self.agents),
                parallel_workers=min(
                    self.config.parallel_workers,
                    max(
                        1,
                        int(np.ceil(len(active_nodes) / self.config.llm_agent_batch_size)),
                    ),
                ),
            )

            # 固定提交顺序。
            step_results.sort(key=lambda item: item.node_id)
            for result in step_results:
                self._collect_target_adjustment(
                    adjustment_pool=target_adjustments,
                    target_content_id=result.target_content_id,
                    action_strength=result.action_strength,
                )

            # 轮末统一提交，保持同步仿真。
            updated_agents: Dict[int, Agent] = {}
            new_actions: List[ActionEvent] = []
            new_contents: List[ContentItem] = []

            step_exposure_flags: List[bool] = []
            step_interaction_strengths: List[float] = []
            step_stimulus_values: List[float] = []
            step_feedback_values: List[float] = []
            step_memory_emotions: List[float] = []
            step_active_flags: List[bool] = []
            step_action_counts = {action_type: 0 for action_type in action_types}
            step_participation_action_counts = {key: 0 for key in participation_keys}
            distribution_labels = ["anxiety", "anger", "suspicion", "none"]

            for result in step_results:
                updated_agents[result.node_id] = result.updated_agent
                action_type = result.decision.action_type
                if action_type not in step_action_counts:
                    action_type = "ignore"
                step_action_counts[action_type] += 1
                action_counts[action_type] += 1
                participation_mode = result.updated_agent.participation_mode
                if participation_mode not in step_participation_action_counts:
                    participation_mode = result.updated_agent.persona.persona_key
                if action_type not in {"ignore", "inactive"}:
                    step_participation_action_counts[participation_mode] += 1
                    participation_action_counts[participation_mode] += 1
                if result.action_event is not None:
                    new_actions.append(result.action_event)
                if result.generated_content is not None:
                    target_content = self._find_content(result.target_content_id)
                    assert result.generated_content.source_agent_id == result.node_id, (
                        "Derived content publisher_id does not match the acting agent."
                    )
                    if result.decision.action_type == "reply":
                        assert target_content is not None, "Reply requires an existing parent content item."
                        assert target_content.content_type in {"comment", "reply", "repost"}, (
                            "Reply target is not a replyable conversation item."
                        )
                        assert result.generated_content.parent_content_id == target_content.content_id, (
                            "Reply parent_content_id does not match its target content."
                        )
                    new_contents.append(result.generated_content)
                    if (
                        result.decision.action_type in {"comment", "reply", "repost"}
                        and len(interaction_cases) < max_interaction_case_records
                    ):
                        interaction_cases.append(
                            {
                                "record_type": "interaction",
                                "scenario_type": self.scenario_type,
                                "source_event_id": result.generated_content.source_event_id,
                                "step": step,
                                "agent_id": result.node_id,
                                "persona_key": result.updated_agent.persona.persona_key,
                                "role_name": result.updated_agent.persona.role_name,
                                "used_runtime_llm": result.used_runtime_llm,
                                "action_type": result.decision.action_type,
                                "action_strength": result.action_strength,
                                "emotion_after": result.updated_agent.emotion,
                                "emotion_anxiety": result.perception.emotion_distribution.get("anxiety", 0.0),
                                "emotion_anger": result.perception.emotion_distribution.get("anger", 0.0),
                                "emotion_suspicion": result.perception.emotion_distribution.get("suspicion", 0.0),
                                "emotion_none": result.perception.emotion_distribution.get("none", 0.0),
                                "emotion_structure_source": result.perception.emotion_structure_source,
                                "social_pressure": result.perception.social_pressure,
                                "stimulus": result.updated_agent.last_stimulus,
                                "feedback": result.updated_agent.last_feedback,
                                "target_content_id": result.target_content_id or "",
                                "target_content_type": target_content.content_type if target_content is not None else "",
                                "target_agent_id": (
                                    target_content.source_agent_id
                                    if target_content is not None and result.decision.action_type == "reply"
                                    else -1
                                ),
                                "target_is_misinformation": bool(
                                    target_content
                                    and target_content.is_misinformation(self.config.misinformation_threshold)
                                ),
                                "generated_content_id": result.generated_content.content_id,
                                "generated_text": result.generated_text,
                            }
                        )

                step_exposure_flags.append(result.feed_misinfo_exposed)
                step_interaction_strengths.append(result.updated_agent.last_interaction_strength)
                step_stimulus_values.append(result.updated_agent.last_stimulus)
                step_feedback_values.append(result.updated_agent.last_feedback)
                step_memory_emotions.append(result.updated_agent.avg_past_emotion())
                step_active_flags.append(result.active)
                agent_state_rows.append(
                    {
                        "scenario_type": self.scenario_type,
                        "step": step,
                        "agent_id": result.node_id,
                        "persona_key": result.updated_agent.persona.persona_key,
                        "role_name": result.updated_agent.persona.role_name,
                        "active": result.active,
                        "used_runtime_llm": result.used_runtime_llm,
                        "misinfo_exposed": result.feed_misinfo_exposed,
                        "focus_content_id": result.perception.focus_content_id or "",
                        "emotion_anxiety": float(result.updated_agent.perceived_emotion_distribution.get("anxiety", 0.0)),
                        "emotion_anger": float(result.updated_agent.perceived_emotion_distribution.get("anger", 0.0)),
                        "emotion_suspicion": float(result.updated_agent.perceived_emotion_distribution.get("suspicion", 0.0)),
                        "emotion_none": float(result.updated_agent.perceived_emotion_distribution.get("none", 0.0)),
                        "emotion_structure_source": result.perception.emotion_structure_source,
                        "emotion_before": float(previous_emotions.get(result.node_id, result.updated_agent.emotion)),
                        "emotion_after": float(result.updated_agent.emotion),
                        "negative_emotion_delta": float(result.perception.negative_emotion_delta),
                        "social_pressure": float(result.perception.social_pressure),
                        "action_type": result.decision.action_type,
                        "action_intensity": float(result.decision.action_intensity),
                        "action_strength": float(result.action_strength),
                        "stimulus": float(result.updated_agent.last_stimulus),
                        "feedback": float(result.updated_agent.last_feedback),
                        "generated_text": result.generated_text,
                    }
                )
                if result.decision.action_type == "repost":
                    target_content = self._find_content(result.target_content_id)
                    assert "repost" in result.updated_agent.persona.allowed_actions, (
                        "Repost was selected for an agent without repost permission."
                    )
                    assert target_content is not None and target_content.is_misinformation(
                        self.config.misinformation_threshold
                    ), (
                        "Repost requires an existing misinformation target."
                    )

            self.agents = updated_agents
            emotion_distribution_sums = {
                label: sum(
                    float(agent.perceived_emotion_distribution.get(label, 0.0))
                    for agent in self.agents.values()
                )
                for label in distribution_labels
            }
            emotion_distribution_total = len(self.agents)
            self._apply_target_adjustments(target_adjustments)
            for item in new_contents:
                self._register_content(item)
            self.content_pool.extend(new_contents)
            self.recent_actions.extend(new_actions)
            self.recent_actions = [action for action in self.recent_actions if step - action.step <= self.config.max_content_age]
            self._decay_and_prune_content(step)

            emotions = [agent.emotion for agent in self.agents.values()]
            group_value = group_mean_negative_emotion(emotions)
            high_ratio_value = compute_high_negative_ratio(emotions, self.config.high_negative_threshold)
            medium_high_threshold = min(
                self.config.medium_high_negative_threshold,
                self.config.high_negative_threshold,
            )
            medium_high_ratio_value = compute_high_negative_ratio(emotions, medium_high_threshold)
            exposure_rate_value = compute_group_exposure_rate(step_exposure_flags)
            activation_rate_value = compute_group_activation_rate(step_active_flags)
            interaction_rate_value = compute_interaction_rate(step_interaction_strengths)
            dispersion_value = compute_group_dispersion(emotions)
            active_count = int(sum(1 for flag in step_active_flags if flag))
            active_rate_value = (
                float(active_count / len(step_active_flags)) if step_active_flags else 0.0
            )
            exposure_success_count = int(sum(1 for flag in step_exposure_flags if flag))
            exposure_success_rate_value = (
                float(exposure_success_count / len(step_exposure_flags)) if step_exposure_flags else 0.0
            )
            exposure_activation_count = active_count
            exposure_activation_rate_value = active_rate_value
            memory_activation_count = 0
            current_interaction_count = int(sum(1 for value in step_interaction_strengths if value > 0.0))
            assert exposure_success_count <= exposure_activation_count
            assert current_interaction_count <= exposure_activation_count + memory_activation_count
            if exposure_activation_count == 0 and memory_activation_count == 0:
                assert current_interaction_count == 0

            generated_action_counts = {
                action_type: sum(1 for item in new_contents if item.content_type == action_type)
                for action_type in {"comment", "reply", "repost"}
            }
            for action_type, generated_count in generated_action_counts.items():
                assert generated_count <= step_action_counts[action_type], (
                    f"Generated {action_type} content exceeds selected {action_type} actions."
                )
            derived_content_count = len(new_contents)
            assert derived_content_count == sum(generated_action_counts.values())

            repost_permission_exposure_count = 0
            repost_threshold_eligible_count = 0
            repost_blocked_by_reply_precedence_count = 0
            reply_target_available_count = 0
            for result in step_results:
                feed = feeds[result.node_id]
                focus = feed.items[0] if feed.items else None
                allowed_actions = set(result.updated_agent.persona.allowed_actions)
                if focus is not None and "reply" in allowed_actions and focus.content_type in {"comment", "reply", "repost"}:
                    reply_target_available_count += 1
                if (
                    focus is None
                    or "repost" not in allowed_actions
                    or not focus.is_misinformation(self.config.misinformation_threshold)
                ):
                    continue
                repost_permission_exposure_count += 1
                activation_score = clip_01(
                    0.50 * result.updated_agent.emotion
                    + 0.30 * result.updated_agent.interaction_tendency
                    + 0.20 * result.perception.social_pressure
                )
                repost_threshold = clip_01(result.updated_agent.persona.drive_threshold + 0.38)
                if activation_score < repost_threshold:
                    continue
                repost_threshold_eligible_count += 1
                if "reply" in allowed_actions and focus.content_type in {"comment", "reply", "repost"}:
                    repost_blocked_by_reply_precedence_count += 1

            current_content = [
                item
                for item in self.content_pool
                if item.content_id in feed_diagnostics["exposed_content_ids"]
            ]
            current_content_scores = self._content_score_summary(current_content)
            derived_content_scores = self._content_score_summary(new_contents)
            initial_content_scores = self._content_score_summary(initial_seed_contents)
            pool_misleading_values = np.asarray(
                [item.misleading_ratio for item in self.content_pool], dtype=float
            )
            misleading_stock_total = float(np.sum(pool_misleading_values)) if pool_misleading_values.size else 0.0
            misleading_stock_mean = float(np.mean(pool_misleading_values)) if pool_misleading_values.size else 0.0
            effective_retention, retention_applied_count = self._effective_retention(new_contents)
            emotion_distribution = self._emotion_distribution(emotions)
            emotion_distribution_means = {
                f"perceived_{label}_mean": (
                    float(emotion_distribution_sums[label] / emotion_distribution_total)
                    if emotion_distribution_total
                    else 0.0
                )
                for label in distribution_labels
            }

            previous_group_value = group_negative_emotion[-1] if group_negative_emotion else group_value
            step_emotion_increment = float(group_value - previous_group_value)
            group_negative_emotion.append(group_value)
            high_negative_ratio.append(high_ratio_value)
            group_exposure_rate.append(exposure_rate_value)
            group_activation_rate.append(activation_rate_value)
            group_dispersion.append(dispersion_value)
            active_counts.append(active_count)
            active_rate.append(active_rate_value)
            exposure_counts.append(int(sum(1 for flag in step_exposure_flags if flag)))
            interaction_counts.append(current_interaction_count)
            step_row: Dict[str, float] = {
                    "step": float(step),
                    "group_negative_emotion": group_value,
                    "step_emotion_increment": step_emotion_increment,
                    "high_negative_ratio": high_ratio_value,
                    "group_exposure_rate": exposure_rate_value,
                    "group_activation_rate": activation_rate_value,
                    "interaction_rate": interaction_rate_value,
                    "active_rate": active_rate_value,
                    "exposure_success_rate": exposure_success_rate_value,
                    "exposure_activation_rate": exposure_activation_rate_value,
                    "group_dispersion": dispersion_value,
                    "exposure_count": float(exposure_counts[-1]),
                    "misinfo_exposure_count": float(exposure_counts[-1]),
                    "emotion_distribution_total": float(emotion_distribution_total),
                    "runtime_llm_agent_count": float(
                        sum(1 for result in step_results if result.used_runtime_llm)
                    ),
                    "exposure_success_count": float(exposure_success_count),
                    "exposure_activation_count": float(exposure_activation_count),
                    "memory_activation_count": float(memory_activation_count),
                    "interaction_count": float(interaction_counts[-1]),
                    "current_interaction_count": float(current_interaction_count),
                    "active_count": float(active_count),
                    "content_pool_size": float(feed_diagnostics["content_pool_size"]),
                    "visible_content_count": float(feed_diagnostics["visible_content_count"]),
                    "candidate_content_count": float(feed_diagnostics["candidate_content_count"]),
                    "candidate_agent_content_pairs": float(feed_diagnostics["candidate_agent_content_pairs"]),
                    "inactive_count": float(step_action_counts["inactive"]),
                    "ignore_count": float(step_action_counts["ignore"]),
                    "like_count": float(step_action_counts["like"]),
                    "comment_count": float(step_action_counts["comment"]),
                    "reply_count": float(step_action_counts["reply"]),
                    "repost_count": float(step_action_counts["repost"]),
                    "derived_content_count": float(derived_content_count),
                    "derived_comment_count": float(generated_action_counts["comment"]),
                    "derived_reply_count": float(generated_action_counts["reply"]),
                    "derived_repost_count": float(generated_action_counts["repost"]),
                    "reply_target_available_count": float(reply_target_available_count),
                    "repost_permission_exposure_count": float(repost_permission_exposure_count),
                    "repost_threshold_eligible_count": float(repost_threshold_eligible_count),
                    "repost_blocked_by_reply_precedence_count": float(repost_blocked_by_reply_precedence_count),
                    "repost_probability_model_enabled": 0.0,
                    "mean_misleading_ratio_current_content": current_content_scores["mean_misleading_ratio"],
                    "mean_arousal_score_current_content": current_content_scores["mean_arousal_score"],
                    "max_misleading_ratio_current_content": current_content_scores["max_misleading_ratio"],
                    "max_arousal_score_current_content": current_content_scores["max_arousal_score"],
                    "mean_misleading_ratio_derived_content": derived_content_scores["mean_misleading_ratio"],
                    "mean_arousal_score_derived_content": derived_content_scores["mean_arousal_score"],
                    "max_misleading_ratio_derived_content": derived_content_scores["max_misleading_ratio"],
                    "max_arousal_score_derived_content": derived_content_scores["max_arousal_score"],
                    "mean_misleading_ratio_initial_content": initial_content_scores["mean_misleading_ratio"],
                    "mean_arousal_score_initial_content": initial_content_scores["mean_arousal_score"],
                    "max_misleading_ratio_initial_content": initial_content_scores["max_misleading_ratio"],
                    "max_arousal_score_initial_content": initial_content_scores["max_arousal_score"],
                    "effective_retention": effective_retention,
                    "retention_applied_count": float(retention_applied_count),
                    "misleading_stock": misleading_stock_total,
                    "misleading_stock_total": misleading_stock_total,
                    "misleading_stock_mean": misleading_stock_mean,
                    "high_negative_threshold": float(self.config.high_negative_threshold),
                    "medium_high_negative_threshold": float(medium_high_threshold),
                    "medium_high_negative_ratio": medium_high_ratio_value,
                    **emotion_distribution,
                    **emotion_distribution_means,
                    "mean_stimulus": float(np.mean(np.asarray(step_stimulus_values, dtype=float))) if step_stimulus_values else 0.0,
                    "mean_feedback": float(np.mean(np.asarray(step_feedback_values, dtype=float))) if step_feedback_values else 0.0,
                    "mean_memory_emotion": (
                        float(np.mean(np.asarray(step_memory_emotions, dtype=float))) if step_memory_emotions else 0.0
                    ),
            }
            for key in participation_keys:
                step_row[f"{key}_action_count"] = float(step_participation_action_counts[key])
            time_series.append(step_row)

            self._emit_progress(
                "step_end",
                step=step,
                total_steps=self.config.num_steps,
                group_negative_emotion=group_value,
                high_negative_ratio=high_ratio_value,
                group_exposure_rate=exposure_rate_value,
                group_activation_rate=activation_rate_value,
                active_rate=active_rate_value,
                interaction_rate=interaction_rate_value,
                group_dispersion=dispersion_value,
                exposure_count=exposure_counts[-1],
                interaction_count=interaction_counts[-1],
                active_count=active_count,
            )

            previous_emotions = {node_id: agent.emotion for node_id, agent in self.agents.items()}
            previous_interaction_strengths = {
                node_id: agent.last_interaction_strength for node_id, agent in self.agents.items()
            }

        sampled_agent_state_rows = self._sample_agent_state_rows(agent_state_rows)

        result = SimulationResult(
            seed_nodes=self.seed_nodes,
            scenario_type=self.scenario_type,
            active_source_event_id=self.active_source_event.source_event_id,
            active_source_event_summary=self.active_source_event.source_event_summary,
            group_negative_emotion=group_negative_emotion,
            high_negative_ratio=high_negative_ratio,
            group_exposure_rate=group_exposure_rate,
            group_activation_rate=group_activation_rate,
            group_dispersion=group_dispersion,
            peak=compute_peak(group_negative_emotion),
            amplification=compute_amplification(group_negative_emotion),
            recovery_gap=compute_recovery_gap(group_negative_emotion),
            mean_step_emotion_increment=compute_mean_step_emotion_increment(group_negative_emotion),
            peak_average_rise_speed=compute_peak_average_rise_speed(group_negative_emotion),
            duration=compute_duration(group_negative_emotion, self.config.high_negative_threshold),
            exposure_counts=exposure_counts,
            interaction_counts=interaction_counts,
            active_counts=active_counts,
            active_rate=active_rate,
            time_series=time_series,
            action_counts=action_counts,
            participation_counts=participation_counts,
            participation_action_counts=participation_action_counts,
            interaction_cases=interaction_cases,
            sampled_agent_state_rows=sampled_agent_state_rows,
            runtime_llm_request_count=self.llm_client.request_count,
            runtime_llm_failure_count=self.llm_client.failure_count,
            runtime_llm_batch_request_count=self.llm_client.batch_request_count,
            runtime_llm_inference_item_count=self.llm_client.inference_item_count,
            runtime_llm_valid_item_count=self.llm_client.valid_inference_item_count,
            runtime_llm_invalid_item_count=self.llm_client.invalid_inference_item_count,
            runtime_llm_fallback_item_count=self.llm_client.fallback_item_count,
            runtime_llm_input_token_count=self.llm_client.input_token_count,
            runtime_llm_output_token_count=self.llm_client.output_token_count,
            runtime_llm_estimated_cost_rmb=self.llm_client.estimated_cost_rmb(),
            runtime_llm_diagnostics=dict(self.runtime_llm_diagnostics),
            runtime_llm_model=self.llm_config.model,
            risk_boost=self.scenario.risk_boost,
            memory_boost=self.scenario.memory_boost,
            share_boost=self.scenario.share_boost,
            strategy_parameter_source=self.strategy_parameter_source,
        )
        self._emit_progress(
            "simulation_end",
            peak=result.peak,
            amplification=result.amplification,
            recovery_gap=result.recovery_gap,
            final_group_negative_emotion=result.group_negative_emotion[-1] if result.group_negative_emotion else 0.0,
        )
        return result
