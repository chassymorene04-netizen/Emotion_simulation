from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from experimental_data import is_clean_action_text
from prompts import build_action_text_prompt
from config import LLMConfig, SimulationConfig
from models import ActionDecision, AgentPersona, ContentItem, FeedBundle, PerceptionResult, ScenarioConfig
from formulas import (
    clip_01,
    effective_gamma_feedback_weight,
    effective_memory_emotion_weight,
    emotion_update,
    misinformation_stimulus,
)
from llm_client import LLMClient


ACTION_CODE_TO_TYPE = {
    0: "ignore",
    1: "like",
    2: "comment",
    3: "reply",
    4: "repost",
}
ACTION_TYPE_TO_CODE = {
    action_type: action_code
    for action_code, action_type in ACTION_CODE_TO_TYPE.items()
}
DEFAULT_ACTION_WEIGHTS = {
    "like": 0.40,
    "comment": 0.80,
    "reply": 0.70,
    "repost": 1.00,
}


@dataclass
class Agent:
    node_id: int
    persona: AgentPersona
    emotion: float
    sensitivity: float
    memory_window: int
    interaction_tendency: float
    participation_mode: str = "regular"
    stimulus_memory: List[float] = field(default_factory=list)
    feedback_memory: List[float] = field(default_factory=list)
    emotion_history: List[float] = field(default_factory=list)
    exposure_history: List[int] = field(default_factory=list)
    topic_history: List[str] = field(default_factory=list)
    style_history: List[str] = field(default_factory=list)
    action_history: List[str] = field(default_factory=list)
    last_stimulus: float = 0.0
    last_feedback: float = 0.0
    last_interaction_strength: float = 0.0
    last_action_type: str = "ignore"
    last_action_text: str = ""
    last_perception_delta: float = 0.0
    last_social_pressure: float = 0.0
    last_feed_ids: List[str] = field(default_factory=list)
    perceived_emotion_distribution: Dict[str, float] = field(
        default_factory=lambda: {
            "anxiety": 0.0,
            "anger": 0.0,
            "suspicion": 0.0,
            "none": 1.0,
        }
    )

    def _recent(self, values: List[str] | List[int] | List[float]) -> list:
        if self.memory_window <= 0:
            return []
        return list(values[-self.memory_window :])

    def repeated_exposure_history_signal(self) -> float:
        window = self._recent(self.exposure_history)
        return float(sum(int(value) for value in window) / len(window)) if window else 0.0

    def avg_past_stimulus(self) -> float:
        window = self._recent(self.stimulus_memory)
        return float(np.mean(np.asarray(window, dtype=float))) if window else 0.0

    def avg_past_emotion(self) -> float:
        window = self._recent(self.emotion_history)
        return float(np.mean(np.asarray(window, dtype=float))) if window else float(self.emotion)

    def style_repeat_ratio(self, style_name: str) -> float:
        window = self._recent(self.style_history)
        return float(sum(1 for value in window if value == style_name) / len(window)) if window else 0.0

    def topic_repeat_ratio(self, event_id: str) -> float:
        window = self._recent(self.topic_history)
        return float(sum(1 for value in window if value == event_id) / len(window)) if window else 0.0

    def record_feed(self, feed: FeedBundle) -> None:
        self.last_feed_ids = [item.content_id for item in feed.items]
        self.exposure_history.append(1 if feed.misinfo_exposed else 0)
        if feed.misinfo_exposed and feed.items:
            focus = feed.items[0]
            self.style_history.append(focus.scenario_type)
            self.topic_history.append(focus.source_event_id)

    def perceive_feedback(self, feedback_value: float) -> None:
        self.feedback_memory.append(clip_01(feedback_value))

    def update_perceived_emotion_distribution(self, distribution: Dict[str, float]) -> None:

        normalized = self._normalize_emotion_distribution(distribution)
        if normalized is not None:
            self.perceived_emotion_distribution = normalized

    def decay_perceived_emotion_distribution(self, decay_rate: float) -> None:

        rate = clip_01(decay_rate)
        negative_labels = ("anxiety", "anger", "suspicion")
        decayed = {
            label: clip_01(self.perceived_emotion_distribution.get(label, 0.0))
            * (1.0 - rate)
            for label in negative_labels
        }
        decayed["none"] = max(0.0, 1.0 - sum(decayed.values()))
        self.perceived_emotion_distribution = (
            self._normalize_emotion_distribution(decayed)
            or {"anxiety": 0.0, "anger": 0.0, "suspicion": 0.0, "none": 1.0}
        )

    def should_interact(self, decision: ActionDecision) -> bool:
        return decision.action_type not in {"inactive", "ignore"} and decision.action_intensity > 0.0

    def _should_use_runtime_llm(self, llm_client: LLMClient, config: SimulationConfig) -> bool:
        return bool(config.runtime_llm_enabled and llm_client.has_api_key())

    def _run_llm_json(
        self,
        prompt: str,
        llm_client: LLMClient,
        temperature: float,
    ) -> Optional[Dict[str, object]]:
        """直接调用运行时 LLM，不缓存感知或决策响应。"""

        payload = llm_client.generate_json(prompt, temperature=temperature)
        if isinstance(payload, dict):
            return payload
        if payload is not None:
            llm_client.record_invalid_response()
        return None

    @staticmethod
    def _find_focus_content(feed: FeedBundle, preferred_id: Optional[str]) -> Optional[ContentItem]:
        if preferred_id:
            for item in feed.items:
                if item.content_id == preferred_id:
                    return item
        return feed.items[0] if feed.items else None

    @staticmethod
    def _dominant_label_from_distribution(distribution: Dict[str, float]) -> str:
        return max(
            ["anxiety", "anger", "suspicion", "none"],
            key=lambda label: float(distribution.get(label, 0.0)),
        )

    @staticmethod
    def _normalize_emotion_distribution(payload: object) -> Optional[Dict[str, float]]:
        if not isinstance(payload, dict):
            return None
        labels = ["anxiety", "anger", "suspicion", "none"]
        values: Dict[str, float] = {}
        try:
            for label in labels:
                if label not in payload or isinstance(payload[label], bool):
                    return None
                value = float(payload[label])
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    return None
                values[label] = value
        except (TypeError, ValueError):
            return None
        total = sum(values.values())
        if total <= 0.0 or abs(total - 1.0) > 0.02:
            return None
        return {label: clip_01(values[label] / total) for label in labels}

    def _negative_emotion_delta(
        self,
        focus: ContentItem,
        scenario: ScenarioConfig,
    ) -> float:

        return clip_01(
            0.28 * focus.arousal_score
            + 0.28 * focus.misleading_ratio
            + 0.10 * self.sensitivity
            + 0.10 * self.persona.susceptibility_bias
            + 0.24 * scenario.risk_boost
        )

    def parse_llm_perception_payload(
        self,
        payload: object,
        base_social_pressure: float,
    ) -> Optional[tuple[Dict[str, float], float]]:

        if isinstance(payload, list):
            if len(payload) != 6:
                return None
            try:
                if int(payload[0]) != self.node_id:
                    return None
            except (TypeError, ValueError):
                return None
            distribution_payload = {
                "anxiety": payload[1],
                "anger": payload[2],
                "suspicion": payload[3],
                "none": payload[4],
            }
            social_pressure_payload = payload[5]
        elif isinstance(payload, dict):
            distribution_payload = payload.get("emotion_distribution")
            social_pressure_payload = payload.get("social_pressure")
        else:
            return None

        distribution = self._normalize_emotion_distribution(distribution_payload)
        if distribution is None:
            return None
        try:
            llm_social_pressure = clip_01(float(social_pressure_payload))
        except (TypeError, ValueError):
            return None
        social_pressure = clip_01(
            0.75 * base_social_pressure + 0.25 * llm_social_pressure
        )
        return distribution, social_pressure

    def perceive_feed(
        self,
        feed: FeedBundle,
        llm_client: LLMClient,
        config: SimulationConfig,
        scenario: ScenarioConfig,
        use_llm_perception: bool = False,
        llm_payload: Optional[object] = None,
        emotion_structure_source: str = "unchanged",
    ) -> PerceptionResult:
        focus = self._find_focus_content(feed, None)
        if focus is None:
            return PerceptionResult(focus_content_id=None)
        else:
            negative_delta = self._negative_emotion_delta(focus, scenario)
            base_social_pressure = clip_01(scenario.share_boost)
            social_pressure = base_social_pressure
            llm_label_mode = use_llm_perception and self._should_use_runtime_llm(llm_client, config)
            llm_perception = None
            if llm_label_mode and llm_payload is not None:
                llm_perception = self.parse_llm_perception_payload(
                    llm_payload,
                    base_social_pressure,
                )

            if llm_perception is not None:
                emotion_distribution, social_pressure = llm_perception
                structure_source = emotion_structure_source
            else:
                emotion_distribution = dict(self.perceived_emotion_distribution)
                structure_source = "pending_or_retained"
            dominant_label = self._dominant_label_from_distribution(emotion_distribution)
            result = PerceptionResult(
                focus_content_id=focus.content_id,
                negative_emotion_delta=negative_delta,
                social_pressure=social_pressure,
                emotion_distribution=emotion_distribution,
                dominant_emotion_label=dominant_label,
                emotion_structure_source=structure_source,
            )
        return result

    def _heuristic_decision(
        self,
        feed: FeedBundle,
        perception: PerceptionResult,
        config: SimulationConfig,
        scenario: ScenarioConfig,
        rng: np.random.Generator,
    ) -> ActionDecision:
        focus = self._find_focus_content(feed, perception.focus_content_id)
        allowed_actions = set(self.persona.allowed_actions)

        if focus is None:
            return ActionDecision("ignore", None, 0.0, "无焦点内容")

        activation_score = clip_01(
            0.50 * self.emotion + 0.30 * self.interaction_tendency + 0.20 * perception.social_pressure
        )

        threshold = clip_01(self.persona.drive_threshold)
        if activation_score < threshold:
            return ActionDecision("ignore", None, 0.0, "未达到互动门槛")

        feasible_actions = [
            action
            for action in self._candidate_actions_for_focus(
                focus,
                config.misinformation_threshold,
            )
            if action != "ignore"
        ]

        if not feasible_actions:
            return ActionDecision("ignore", None, 0.0, "达到互动门槛但无合法互动行为")

        action_weights = {
            action: max(
                float(
                    config.action_weights.get(
                        action,
                        DEFAULT_ACTION_WEIGHTS[action],
                    )
                ),
                0.0,
            )
            for action in feasible_actions
        }
        weights = np.asarray([action_weights[action] for action in feasible_actions], dtype=float)

        if float(weights.sum()) <= 0.0:
            action_type = feasible_actions[0]
        else:
            probabilities = weights / weights.sum()
            action_type = str(rng.choice(np.asarray(feasible_actions, dtype=object), p=probabilities))

        intensity = self._action_intensity_from_activation(
            action_type,
            activation_score,
            config,
        )
        reason_map = {
            "like": "达到互动门槛后选择低成本反馈",
            "comment": "达到互动门槛后选择表达性互动",
            "reply": "达到互动门槛且存在可回复对象",
            "repost": "达到互动门槛且内容具有可传播误导框架",
        }
        return ActionDecision(
            action_type,
            focus.content_id,
            intensity,
            reason_map.get(action_type, "达到互动门槛后选择互动行为"),
        )

    @staticmethod
    def _action_intensity_from_activation(
        action_type: str,
        activation_score: float,
        config: SimulationConfig,
    ) -> float:
        """统一计算规则代理与LLM代理的最终行为强度。"""

        if action_type in {"inactive", "ignore"}:
            return 0.0
        default_weight = DEFAULT_ACTION_WEIGHTS.get(action_type, 0.0)
        action_weight = max(
            float(config.action_weights.get(action_type, default_weight)),
            0.0,
        )
        return clip_01(
            0.10
            + 0.90 * clip_01(activation_score) * action_weight
        )

    def _candidate_actions_for_focus(
        self,
        focus: Optional[ContentItem],
        misinformation_threshold: float,
    ) -> List[str]:
        """返回符合内容情境与用户画像权限的候选行为。"""

        if focus is None:
            return ["ignore"]

        allowed_actions = set(self.persona.allowed_actions)
        candidates = ["ignore"]
        if "like" in allowed_actions:
            candidates.append("like")
        if "comment" in allowed_actions:
            candidates.append("comment")
        if "reply" in allowed_actions and focus.content_type in {"comment", "reply", "repost"}:
            candidates.append("reply")
        if "repost" in allowed_actions and focus.is_misinformation(misinformation_threshold):
            candidates.append("repost")
        return candidates or ["ignore"]

    def parse_llm_action_payload(
        self,
        feed: FeedBundle,
        perception: PerceptionResult,
        config: SimulationConfig,
        payload: object,
    ) -> Optional[ActionDecision]:
        """校验单个 Agent 的批量行为决策结果。"""

        focus = self._find_focus_content(feed, perception.focus_content_id)
        candidates = self._candidate_actions_for_focus(
            focus,
            config.misinformation_threshold,
        )
        if focus is None:
            return ActionDecision("ignore", None, 0.0, "无焦点内容")

        if isinstance(payload, list):
            if len(payload) not in {2, 3}:
                return None
            try:
                if int(payload[0]) != self.node_id:
                    return None
                action_code_value = float(payload[1])
                if not action_code_value.is_integer():
                    return None
                action_type = ACTION_CODE_TO_TYPE.get(int(action_code_value), "")
            except (TypeError, ValueError):
                return None
        elif isinstance(payload, dict):
            # 兼容旧缓存或服务偶发返回的对象结构。
            action_type = str(payload.get("action_type", "")).strip().lower()
        else:
            return None

        if action_type not in candidates:
            return None

        activation_score = clip_01(
            0.50 * self.emotion
            + 0.30 * self.interaction_tendency
            + 0.20 * perception.social_pressure
        )
        if activation_score < clip_01(self.persona.drive_threshold):
            return ActionDecision("ignore", None, 0.0, "未达到互动门槛")

        if action_type == "ignore":
            return ActionDecision("ignore", None, 0.0, "LLM选择忽略")
        intensity = self._action_intensity_from_activation(
            action_type,
            activation_score,
            config,
        )
        return ActionDecision(
            action_type,
            focus.content_id,
            intensity,
            "LLM根据用户画像和感知状态选择行为",
        )

    @staticmethod
    def _heuristic_action_text(decision: ActionDecision, target_content: Optional[ContentItem]) -> str:
        if decision.action_type == "reply":
            return "这个说法确实让人不太放心，后续需要继续看。"
        if decision.action_type == "repost":
            return "转发给大家看看，这件事需要更多人关注。"
        if decision.action_type == "comment":
            return "这件事让人不安，需要继续关注后续变化。"
        return ""

    def generate_action_text(
        self,
        feed: FeedBundle,
        focus_content: Optional[ContentItem],
        decision: ActionDecision,
        target_content: Optional[ContentItem],
        llm_client: LLMClient,
        config: SimulationConfig,
        llm_config: LLMConfig,
        step: int,
        rng: np.random.Generator,
    ) -> str:
        text = self._heuristic_action_text(decision, target_content)
        if decision.action_type in {"ignore", "like"}:
            return text

        if target_content is not None and self._should_use_runtime_llm(llm_client, config):
            payload = self._run_llm_json(
                build_action_text_prompt(
                    self,
                    feed,
                    focus_content or target_content,
                    target_content,
                    decision,
                    step,
                ),
                llm_client,
                llm_config.content_temperature,
            )
            generated = str(payload.get("text", "")).strip() if isinstance(payload, dict) else ""
            if is_clean_action_text(generated):
                text = generated
        self.last_action_text = text
        return text

    def _update_emotion_from_perception(
        self,
        feed: FeedBundle,
        perception: PerceptionResult,
        focus_content: Optional[ContentItem],
        scenario: ScenarioConfig,
        network_feedback: float,
        config: SimulationConfig,
    ) -> None:
        is_misinfo_focus = bool(
            focus_content and focus_content.is_misinformation(config.misinformation_threshold)
        )
        material_arousal = focus_content.arousal_score if focus_content is not None else 0.25
        stimulus_value = misinformation_stimulus(
            material_arousal_score=material_arousal if is_misinfo_focus else 0.25,
            sensitivity=self.sensitivity,
            memory_boost=(
                scenario.memory_boost
                if feed.misinfo_exposed
                or (
                    focus_content is not None
                    and (
                        self.topic_repeat_ratio(focus_content.source_event_id) > 0.0
                        or self.style_repeat_ratio(focus_content.scenario_type) > 0.0
                    )
                )
                else 0.05
            ),
            avg_past_stimulus=self.avg_past_stimulus(),
        )
        stimulus_value = clip_01(0.45 * stimulus_value + 0.55 * perception.negative_emotion_delta)
        stimulus_value = clip_01(stimulus_value * self.persona.emotion_sensitivity_weight)
        feedback_value = clip_01(network_feedback)
        self.perceive_feedback(feedback_value)
        self.stimulus_memory.append(stimulus_value)
        self.last_stimulus = stimulus_value
        memory_weight = effective_memory_emotion_weight(
            base_memory_emotion_weight=config.memory_emotion_weight,
            style_repeat_ratio=feed.style_repeat_ratio,
            topic_repeat_ratio=feed.topic_repeat_ratio,
            repeat_strength=config.repeat_strength,
        )
        feedback_weight = effective_gamma_feedback_weight(
            base_gamma_feedback=config.gamma_feedback,
            visibility_signal=feed.visibility_signal,
            visibility_strength=config.visibility_strength,
        )
        self.emotion = emotion_update(
            self.emotion,
            stimulus_value,
            feedback_value,
            config.emotion_decay,
            config.beta_stimulus,
            feedback_weight,
            self.avg_past_emotion(),
            memory_weight,
        )
        self.emotion_history.append(self.emotion)
        self.last_feedback = feedback_value

    def apply_perception_update(
        self,
        feed: FeedBundle,
        perception: PerceptionResult,
        scenario: ScenarioConfig,
        network_feedback: float,
        config: SimulationConfig,
    ) -> Optional[ContentItem]:
        """应用既有情绪公式，并返回本轮焦点内容。"""

        if feed.misinfo_exposed and perception.emotion_structure_source == "llm_refresh":
            self.update_perceived_emotion_distribution(perception.emotion_distribution)
        elif not feed.misinfo_exposed:
            self.decay_perceived_emotion_distribution(config.emotion_structure_decay_rate)
        focus_content = self._find_focus_content(feed, perception.focus_content_id)
        self._update_emotion_from_perception(
            feed,
            perception,
            focus_content,
            scenario,
            network_feedback,
            config,
        )
        return focus_content

    def decide_action(
        self,
        feed: FeedBundle,
        perception: PerceptionResult,
        config: SimulationConfig,
        scenario: ScenarioConfig,
        rng: np.random.Generator,
        llm_client: LLMClient,
        use_runtime_llm: bool,
        llm_payload: Optional[object] = None,
    ) -> tuple[ActionDecision, bool]:
        """采用批量 LLM 结果决策；无效条目单独回退到既有规则。"""

        decision = None
        used_runtime_llm = False
        if use_runtime_llm:
            decision = self.parse_llm_action_payload(
                feed,
                perception,
                config,
                llm_payload,
            )
            if decision is None and llm_payload is not None:
                llm_client.record_invalid_response()
            used_runtime_llm = decision is not None
        if decision is None:
            decision = self._heuristic_decision(feed, perception, config, scenario, rng)
        return decision, used_runtime_llm

    def finalize_decision(
        self,
        feed: FeedBundle,
        focus_content: Optional[ContentItem],
        decision: ActionDecision,
        llm_client: LLMClient,
        config: SimulationConfig,
        llm_config: LLMConfig,
        step: int,
        rng: np.random.Generator,
    ) -> str:

        self.last_action_type = decision.action_type
        self.action_history.append(decision.action_type)
        self.last_interaction_strength = (
            0.0 if decision.action_type == "ignore" else clip_01(decision.action_intensity)
        )
        return self.generate_action_text(
            feed,
            focus_content,
            decision,
            focus_content,
            llm_client,
            config,
            llm_config,
            step,
            rng,
        )
