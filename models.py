from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from config import SCENARIO_TYPES


@dataclass
class ContentItem:

    content_id: str
    source_agent_id: int
    source_role: str
    content_type: str
    text: str
    scenario_type: str
    source_event_id: str
    arousal_score: float
    misleading_ratio: float
    hotness_score: float
    created_step: int
    parent_content_id: Optional[str] = None
    thread_root_content_id: Optional[str] = None
    thread_depth: int = 0
    metadata: Dict[str, object] = field(default_factory=dict)

    def is_misinformation(self, threshold: float = 0.30) -> bool:
        threshold = max(0.0, min(1.0, float(threshold)))
        return self.misleading_ratio >= threshold

    def root_id(self) -> str:
        return self.thread_root_content_id or self.content_id

    def summary_line(self) -> str:
        short_text = self.text.replace("\n", " ").strip()
        if len(short_text) > 90:
            short_text = short_text[:90] + "..."
        return (
            f"[{self.content_id}] 类型={self.content_type}; 场景={self.scenario_type}; "
            f"线程={self.root_id()}; 层级={self.thread_depth}; 误导占比={self.misleading_ratio:.2f}; "
            f"唤醒度={self.arousal_score:.2f}; 文本={short_text}"
        )


@dataclass
class FeedBundle:
    """单个 agent 在某一轮形成的单内容暴露情境。"""

    agent_id: int
    items: List[ContentItem]
    recommendation_signal: float
    repeated_signal: float
    visibility_signal: float
    misinfo_exposed: bool
    style_repeat_ratio: float = 0.0
    topic_repeat_ratio: float = 0.0
    thread_ids: List[str] = field(default_factory=list)
    root_post_item: Optional[ContentItem] = None
    interaction_context_item: Optional[ContentItem] = None


@dataclass
class PerceptionResult:
    """agent 对本轮焦点内容形成的语义感知。"""

    focus_content_id: Optional[str]
    reason: str = ""
    negative_emotion_delta: float = 0.0
    social_pressure: float = 0.0
    impression_text: str = ""
    emotion_distribution: Dict[str, float] = field(
        default_factory=lambda: {
            "anxiety": 0.0,
            "anger": 0.0,
            "suspicion": 0.0,
            "none": 1.0,
        }
    )
    dominant_emotion_label: str = "none"
    dominant_emotion_reason: str = ""
    emotion_structure_source: str = "unchanged"


@dataclass
class ActionDecision:
    """agent 决策阶段输出。"""

    action_type: str
    target_content_id: Optional[str]
    action_intensity: float
    reason: str


@dataclass
class ActionEvent:
    """记录单轮中的行为事件。"""

    actor_id: int
    actor_role: str
    action_type: str
    target_content_id: Optional[str]
    generated_content_id: Optional[str]
    action_intensity: float
    step: int


@dataclass(frozen=True)
class ScenarioConfig:
    """误导信息场景配置对象。"""

    scene_name: str
    memory_boost: float
    share_boost: float
    risk_boost: float


# 历史参数只供敏感性分析和结果核对使用，主仿真参数来自种子材料评分。
REFERENCE_SCENARIO_PARAMETER_LIBRARY: Dict[str, Dict[str, float]] = {
    "risk_exaggeration": {
        "memory_boost": 0.18,
        "share_boost": 0.08,
        "risk_boost": 0.58,
    },
    "selective_interpretation": {
        "memory_boost": 0.48,
        "share_boost": 0.05,
        "risk_boost": 0.05,
    },
    "emotional_wording": {
        "memory_boost": 0.02,
        "share_boost": 0.54,
        "risk_boost": 0.02,
    },
}


def build_scenarios() -> Dict[str, ScenarioConfig]:
    scenarios: Dict[str, ScenarioConfig] = {}
    for scenario_type in SCENARIO_TYPES:
        params = REFERENCE_SCENARIO_PARAMETER_LIBRARY[scenario_type]
        scenarios[scenario_type] = ScenarioConfig(
            scene_name=scenario_type,
            memory_boost=float(params["memory_boost"]),
            share_boost=float(params["share_boost"]),
            risk_boost=float(params["risk_boost"]),
        )
    return scenarios


def get_scenario_config(scenario_type: str) -> ScenarioConfig:
    if scenario_type not in SCENARIO_TYPES:
        raise ValueError(f"未知场景类型：{scenario_type}")
    return build_scenarios()[scenario_type]


SILENT_OBSERVER_PERSONA_KEY = "silent_observer"
EMOTIONAL_SENSITIVE_PERSONA_KEY = "emotional_sensitive"
LOW_REACTIVITY_PERSONA_KEY = "low_reactivity_user"
HIGH_INTERACTION_FOLLOWER_PERSONA_KEY = "high_interaction_follower"

NORMAL_PERSONA_KEYS = [
    SILENT_OBSERVER_PERSONA_KEY,
    EMOTIONAL_SENSITIVE_PERSONA_KEY,
    LOW_REACTIVITY_PERSONA_KEY,
    HIGH_INTERACTION_FOLLOWER_PERSONA_KEY,
]


@dataclass
class AgentPersona:
    role_family: str
    persona_key: str
    role_name: str
    description: str
    preferred_style: str
    susceptibility_bias: float
    interaction_tendency_range: tuple[float, float]
    drive_threshold: float
    allowed_actions: tuple[str, ...]
    emotion_sensitivity_weight: float

    def short_profile(self) -> str:
        return (
            f"角色={self.role_name}; 类型={self.role_family}; 偏好={self.preferred_style}; "
            f"易感偏置={self.susceptibility_bias:.2f}; 说明={self.description}"
        )


NORMAL_PERSONA_LIBRARY: Dict[str, Dict[str, object]] = {
    SILENT_OBSERVER_PERSONA_KEY: {
        "role_name": "沉默旁观型用户",
        "description": "网络中的潜水用户，不太愿意参与互动，主要浏览、忽略或点赞。",
        "preferred_style": "risk_exaggeration",
        "susceptibility_bias": 0.55,
        "interaction_tendency_range": (0.05, 0.25),
        "drive_threshold": 0.18,
        "allowed_actions": ("ignore", "like"),
        "emotion_sensitivity_weight": 0.85,
    },
    EMOTIONAL_SENSITIVE_PERSONA_KEY: {
        "role_name": "情绪敏感型用户",
        "description": "极其容易被高唤醒内容触发强烈的负面情绪反应，并容易产生表达性互动。",
        "preferred_style": "emotional_wording",
        "susceptibility_bias": 0.86,
        "interaction_tendency_range": (0.45, 0.85),
        "drive_threshold": 0.20,
        "allowed_actions": ("ignore", "like", "comment", "reply", "repost"),
        "emotion_sensitivity_weight": 1.15,
    },
    LOW_REACTIVITY_PERSONA_KEY: {
        "role_name": "谨慎理智型用户",
        "description": "不易被网络中的信息挑起愤怒、焦虑等强烈负面情绪，对关注的事件更多持怀疑态度，倾向对关注的事件和他人进行讨论。",
        "preferred_style": "selective_interpretation",
        "susceptibility_bias": 0.42,
        "interaction_tendency_range": (0.20, 0.55),
        "drive_threshold": 0.32,
        "allowed_actions": ("ignore", "like", "comment", "reply", "repost"),
        "emotion_sensitivity_weight": 0.75,
    },
    HIGH_INTERACTION_FOLLOWER_PERSONA_KEY: {
        "role_name": "高互动跟风型用户",
        "description": "希望引发关注的事件的社会关注，参与文本互动倾向较高，主要热衷于转发信息。",
        "preferred_style": "risk_exaggeration",
        "susceptibility_bias": 0.68,
        "interaction_tendency_range": (0.65, 0.90),
        "drive_threshold": 0.20,
        "allowed_actions": ("ignore", "like", "comment", "reply", "repost"),
        "emotion_sensitivity_weight": 1.00,
    },
}


def _payload_to_persona(
    payload: Dict[str, object],
    role_family: str,
    persona_key: str,
) -> AgentPersona:
    interaction_range = tuple(payload.get("interaction_tendency_range", (1.0, 1.0)))
    allowed_actions = tuple(payload.get("allowed_actions", ("comment", "reply", "repost")))
    return AgentPersona(
        role_family=role_family,
        persona_key=persona_key,
        role_name=str(payload["role_name"]),
        description=str(payload["description"]),
        preferred_style=str(payload["preferred_style"]),
        susceptibility_bias=float(payload["susceptibility_bias"]),
        interaction_tendency_range=(float(interaction_range[0]), float(interaction_range[1])),
        drive_threshold=float(payload.get("drive_threshold", 0.0)),
        allowed_actions=tuple(str(action) for action in allowed_actions),
        emotion_sensitivity_weight=float(payload.get("emotion_sensitivity_weight", 1.0)),
    )


def sample_persona(
    rng: np.random.Generator,
    role_family: str,
    scenario_type: str,
    forced_persona_key: Optional[str] = None,
) -> AgentPersona:
    del scenario_type
    persona_key = forced_persona_key or NORMAL_PERSONA_KEYS[
        int(rng.integers(0, len(NORMAL_PERSONA_KEYS)))
    ]
    if persona_key not in NORMAL_PERSONA_LIBRARY:
        raise ValueError(f"Unknown normal persona key: {persona_key}")
    return _payload_to_persona(NORMAL_PERSONA_LIBRARY[persona_key], role_family, persona_key)
