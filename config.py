from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EXPERIMENTAL_STIMULI_PATH = DATA_DIR / "experimental_stimuli.json"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
RAW_DIR = OUTPUTS_DIR / "raw"
EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"

SCENARIO_TYPES = [
    "risk_exaggeration",
    "selective_interpretation",
    "emotional_wording",
]

COMPOUND_SCENARIO_TYPE = "compound_amplification"
EXPERIMENT_1_SCENARIO_TYPES = [*SCENARIO_TYPES, COMPOUND_SCENARIO_TYPE]

DEFAULT_SCENARIO = "risk_exaggeration"


def load_environment() -> None:
    """读取项目目录下的 `.env`。"""

    if load_dotenv is not None:
        load_dotenv(BASE_DIR / ".env")


def _get_bool(name: str, default: bool) -> bool:
    """读取布尔型环境变量。"""

    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_float_range(name: str, default: Tuple[float, float]) -> Tuple[float, float]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != 2:
        return default
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError:
        return default


def _get_int_range(name: str, default: Tuple[int, int]) -> Tuple[int, int]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != 2:
        return default
    try:
        low, high = int(parts[0]), int(parts[1])
    except ValueError:
        return default
    return (min(low, high), max(low, high))


def _get_string_list(name: str) -> Tuple[str, ...]:
    raw = os.getenv(name, "")
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _get_int_list(name: str, default: Tuple[int, ...]) -> Tuple[int, ...]:
    """读取保持原有顺序的正整数列表。"""

    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError:
        return default
    if not values or any(value <= 0 for value in values) or len(set(values)) != len(values):
        return default
    return values


def _get_float_list(name: str, default: Tuple[float, ...]) -> Tuple[float, ...]:
    """Read a unique, ordered list of finite sensitivity values."""

    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError:
        return default
    if not values or len(set(values)) != len(values):
        return default
    return values


@dataclass
class LLMConfig:
    """大模型文本生成与内容评分配置。"""

    api_key: Optional[str] = None
    base_url: str = "https://www.anyapi.vip/v1"
    model: str = "gpt-5.4-mini"
    timeout: int = 60
    content_temperature: float = 0.55
    perception_temperature: float = 0.15

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """从环境变量中读取 LLM 配置。"""

        load_environment()
        return cls(
            api_key=os.getenv("LLM_API_KEY") or None,
            base_url=os.getenv("LLM_BASE_URL", "https://www.anyapi.vip/v1"),
            model=os.getenv("LLM_MODEL", "gpt-5.4-mini"),
            timeout=int(os.getenv("LLM_TIMEOUT", "60")),
            content_temperature=float(os.getenv("LLM_CONTENT_TEMPERATURE", "0.55")),
            perception_temperature=float(os.getenv("LLM_PERCEPTION_TEMPERATURE", "0.15")),
        )

    def runtime_mode_name(self) -> str:
        """返回当前运行模式说明。"""

        return "LLM文本增强" if self.api_key else "启发式文本回退"


@dataclass(frozen=True)
class LLMProfile:
    """实验六中可替换的运行时 LLM 配置。"""

    profile_id: str
    model: str
    base_url: str
    api_key: str
    timeout: int

    def to_llm_config(self, base_config: LLMConfig) -> LLMConfig:
        return LLMConfig(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            timeout=self.timeout,
            content_temperature=base_config.content_temperature,
            perception_temperature=base_config.perception_temperature,
        )


def load_llm_profiles(
    profile_ids: Tuple[str, ...],
    base_config: LLMConfig,
) -> list[LLMProfile]:
    """读取实验六的多模型接口配置。"""

    load_environment()
    if not profile_ids:
        raise ValueError("实验六未配置 MODEL_ROBUSTNESS_PROFILES。")

    profiles: list[LLMProfile] = []
    seen_ids: set[str] = set()
    for profile_id in profile_ids:
        normalized_id = profile_id.strip().lower()
        if not normalized_id or normalized_id in seen_ids:
            raise ValueError(f"实验六模型标识重复或为空：{profile_id!r}")
        seen_ids.add(normalized_id)
        prefix = f"LLM_PROFILE_{normalized_id.upper().replace('-', '_')}"
        if normalized_id == "primary":
            raise ValueError("实验六的 primary 基准由实验一结果自动读取，请勿写入 MODEL_ROBUSTNESS_PROFILES。")
        model = os.getenv(f"{prefix}_MODEL", "").strip()
        base_url = os.getenv(f"{prefix}_BASE_URL", "").strip()
        api_key = os.getenv(f"{prefix}_API_KEY", "").strip()
        timeout = int(os.getenv(f"{prefix}_TIMEOUT", str(base_config.timeout)))
        if not model or not base_url or not api_key:
            raise ValueError(
                f"实验六模型 {normalized_id} 的 MODEL、BASE_URL 或 API_KEY 未完整配置。"
            )
        profiles.append(
            LLMProfile(
                profile_id=normalized_id,
                model=model,
                base_url=base_url,
                api_key=api_key,
                timeout=max(1, timeout),
            )
        )
    return profiles


def load_enabled_model_robustness_profile_ids(
    profile_ids: Tuple[str, ...],
) -> tuple[str, ...]:
    """Return the Experiment 6 profiles explicitly enabled for this run."""

    load_environment()
    enabled: list[str] = []
    seen_ids: set[str] = set()
    for profile_id in profile_ids:
        normalized_id = profile_id.strip().lower()
        if not normalized_id or normalized_id in seen_ids:
            continue
        seen_ids.add(normalized_id)
        suffix = normalized_id.upper().replace("-", "_")
        if _get_bool(f"RUN_EXPERIMENT_6_{suffix}", False):
            enabled.append(normalized_id)
    return tuple(enabled)


@dataclass
class SimulationConfig:

    num_agents: int = 100
    num_steps: int = 25
    network_type: str = "directed_random"
    graph_edge_probability: float = 0.10
    seed_fraction: float = 0.20
    random_seed: int = 42

    recommendation_strength: float = 1.0
    repeat_strength: float = 1.0
    visibility_strength: float = 1.0

    runtime_llm_enabled: bool = True
    llm_agent_batch_size: int = 20
    llm_agent_cap_per_step: int = 10
    parallel_workers: int = 3
    debug_num_agents: int = 4
    debug_num_steps: int = 1
    debug_parallel_workers: int = 2
    progress_detail: str = "brief"
    max_content_age: int = 3
    misinformation_threshold_override_enabled: bool = False
    misinformation_threshold: float = 0.30

    emotion_decay: float = 0.24
    emotion_structure_decay_rate: float = 0.03
    beta_stimulus: float = 0.24
    gamma_feedback: float = 0.14
    memory_emotion_weight: float = 0.06
    experiment_runs_per_scenario: int = 1
    run_experiment_1: bool = True
    run_experiment_2: bool = False
    run_experiment_2_recommendation: bool = True
    run_experiment_2_repeat: bool = True
    run_experiment_2_visibility: bool = True
    run_experiment_3: bool = False
    run_experiment_4: bool = False
    run_experiment_5_agent_scale: bool = False
    experiment_5_agent_scales: Tuple[int, ...] = (200, 300, 400, 500)
    model_robustness_profiles: Tuple[str, ...] = ()
    enabled_model_robustness_profiles: Tuple[str, ...] = ()
    run_experiment_4_beta_gamma: bool = True
    run_experiment_4_memory_weight: bool = True
    run_experiment_4_emotion_decay: bool = True
    run_experiment_4_content_age: bool = True
    run_experiment_4_network_density: bool = True
    run_experiment_7_strategy_parameter_sensitivity: bool = False
    experiment_7_risk_boost_values: Tuple[float, ...] = (0.46, 0.52, 0.58, 0.64, 0.70)
    experiment_7_memory_boost_values: Tuple[float, ...] = (0.36, 0.42, 0.48, 0.54, 0.60)
    experiment_7_share_boost_values: Tuple[float, ...] = (0.42, 0.48, 0.54, 0.60, 0.66)

    initial_emotion_range: Tuple[float, float] = (0.15, 0.25)
    sensitivity_range: Tuple[float, float] = (0.35, 0.95)
    use_persona_ratio_allocation: bool = True
    initial_misinfo_exposure_ratio: float = 0.10
    initial_silent_observer_ratio: float = 0.20
    initial_emotional_sensitive_ratio: float = 0.35
    initial_low_reactivity_user_ratio: float = 0.15
    initial_high_interaction_follower_ratio: float = 0.30
    silent_observer_ratio: float = 0.60
    emotional_sensitive_ratio: float = 0.15
    low_reactivity_user_ratio: float = 0.15
    high_interaction_follower_ratio: float = 0.10
    memory_window_range: Tuple[int, int] = (3, 6)

    action_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "ignore": 0.0,
            "like": 0.20,
            "comment": 0.60,
            "reply": 0.50,
            "repost": 1.00,
        }
    )

    high_negative_threshold: float = 0.70
    medium_high_negative_threshold: float = 0.45

    experimental_stimuli_path: Path = field(default_factory=lambda: EXPERIMENTAL_STIMULI_PATH)
    output_dir: Path = field(default_factory=lambda: OUTPUTS_DIR)
    figures_dir: Path = field(default_factory=lambda: FIGURES_DIR)
    tables_dir: Path = field(default_factory=lambda: TABLES_DIR)
    raw_dir: Path = field(default_factory=lambda: RAW_DIR)
    experiments_dir: Path = field(default_factory=lambda: EXPERIMENTS_DIR)

    @classmethod
    def from_env(cls) -> "SimulationConfig":
        """从环境变量读取仿真配置。"""

        load_environment()
        return cls(
            num_agents=int(os.getenv("NUM_AGENTS", "4")),
            num_steps=int(os.getenv("NUM_STEPS", "2")),
            network_type=os.getenv("NETWORK_TYPE", "directed_random"),
            graph_edge_probability=float(os.getenv("GRAPH_EDGE_PROBABILITY", "0.10")),
            seed_fraction=float(os.getenv("SEED_FRACTION", "0.20")),
            random_seed=int(os.getenv("SIM_RANDOM_SEED", "42")),
            recommendation_strength=float(os.getenv("RECOMMENDATION_STRENGTH", "1.0")),
            repeat_strength=float(os.getenv("REPEAT_STRENGTH", "1.0")),
            visibility_strength=float(os.getenv("VISIBILITY_STRENGTH", "1.0")),
            runtime_llm_enabled=_get_bool("LLM_RUNTIME_ENABLED", True),
            llm_agent_batch_size=max(1, int(os.getenv("LLM_AGENT_BATCH_SIZE", "20"))),
            llm_agent_cap_per_step=max(
                0, int(os.getenv("LLM_AGENT_CAP_PER_STEP", "10"))
            ),
            experimental_stimuli_path=BASE_DIR
            / os.getenv("EXPERIMENTAL_STIMULI_PATH", "data/experimental_stimuli.json"),
            parallel_workers=max(1, int(os.getenv("PARALLEL_WORKERS", "3"))),
            debug_num_agents=max(1, int(os.getenv("DEBUG_NUM_AGENTS", "4"))),
            debug_num_steps=max(1, int(os.getenv("DEBUG_NUM_STEPS", "1"))),
            debug_parallel_workers=max(1, int(os.getenv("DEBUG_PARALLEL_WORKERS", "2"))),
            progress_detail=os.getenv("PROGRESS_DETAIL", "brief").strip().lower(),
            max_content_age=int(os.getenv("MAX_CONTENT_AGE", "3")),
            misinformation_threshold_override_enabled=True,
            misinformation_threshold=max(0.0, min(1.0, float(os.getenv("MISINFORMATION_THRESHOLD", "0.30")))),
            emotion_decay=float(os.getenv("EMOTION_DECAY", "0.24")),
            emotion_structure_decay_rate=max(
                0.0,
                min(1.0, float(os.getenv("EMOTION_STRUCTURE_DECAY_RATE", "0.03"))),
            ),
            beta_stimulus=float(os.getenv("BETA_STIMULUS", "0.24")),
            gamma_feedback=float(os.getenv("GAMMA_FEEDBACK", "0.14")),
            memory_emotion_weight=float(os.getenv("MEMORY_EMOTION_WEIGHT", "0.06")),
            experiment_runs_per_scenario=int(os.getenv("EXPERIMENT_RUNS_PER_SCENARIO", "1")),
            run_experiment_1=_get_bool("RUN_EXPERIMENT_1", True),
            run_experiment_2=_get_bool("RUN_EXPERIMENT_2", False),
            run_experiment_2_recommendation=_get_bool("RUN_EXPERIMENT_2_RECOMMENDATION", True),
            run_experiment_2_repeat=_get_bool("RUN_EXPERIMENT_2_REPEAT", True),
            run_experiment_2_visibility=_get_bool("RUN_EXPERIMENT_2_VISIBILITY", True),
            run_experiment_3=_get_bool("RUN_EXPERIMENT_3", False),
            run_experiment_4=_get_bool("RUN_EXPERIMENT_4", False),
            run_experiment_5_agent_scale=_get_bool("RUN_EXPERIMENT_5_AGENT_SCALE", False),
            experiment_5_agent_scales=_get_int_list(
                "EXPERIMENT_5_AGENT_SCALES", (200, 300, 400, 500)
            ),
            model_robustness_profiles=_get_string_list("MODEL_ROBUSTNESS_PROFILES"),
            enabled_model_robustness_profiles=load_enabled_model_robustness_profile_ids(
                _get_string_list("MODEL_ROBUSTNESS_PROFILES")
            ),
            run_experiment_4_beta_gamma=_get_bool("RUN_EXPERIMENT_4_BETA_GAMMA", True),
            run_experiment_4_memory_weight=_get_bool("RUN_EXPERIMENT_4_MEMORY_WEIGHT", True),
            run_experiment_4_emotion_decay=_get_bool("RUN_EXPERIMENT_4_EMOTION_DECAY", True),
            run_experiment_4_content_age=_get_bool("RUN_EXPERIMENT_4_CONTENT_AGE", True),
            run_experiment_4_network_density=_get_bool("RUN_EXPERIMENT_4_NETWORK_DENSITY", True),
            run_experiment_7_strategy_parameter_sensitivity=_get_bool(
                "RUN_EXPERIMENT_7_STRATEGY_PARAMETER_SENSITIVITY", False
            ),
            experiment_7_risk_boost_values=_get_float_list(
                "EXPERIMENT_7_RISK_BOOST_VALUES", (0.46, 0.52, 0.58, 0.64, 0.70)
            ),
            experiment_7_memory_boost_values=_get_float_list(
                "EXPERIMENT_7_MEMORY_BOOST_VALUES", (0.36, 0.42, 0.48, 0.54, 0.60)
            ),
            experiment_7_share_boost_values=_get_float_list(
                "EXPERIMENT_7_SHARE_BOOST_VALUES", (0.42, 0.48, 0.54, 0.60, 0.66)
            ),
            high_negative_threshold=float(os.getenv("HIGH_NEGATIVE_THRESHOLD", "0.60")),
            medium_high_negative_threshold=float(os.getenv("MEDIUM_HIGH_NEGATIVE_THRESHOLD", "0.45")),
            initial_emotion_range=_get_float_range("INITIAL_EMOTION_RANGE", (0.15, 0.25)),
            sensitivity_range=_get_float_range("SENSITIVITY_RANGE", (0.35, 0.95)),
            use_persona_ratio_allocation=_get_bool("USE_PERSONA_RATIO_ALLOCATION", True),
            initial_misinfo_exposure_ratio=float(os.getenv("INITIAL_MISINFO_EXPOSURE_RATIO", "0.10")),
            initial_silent_observer_ratio=float(os.getenv("INITIAL_SILENT_OBSERVER_RATIO", "0.20")),
            initial_emotional_sensitive_ratio=float(os.getenv("INITIAL_EMOTIONAL_SENSITIVE_RATIO", "0.35")),
            initial_low_reactivity_user_ratio=float(os.getenv("INITIAL_LOW_REACTIVITY_USER_RATIO", "0.15")),
            initial_high_interaction_follower_ratio=float(os.getenv("INITIAL_HIGH_INTERACTION_FOLLOWER_RATIO", "0.30")),
            silent_observer_ratio=float(os.getenv("SILENT_OBSERVER_RATIO", "0.60")),
            emotional_sensitive_ratio=float(os.getenv("EMOTIONAL_SENSITIVE_RATIO", "0.15")),
            low_reactivity_user_ratio=float(os.getenv("LOW_REACTIVITY_USER_RATIO", "0.15")),
            high_interaction_follower_ratio=float(os.getenv("HIGH_INTERACTION_FOLLOWER_RATIO", "0.10")),
            memory_window_range=_get_int_range("MEMORY_WINDOW_RANGE", (3, 6)),
            action_weights={
                "ignore": float(os.getenv("IGNORE_ACTION_WEIGHT", "0.0")),
                "like": float(os.getenv("LIKE_ACTION_WEIGHT", "0.20")),
                "comment": float(os.getenv("COMMENT_ACTION_WEIGHT", "0.60")),
                "reply": float(os.getenv("REPLY_ACTION_WEIGHT", "0.50")),
                "repost": float(os.getenv("REPOST_ACTION_WEIGHT", "1.00")),
            },
        )

    def ensure_directories(self) -> None:
        """创建运行时需要的目录。"""

        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

    def clone_with(self, **overrides: Any) -> "SimulationConfig":
        """复制一份配置并覆盖部分字段。"""

        payload = asdict(self)
        payload.update(overrides)
        return SimulationConfig(**payload)

    def to_dict(self) -> Dict[str, Any]:
        """转换为便于保存的字典。"""

        payload = asdict(self)
        for key in [
            "experimental_stimuli_path",
            "output_dir",
            "figures_dir",
            "tables_dir",
            "raw_dir",
            "experiments_dir",
        ]:
            payload[key] = str(payload[key])
        return payload
