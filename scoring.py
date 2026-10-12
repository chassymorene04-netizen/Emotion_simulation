from __future__ import annotations

import statistics
from typing import TYPE_CHECKING, Dict, Mapping, Sequence

import numpy as np

from formulas import clip_01
from prompts import (
    build_emotion_composition_scoring_prompt,
    build_semantic_deviation_scoring_prompt,
    build_strategy_scoring_prompt,
)

if TYPE_CHECKING:
    from llm_client import LLMClient


STRATEGY_SCORE_FIELDS = (
    "risk_salience_score",
    "persistent_suspicion_score",
    "interaction_mobilization_score",
)

STRATEGY_SCORE_TO_PARAMETER_FIELD: Mapping[str, str] = {
    "risk_salience_score": "risk_boost",
    "persistent_suspicion_score": "memory_boost",
    "interaction_mobilization_score": "share_boost",
}

STRATEGY_SCORING_PROMPT_VERSION = "blind_strategy_scoring_continuous_v2"
STRATEGY_PARAMETER_SCALE = 0.80
STRATEGY_PARAMETER_MAPPING_VERSION = "raw_score_batch_mean_times_0_80_v2"
DEVIATION_SCORING_PROMPT_VERSION = "blind_semantic_deviation_v16"

EMOTION_COMPOSITION_FIELDS = ("anger", "panic", "anxiety", "suspicion")
EMOTION_COMPOSITION_WEIGHTS: Mapping[str, float] = {
    "anger": 0.94,
    "panic": 0.90,
    "anxiety": 0.87,
    "suspicion": 0.61,
}

def normalize_strategy_score_payload(payload: object) -> Dict[str, float] | None:
    if not isinstance(payload, dict):
        return None
    normalized: Dict[str, float] = {}
    for field in STRATEGY_SCORE_FIELDS:
        value = payload.get(field)
        if isinstance(value, bool):
            return None
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric_value) or not 0.0 <= numeric_value <= 1.0:
            return None
        normalized[field] = numeric_value
    return normalized


def normalize_strategy_scores(raw_scores: Mapping[str, object]) -> Dict[str, float]:
    normalized = normalize_strategy_score_payload(dict(raw_scores))
    if normalized is None:
        raise ValueError("三维策略评分缺少字段或超出 0—1 范围。")
    return {field: round(float(normalized[field]), 6) for field in STRATEGY_SCORE_FIELDS}


def aggregate_strategy_parameters(
    payloads: Sequence[Mapping[str, object]],
) -> Dict[str, float]:
    if not payloads:
        raise ValueError("没有可用于标定场景参数的种子文本。")

    parameter_values: Dict[str, list[float]] = {
        parameter_field: [] for parameter_field in STRATEGY_SCORE_TO_PARAMETER_FIELD.values()
    }
    for payload in payloads:
        for score_field, parameter_field in STRATEGY_SCORE_TO_PARAMETER_FIELD.items():
            try:
                value = float(payload[score_field])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"种子文本缺少策略评分：{score_field}") from error
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"种子文本策略评分越界：{score_field}={value}")
            parameter_values[parameter_field].append(value)

    return {
        parameter_field: round(
            float(statistics.fmean(values)) * STRATEGY_PARAMETER_SCALE,
            6,
        )
        for parameter_field, values in parameter_values.items()
    }


def normalize_emotion_composition(payload: object) -> Dict[str, float] | None:
    if not isinstance(payload, dict):
        return None
    values: Dict[str, float] = {}
    for field in EMOTION_COMPOSITION_FIELDS:
        try:
            value = float(payload[field])
        except (KeyError, TypeError, ValueError):
            return None
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            return None
        values[field] = value
    raw_total = sum(values.values())
    if raw_total <= 0.0:
        return None
    return {field: values[field] / raw_total for field in EMOTION_COMPOSITION_FIELDS}


def emotion_arousal_score(composition: Mapping[str, float]) -> float:
    return float(
        sum(
            float(composition[field]) * EMOTION_COMPOSITION_WEIGHTS[field]
            for field in EMOTION_COMPOSITION_FIELDS
        )
    )


def normalize_deviation_assessment(payload: object) -> Dict[str, float] | None:
    if not isinstance(payload, dict):
        return None
    try:
        raw_ratio = float(payload["misleading_ratio"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(raw_ratio) or not 0.0 <= raw_ratio <= 1.0:
        return None
    return {"misleading_ratio": round(raw_ratio, 6)}


def score_seed_content(
    *,
    llm_client: "LLMClient",
    source_event_id: str,
    source_event_summary: str,
    title: str,
    text: str,
) -> Dict[str, object]:
    """依次完成策略、情绪构成和叙事偏离盲评。"""

    updates: Dict[str, object] = {}
    strategy_assessment = llm_client.generate_json(
        build_strategy_scoring_prompt(source_event_summary, title, text),
        temperature=0.0,
    )
    normalized_strategy_scores = normalize_strategy_score_payload(strategy_assessment)
    if normalized_strategy_scores is not None:
        updates.update(normalize_strategy_scores(normalized_strategy_scores))
        updates["strategy_scores_generated_by"] = "llm"
    else:
        llm_client.record_invalid_response()
        raise ValueError("独立策略评分器未返回有效的三维连续评分。")

    emotion_assessment = llm_client.generate_json(
        build_emotion_composition_scoring_prompt(
            source_event_summary=source_event_summary,
            title=title,
            text=text,
        ),
        temperature=0.0,
    )
    normalized_emotion = normalize_emotion_composition(emotion_assessment)
    if normalized_emotion is None:
        llm_client.record_invalid_response()
        raise ValueError("独立情绪构成评分器未返回有效的四维比例。")
    composition = normalized_emotion
    arousal_score = emotion_arousal_score(composition)
    updates.update({f"emotion_{field}": value for field, value in composition.items()})
    updates["arousal_score"] = arousal_score

    deviation_assessment = llm_client.generate_json(
        build_semantic_deviation_scoring_prompt(
            source_event_summary=source_event_summary,
            title=title,
            text=text,
        ),
        temperature=0.0,
    )
    normalized_deviation = normalize_deviation_assessment(deviation_assessment)
    if normalized_deviation is None:
        llm_client.record_invalid_response()
        raise ValueError("独立叙事偏离评分器未返回有效连续分数。")
    updates.update(normalized_deviation)
    updates["deviation_scoring_prompt_version"] = DEVIATION_SCORING_PROMPT_VERSION
    updates["scores_generated_by"] = "llm"
    return updates
