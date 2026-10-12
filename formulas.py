from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np


def clip_01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=float)))


def recommendation_signal(
    recommendation_strength: float,
    hotness_score: float,
    arousal_score: float,
    misleading_ratio: float,
) -> float:
    evidence = (
        0.40 * clip_01(hotness_score)
        + 0.30 * clip_01(arousal_score)
        + 0.30
        * clip_01(misleading_ratio)
    )
    return max(0.0, recommendation_strength) * evidence


def repeated_exposure_evidence(style_repeat_ratio: float, topic_repeat_ratio: float) -> float:
    return 0.60 * clip_01(style_repeat_ratio) + 0.40 * clip_01(topic_repeat_ratio)


def repeated_exposure_signal(style_repeat_ratio: float, topic_repeat_ratio: float, repeat_strength: float = 1.0) -> float:
    evidence = repeated_exposure_evidence(style_repeat_ratio, topic_repeat_ratio)
    return max(0.0, repeat_strength) * evidence


def effective_memory_emotion_weight(
    base_memory_emotion_weight: float,
    style_repeat_ratio: float,
    topic_repeat_ratio: float,
    repeat_strength: float,
) -> float:
    repeat_evidence = repeated_exposure_evidence(style_repeat_ratio, topic_repeat_ratio)
    repeat_modifier = 1.0 + 0.80 * (max(0.0, repeat_strength) - 0.25) * repeat_evidence
    repeat_modifier = min(2.0, max(0.5, repeat_modifier))
    return max(0.0, base_memory_emotion_weight) * repeat_modifier


def social_visibility_signal(neighbor_interaction_signal: float, visibility_strength: float = 1.0) -> float:
    return max(0.0, visibility_strength) * clip_01(neighbor_interaction_signal)


def effective_gamma_feedback_weight(
    base_gamma_feedback: float,
    visibility_signal: float,
    visibility_strength: float,
) -> float:
    visibility_evidence = clip_01(visibility_signal)
    visibility_modifier = 1.0 + 1.0 * (max(0.0, visibility_strength) - 0.35) * visibility_evidence
    visibility_modifier = min(2.0, max(0.5, visibility_modifier))
    return max(0.0, base_gamma_feedback) * visibility_modifier


def exposure_probability(
    recommendation_signal_value: float,
    repeated_exposure_signal_value: float,
    visibility_signal_value: float,
) -> float:
    signal = (
        recommendation_signal_value
        + repeated_exposure_signal_value
        + visibility_signal_value
    ) / 3.0
    return clip_01(signal)


def misinformation_stimulus(
    material_arousal_score: float,
    sensitivity: float,
    memory_boost: float,
    avg_past_stimulus: float,
) -> float:
    adjusted_arousal = clip_01(material_arousal_score)
    value = (
        adjusted_arousal * (0.5 + 0.5 * clip_01(sensitivity))
        + 0.75 * clip_01(memory_boost) * clip_01(avg_past_stimulus)
    )
    return clip_01(value)


def interaction_feedback(
    neighbor_ids: Iterable[int],
    previous_interaction_strengths: Dict[int, float],
    previous_emotions: Dict[int, float],
) -> float:
    values = [
        clip_01(previous_interaction_strengths.get(neighbor_id, 0.0)) * clip_01(previous_emotions.get(neighbor_id, 0.0))
        for neighbor_id in neighbor_ids
    ]
    return clip_01(safe_mean(values))


def emotion_update(
    current_emotion: float,
    stimulus_value: float,
    feedback_value: float,
    emotion_decay: float,
    beta_stimulus: float,
    gamma_feedback: float,
    past_emotion_value: float = 0.0,
    memory_emotion_weight: float = 0.0,
) -> float:
    value = (
        (1.0 - emotion_decay) * clip_01(current_emotion)
        + clip_01(beta_stimulus) * clip_01(stimulus_value)
        + clip_01(gamma_feedback) * clip_01(feedback_value)
        + clip_01(memory_emotion_weight) * clip_01(past_emotion_value)
    )
    return clip_01(value)


def group_mean_negative_emotion(emotions: Sequence[float]) -> float:
    return safe_mean(list(emotions))


def high_negative_ratio(emotions: Sequence[float], threshold: float) -> float:
    if not emotions:
        return 0.0
    return float(sum(1 for value in emotions if value >= threshold) / len(emotions))


def group_exposure_rate(exposure_flags: Sequence[bool]) -> float:
    if not exposure_flags:
        return 0.0
    return float(sum(1 for flag in exposure_flags if flag) / len(exposure_flags))


def group_activation_rate(active_flags: Sequence[bool]) -> float:
    if not active_flags:
        return 0.0
    return float(sum(1 for flag in active_flags if bool(flag)) / len(active_flags))


def interaction_rate(interaction_strengths: Sequence[float]) -> float:
    if not interaction_strengths:
        return 0.0
    return float(sum(1 for value in interaction_strengths if value > 0.0) / len(interaction_strengths))


def group_emotion_dispersion(emotions: Sequence[float]) -> float:
    if not emotions:
        return 0.0
    array = np.asarray(emotions, dtype=float)
    return float(np.sqrt(np.mean((array - np.mean(array)) ** 2)))


def compute_peak(series: Sequence[float]) -> float:
    if not series:
        return 0.0
    return float(np.max(np.asarray(series, dtype=float)))


def compute_mean_step_emotion_increment(series: Sequence[float]) -> float:
    """返回群体情绪在每个仿真步上的平均有符号变化。"""

    if len(series) < 2:
        return 0.0
    values = np.asarray(series, dtype=float)
    return float(np.mean(np.diff(values)))


def compute_peak_average_rise_speed(series: Sequence[float]) -> float:
    """返回群体情绪从初始值上升到轨迹峰值的平均速度。"""

    if len(series) < 2:
        return 0.0
    values = np.asarray(series, dtype=float)
    peak_index = int(np.argmax(values))
    if peak_index <= 0:
        return 0.0
    return float(max(0.0, (values[peak_index] - values[0]) / peak_index))


def compute_duration(series: Sequence[float], threshold: float) -> float:
    return float(sum(1 for value in series if value >= threshold))


def compute_amplification(series: Sequence[float]) -> float:
    if not series:
        return 0.0
    return float(max(series) - series[0])


def compute_recovery_gap(series: Sequence[float]) -> float:
    if not series:
        return 0.0
    return float(series[-1] - series[0])


def compute_high_negative_ratio(emotions: Iterable[float], threshold: float) -> float:
    return high_negative_ratio(list(emotions), threshold)


def compute_group_exposure_rate(exposure_flags: Iterable[bool]) -> float:
    return group_exposure_rate(list(exposure_flags))


def compute_group_activation_rate(active_flags: Iterable[bool]) -> float:
    return group_activation_rate(list(active_flags))


def compute_interaction_rate(interaction_strengths: Iterable[float]) -> float:
    return interaction_rate(list(interaction_strengths))


def compute_group_dispersion(emotions: Iterable[float]) -> float:
    return group_emotion_dispersion(list(emotions))


def build_indicator_system_rows() -> list[dict[str, object]]:
    return [
        {
            "indicator_code": "G_t",
            "indicator_name": "群体平均负面情绪",
            "level": "核心过程指标",
            "meaning": "刻画个体负面情绪在群体层面的平均水平。",
        },
        {
            "indicator_code": "H_t",
            "indicator_name": "高负面情绪节点占比",
            "level": "核心过程指标",
            "meaning": "刻画达到高负面情绪阈值的普通用户占比。",
        },
        {
            "indicator_code": "Peak",
            "indicator_name": "情绪峰值",
            "level": "结果指标",
            "meaning": "实验期间群体平均负面情绪达到的最高值。",
        },
        {
            "indicator_code": "Delta_G_t",
            "indicator_name": "单位时间情绪增量",
            "level": "过程指标",
            "meaning": "相邻时间步群体平均负面情绪之差 G(t)-G(t-1)，可为正或负。",
        },
        {
            "indicator_code": "PeakRiseSpeed",
            "indicator_name": "达峰平均上升速度",
            "level": "结果指标",
            "meaning": "从初始群体情绪到峰值的平均爬升速率，用于比较不同条件下的前期上升过程。",
        },
        {
            "indicator_code": "Duration",
            "indicator_name": "持续时间",
            "level": "结果指标",
            "meaning": "群体平均负面情绪达到扩散阈值的时间步数量。",
        },
    ]
