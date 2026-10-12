from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Callable, Iterable, Optional

from config import (
    EXPERIMENT_1_SCENARIO_TYPES,
    EXPERIMENTAL_STIMULI_PATH,
    LLMConfig,
    LLMProfile,
    SCENARIO_TYPES,
    SimulationConfig,
)
from scoring import aggregate_strategy_parameters
from llm_client import LLMClient
from experimental_data import load_experimental_stimuli
from platform_mechanisms import build_network
from prompts import build_model_preflight_prompt
from models import ScenarioConfig
from simulation import EmotionDiffusionSimulation


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """保存 CSV 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            file.write("")
        return

    # Run rows and aggregate rows intentionally use different fields.
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: dict[str, object]) -> None:
    """保存 JSON 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _save_csv_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a complete CSV before replacing the previous artifact."""

    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    save_csv(temporary_path, rows)
    os.replace(temporary_path, path)


def _save_json_atomic(path: Path, payload: dict[str, object]) -> None:
    """Write a complete JSON report before replacing the previous artifact."""

    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    save_json(temporary_path, payload)
    os.replace(temporary_path, path)


def _safe_mean(values: Iterable[float]) -> float:
    data = list(values)
    return float(mean(data)) if data else 0.0


def _safe_std(values: Iterable[float]) -> float:
    data = list(values)
    # Repeated simulations are samples, so use the sample standard deviation.
    return float(stdev(data)) if len(data) > 1 else 0.0


def _safe_cv(mean_value: float, std_value: float) -> float:
    return float(std_value / mean_value) if abs(mean_value) > 1e-12 else 0.0


def _as_step(value: object) -> int:
    """Read a time step consistently from in-memory values or CSV strings."""

    try:
        return int(float(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid time-step value: {value!r}") from error


def _platform_experiment_config(
    base_config: SimulationConfig,
    *,
    random_seed: int,
    overrides: Optional[dict[str, object]] = None,
) -> SimulationConfig:
    """实验二至四读取同一套本地材料，并遵循统一的运行时 LLM 开关。"""

    return base_config.clone_with(random_seed=random_seed, **(overrides or {}))


EXPERIMENT_METADATA_KEYS = [
    "scenario_type",
    "group_label",
    "analysis_part",
    "mechanism_name",
    "mechanism_level",
    "mechanism_family",
    "platform_level",
    "run_index",
    "random_seed",
    "beta_stimulus",
    "gamma_feedback",
    "recommendation_strength",
    "repeat_strength",
    "visibility_strength",
    "graph_edge_probability",
    "max_content_age",
    "memory_boost",
    "memory_window_range",
    "memory_emotion_weight",
    "emotion_decay",
]

EXP1_SUMMARY_RUN_KEYS = [
    "scenario_type",
    "run_index",
    "random_seed",
    "initial_group_negative_emotion",
    "peak",
    "post_peak_drop",
    "mean_step_emotion_increment",
    "peak_average_rise_speed",
    "final_group_negative_emotion",
    "final_high_negative_ratio",
    "runtime_llm_request_count",
    "runtime_llm_failure_count",
    "runtime_llm_model",
    "risk_boost",
    "memory_boost",
    "share_boost",
    "strategy_parameter_source",
]

EXP1_SUMMARY_AGGREGATE_KEYS = [
    "scenario_type",
    "runs",
    *[
        key
        for metric in (
            "initial_group_negative_emotion",
            "peak",
            "post_peak_drop",
            "mean_step_emotion_increment",
            "peak_average_rise_speed",
            "final_group_negative_emotion",
            "final_high_negative_ratio",
            "risk_boost",
            "memory_boost",
            "share_boost",
        )
        for key in (f"mean_{metric}", f"std_{metric}", f"cv_{metric}")
    ],
]

EXP2_RUN_KEYS = [
    "scenario_type",
    "group_label",
    "scenario_group_label",
    "mechanism_name",
    "mechanism_level",
    "run_index",
    "random_seed",
    "recommendation_strength",
    "repeat_strength",
    "visibility_strength",
    "candidate_reach_total",
    "unit_interaction_intensity",
]

EXP1_TIME_SERIES_METRICS = [
    "group_negative_emotion",
    "step_emotion_increment",
    "high_negative_ratio",
    "emotion_distribution_total",
    "perceived_anxiety_mean",
    "perceived_anger_mean",
    "perceived_suspicion_mean",
    "perceived_none_mean",
]

EXP2_TIME_SERIES_METRICS = [
    "candidate_agent_content_pairs",
    "mean_memory_emotion",
]

EXP3_TIME_SERIES_METRICS = ["group_negative_emotion"]
EXP5_TIME_SERIES_METRICS = ["group_negative_emotion", "high_negative_ratio"]
EXP6_TIME_SERIES_METRICS = ["group_negative_emotion", "high_negative_ratio"]

def _select_keys(row: dict[str, object], keys: list[str]) -> dict[str, object]:
    return {key: row[key] for key in keys if key in row}


def _with_row_type(rows: list[dict[str, object]], row_type: str) -> list[dict[str, object]]:
    return [{"row_type": row_type, **row} for row in rows]


def _combine_run_and_summary_rows(
    run_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Store raw repetitions and aggregate statistics in one explicitly typed table."""

    return [*_with_row_type(run_rows, "run"), *_with_row_type(summary_rows, "summary")]


def _series_stat_keys(metric_key: str) -> tuple[str, str]:
    """Return stable mean/std column names for a raw time-series metric."""

    if metric_key.startswith("mean_"):
        return metric_key, f"std_{metric_key[5:]}"
    return f"mean_{metric_key}", f"std_{metric_key}"


def _aggregate_selected_time_series(
    run_rows: list[dict[str, object]],
    *,
    group_key: str,
    ordered_groups: list[str],
    metric_keys: list[str],
    metadata_keys: list[str],
) -> list[dict[str, object]]:
    """Aggregate repeated runs by group and step with mean and sample SD."""

    buckets: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in run_rows:
        buckets[(str(row[group_key]), _as_step(row["step"]))].append(row)

    order_map = {group: index for index, group in enumerate(ordered_groups)}
    aggregated_rows: list[dict[str, object]] = []
    for group_label, step in sorted(buckets, key=lambda item: (order_map.get(item[0], 999), item[1])):
        rows = buckets[(group_label, step)]
        aggregated: dict[str, object] = {group_key: group_label, "step": step, "runs": len(rows)}
        for metadata_key in metadata_keys:
            if metadata_key in rows[0]:
                aggregated[metadata_key] = rows[0][metadata_key]
        for metric_key in metric_keys:
            values = [float(row[metric_key]) for row in rows if row.get(metric_key) not in ("", None)]
            if not values:
                continue
            mean_key, std_key = _series_stat_keys(metric_key)
            aggregated[mean_key] = _safe_mean(values)
            aggregated[std_key] = _safe_std(values)
        aggregated_rows.append(aggregated)
    return aggregated_rows


def _build_group_summary_rows(
    results_by_group: dict[str, list[dict[str, object]]],
    ordered_groups: list[str],
    group_key: str = "group_label",
) -> list[dict[str, object]]:
    """把多次运行结果汇总为均值表。"""

    summary_rows: list[dict[str, object]] = []
    for group_label in ordered_groups:
        rows = results_by_group.get(group_label, [])
        if not rows:
            continue
        summary_row: dict[str, object] = {group_key: group_label, "runs": len(rows)}
        summary_metrics = {
            "initial_group_negative_emotion": [float(row["initial_group_negative_emotion"]) for row in rows],
            "peak": [float(row["peak"]) for row in rows],
            "post_peak_drop": [float(row["peak"]) - float(row["final_group_negative_emotion"]) for row in rows],
            "mean_step_emotion_increment": [float(row["mean_step_emotion_increment"]) for row in rows],
            "peak_average_rise_speed": [float(row["peak_average_rise_speed"]) for row in rows],
            "final_group_negative_emotion": [float(row["final_group_negative_emotion"]) for row in rows],
            "final_high_negative_ratio": [float(row["final_high_negative_ratio"]) for row in rows],
        }
        if all(row.get("duration") not in (None, "") for row in rows):
            summary_metrics["duration"] = [float(row["duration"]) for row in rows]
        for metric_name, values in summary_metrics.items():
            summary_row[f"mean_{metric_name}"] = _safe_mean(values)
            summary_row[f"std_{metric_name}"] = _safe_std(values)
            summary_row[f"cv_{metric_name}"] = _safe_cv(
                float(summary_row[f"mean_{metric_name}"]),
                float(summary_row[f"std_{metric_name}"]),
            )
        for optional_key in [
            "scenario_type",
            "mechanism_name",
            "mechanism_level",
            "mechanism_family",
            "platform_level",
            "analysis_part",
            "beta_stimulus",
            "gamma_feedback",
            "recommendation_strength",
            "repeat_strength",
            "visibility_strength",
            "graph_edge_probability",
            "max_content_age",
            "memory_emotion_weight",
            "emotion_decay",
        ]:
            if optional_key in rows[0]:
                summary_row[optional_key] = rows[0][optional_key]
        summary_rows.append(summary_row)
    return summary_rows


def _first_step_at_or_above(rows: list[dict[str, object]], key: str, threshold: float) -> object:
    for row in sorted(rows, key=lambda item: int(item.get("step", 0))):
        if float(row.get(key, 0.0)) >= threshold:
            return int(row.get("step", 0))
    return ""


def _early_rise_speed(rows: list[dict[str, object]], key: str, early_steps: int = 5) -> float:
    ordered = sorted(rows, key=lambda item: int(item.get("step", 0)))
    if len(ordered) < 2:
        return 0.0
    end_index = min(early_steps, len(ordered) - 1)
    start_step = int(ordered[0].get("step", 0))
    end_step = int(ordered[end_index].get("step", end_index))
    step_gap = max(1, end_step - start_step)
    return float((float(ordered[end_index].get(key, 0.0)) - float(ordered[0].get(key, 0.0))) / step_gap)


def _build_exp2_run_diagnostics(
    time_rows: list[dict[str, object]],
    num_agents: int,
    metadata: dict[str, object],
) -> dict[str, object]:
    ordered = sorted(time_rows, key=lambda item: int(item.get("step", 0)))
    candidate_reach_total = sum(float(row.get("candidate_agent_content_pairs", 0.0)) for row in ordered)
    textual_interactions = sum(
        float(row.get("derived_comment_count", 0.0)) + float(row.get("derived_reply_count", 0.0))
        for row in ordered
    )
    return {
        **metadata,
        "candidate_reach_total": candidate_reach_total,
        "unit_interaction_intensity": textual_interactions / max(1, num_agents),
    }


def _build_exp2_group_summary_rows(
    results_by_group: dict[str, list[dict[str, object]]],
    ordered_groups: list[str],
) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    numeric_keys = ["candidate_reach_total", "unit_interaction_intensity"]
    metadata_keys = [
        "scenario_group_label",
        "scenario_type",
        "mechanism_name",
        "mechanism_level",
        "mechanism_family",
        "recommendation_strength",
        "repeat_strength",
        "visibility_strength",
    ]
    for group_label in ordered_groups:
        rows = results_by_group.get(group_label, [])
        if not rows:
            continue
        summary_row: dict[str, object] = {
            "group_label": rows[0].get("group_label", group_label),
            "runs": len(rows),
        }
        for key in numeric_keys:
            values = [float(row.get(key, 0.0)) for row in rows]
            summary_row[f"mean_{key}"] = _safe_mean(values)
            summary_row[f"std_{key}"] = _safe_std(values)
            summary_row[f"cv_{key}"] = _safe_cv(
                float(summary_row[f"mean_{key}"]),
                float(summary_row[f"std_{key}"]),
            )
        for key in metadata_keys:
            if key in rows[0]:
                summary_row[key] = rows[0][key]
        summary_rows.append(summary_row)
    return summary_rows


SCENARIO_DISPLAY_NAMES = {
    "risk_exaggeration": "风险夸大",
    "selective_interpretation": "选择性解释",
    "emotional_wording": "情绪化措辞",
    "compound_amplification": "复合强化",
}

ACTION_DISPLAY_NAMES = {
    "comment": "评论",
    "reply": "回复",
    "repost": "转发",
}

def _short_text(value: object, limit: int = 120) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return text[:limit] + "..." if len(text) > limit else text


def _build_agent_interaction_cases_markdown(rows: list[dict[str, object]], max_items: int = 6) -> str:
    lines = [
        "# Agent 互动行为文本案例",
        "",
        "该文件从仿真过程中的文本性互动行为中抽取典型传播链。机制说明只依据本次抽样实际出现的行为，不推断未发生的评论、回复或转发。该输出不参与数值指标计算。",
        "",
    ]
    for scenario_type in EXPERIMENT_1_SCENARIO_TYPES:
        scenario_rows = [row for row in rows if str(row.get("scenario_type")) == scenario_type]
        if not scenario_rows:
            continue
        first_run = min(int(row.get("run_index", 0)) for row in scenario_rows)
        scenario_rows = [row for row in scenario_rows if int(row.get("run_index", 0)) == first_run]
        initial_rows = [row for row in scenario_rows if row.get("record_type") == "initial_misinformation"]
        interaction_rows = [
            row
            for row in scenario_rows
            if row.get("record_type") == "interaction" and str(row.get("generated_text", "")).strip()
        ]
        interaction_rows.sort(key=lambda row: (int(row.get("step", 0)), int(row.get("agent_id", -1))))
        selected_rows: list[dict[str, object]] = []
        later_rows = [row for row in interaction_rows if int(row.get("step", 0)) > 0]
        if later_rows:
            selected_rows.append(later_rows[0])
        for action_type in ("comment", "reply", "repost"):
            candidates = [row for row in interaction_rows if row.get("action_type") == action_type]
            if candidates:
                candidate = next(
                    (row for row in candidates if int(row.get("step", 0)) > 0),
                    candidates[0],
                )
                if candidate not in selected_rows:
                    selected_rows.append(candidate)
        for row in interaction_rows:
            if len(selected_rows) >= max_items:
                break
            if row not in selected_rows:
                selected_rows.append(row)
        selected_rows.sort(key=lambda row: (int(row.get("step", 0)), int(row.get("agent_id", -1))))

        lines.append(f"## 【场景】{SCENARIO_DISPLAY_NAMES.get(scenario_type, scenario_type)}")
        lines.append("")
        lines.append("**【初始误导内容】**")
        lines.append("")
        if initial_rows:
            lines.append(f"> {_short_text(initial_rows[0].get('content_text'), 180)}")
        else:
            lines.append("> 本次抽样运行中未记录初始误导内容。")
        lines.append("")
        lines.append("**【传播链】**")
        lines.append("")
        if interaction_rows:
            for row in selected_rows:
                step = int(row.get("step", 0))
                agent_id = int(row.get("agent_id", -1))
                action = ACTION_DISPLAY_NAMES.get(str(row.get("action_type")), str(row.get("action_type")))
                emotion_after = float(row.get("emotion_after", 0.0))
                lines.append(
                    f"- Step {step} Agent_{agent_id} {action}："
                    f"{_short_text(row.get('generated_text'), 160)}（情绪={emotion_after:.2f}）"
                )
        else:
            lines.append("本次抽样运行中未形成可展示的文本性互动链。")
        lines.append("")
        lines.append("**【对应机制解释】**")
        lines.append("")
        sampled_action_counts = {
            action_type: sum(1 for row in interaction_rows if str(row.get("action_type")) == action_type)
            for action_type in ("comment", "reply", "repost")
        }
        observed_actions = [
            f"{ACTION_DISPLAY_NAMES[action_type]}={count}"
            for action_type, count in sampled_action_counts.items()
            if count > 0
        ]
        if observed_actions:
            lines.append(
                "本次抽样实际记录的文本行为为：" + "，".join(observed_actions) +
                "。未出现的行为不纳入该段机制解释。"
            )
        else:
            lines.append("本次抽样没有记录到可用于解释传播链的文本行为。")
        lines.append("")
        del lines[-4:]
    return "\n".join(lines).strip() + "\n"


def _build_agent_state_trace_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Agent 状态变化报告",
        "",
        "该报告从每个场景的首个运行中挑选实际进入激活流程的 3-5 个用户，记录其从首次激活到仿真结束的逐步状态变化。",
        "该输出只展示真实运行结果，不参与数值指标计算。",
        "",
    ]
    for scenario_type in EXPERIMENT_1_SCENARIO_TYPES:
        scenario_rows = [row for row in rows if str(row.get("scenario_type")) == scenario_type]
        if not scenario_rows:
            continue
        lines.append(f"## 场景：{SCENARIO_DISPLAY_NAMES.get(scenario_type, scenario_type)}")
        lines.append("")
        agent_ids = sorted({int(row.get("agent_id", -1)) for row in scenario_rows})
        for agent_id in agent_ids:
            agent_rows = [row for row in scenario_rows if int(row.get("agent_id", -1)) == agent_id]
            agent_rows.sort(key=lambda row: int(row.get("step", 0)))
            first_row = agent_rows[0]
            lines.append(
                f"### Agent_{agent_id} | {first_row.get('role_name', '')} | {first_row.get('persona_key', '')}"
            )
            lines.append("")
            for row in agent_rows:
                step = int(row.get("step", 0))
                active = "yes" if bool(row.get("active")) else "no"
                exposed = "yes" if bool(row.get("misinfo_exposed")) else "no"
                emotion_mix = (
                    f"anxiety={float(row.get('emotion_anxiety', 0.0)):.2f}, "
                    f"anger={float(row.get('emotion_anger', 0.0)):.2f}, "
                    f"suspicion={float(row.get('emotion_suspicion', 0.0)):.2f}, "
                    f"none={float(row.get('emotion_none', 0.0)):.2f}"
                )
                lines.append(
                    f"- Step {step} | active={active} | exposed={exposed} | "
                    f"emotion={float(row.get('emotion_before', 0.0)):.3f}->{float(row.get('emotion_after', 0.0)):.3f} | "
                    f"emotion_mix=({emotion_mix}) | action={row.get('action_type', 'inactive')} | "
                    f"strength={float(row.get('action_strength', 0.0)):.3f} | "
                    f"stimulus={float(row.get('stimulus', 0.0)):.3f} | feedback={float(row.get('feedback', 0.0)):.3f}"
                )
            lines.append("")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _build_experiment_progress_callback(
    progress_callback: Callable[[str], None],
    scenario_type: str,
    run_index: int,
    experiment_name: str = "实验一",
    group_label: Optional[str] = None,
) -> Callable[[str, dict[str, object]], None]:
    """构造单次实验运行的进度回调。

    使用显式闭包，避免 IDE 对循环变量捕获产生误报。
    """

    def _callback(event: str, payload: dict[str, object]) -> None:
        progress_callback(
            _format_simulation_progress(
                prefix=f"[{experiment_name}][{group_label or scenario_type}][run {run_index + 1}]",
                event=event,
                payload=payload,
            )
        )

    return _callback


def run_experiment_1_expression_styles(
    base_config: SimulationConfig,
    llm_config: LLMConfig,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict[str, object]:
    """实验一：固定平台机制，比较三类单一策略与复合强化策略。"""

    experiment_dir = base_config.experiments_dir / "exp1_expression_styles"
    run_table_rows: list[dict[str, object]] = []
    grouped_time_series_rows: list[dict[str, object]] = []
    interaction_case_rows: list[dict[str, object]] = []
    runtime_diagnostic_rows: list[dict[str, object]] = []
    results_by_scenario: dict[str, list[dict[str, object]]] = defaultdict(list)

    for scenario_index, scenario_type in enumerate(EXPERIMENT_1_SCENARIO_TYPES):
        for run_index in range(base_config.experiment_runs_per_scenario):
            run_seed = int(base_config.random_seed + run_index)
            run_config = base_config.clone_with(random_seed=run_seed)
            if progress_callback is not None and base_config.progress_detail != "silent":
                progress_callback(
                    f"[实验一] {scenario_type} run {run_index + 1}/"
                    f"{base_config.experiment_runs_per_scenario} 开始 | seed={run_seed}"
                )

            show_step_progress = progress_callback is not None and base_config.progress_detail == "detailed"

            simulation = EmotionDiffusionSimulation(
                config=run_config,
                scenario_type=scenario_type,
                llm_config=llm_config,
                progress_callback=(
                    None
                    if not show_step_progress
                    else _build_experiment_progress_callback(
                        progress_callback=progress_callback,
                        scenario_type=scenario_type,
                        run_index=run_index,
                    )
                ),
            )
            result = simulation.run()
            if run_index == 0:
                for case_row in result.interaction_cases:
                    interaction_case_rows.append(
                        {
                            "scenario_type": scenario_type,
                            "run_index": run_index,
                            "random_seed": run_seed,
                            **case_row,
                        }
                    )
            if progress_callback is not None and base_config.progress_detail != "silent":
                progress_callback(
                    f"[实验一] {scenario_type} run {run_index + 1}/"
                    f"{base_config.experiment_runs_per_scenario} 完成 | "
                    f"peak={result.peak:.4f} | final_G={result.group_negative_emotion[-1]:.4f} | "
                    f"final_H={result.high_negative_ratio[-1]:.4f}"
                )

            summary = result.summary_dict()
            summary["run_index"] = run_index
            summary["random_seed"] = run_seed
            summary["recommendation_strength"] = run_config.recommendation_strength
            summary["repeat_strength"] = run_config.repeat_strength
            summary["visibility_strength"] = run_config.visibility_strength
            summary["memory_emotion_weight"] = run_config.memory_emotion_weight
            summary["memory_window_range"] = f"{run_config.memory_window_range[0]}-{run_config.memory_window_range[1]}"
            summary["post_peak_drop"] = max(
                0.0,
                float(summary["peak"]) - float(summary["final_group_negative_emotion"]),
            )
            run_table_rows.append(summary)
            results_by_scenario[scenario_type].append(summary)
            runtime_diagnostic_rows.append(
                {
                    "scenario_type": scenario_type,
                    "run_index": run_index,
                    "random_seed": run_seed,
                    "request_count": result.runtime_llm_request_count,
                    "failure_count": result.runtime_llm_failure_count,
                    "inference_item_count": result.runtime_llm_inference_item_count,
                    "fallback_item_count": result.runtime_llm_fallback_item_count,
                    "input_token_count": result.runtime_llm_input_token_count,
                    "output_token_count": result.runtime_llm_output_token_count,
                    "estimated_cost_rmb": result.runtime_llm_estimated_cost_rmb,
                    **result.runtime_llm_diagnostics,
                }
            )

            for row in result.time_series:
                grouped_time_series_rows.append(
                    {
                        "scenario_type": scenario_type,
                        "run_index": run_index,
                        "random_seed": run_seed,
                        **row,
                    }
                )

    summary_rows: list[dict[str, object]] = []
    for scenario_type in EXPERIMENT_1_SCENARIO_TYPES:
        rows = results_by_scenario[scenario_type]
        mean_initial = _safe_mean(float(row["initial_group_negative_emotion"]) for row in rows)
        std_initial = _safe_std(float(row["initial_group_negative_emotion"]) for row in rows)
        mean_peak = _safe_mean(float(row["peak"]) for row in rows)
        std_peak = _safe_std(float(row["peak"]) for row in rows)
        mean_post_peak_drop = _safe_mean(float(row["post_peak_drop"]) for row in rows)
        std_post_peak_drop = _safe_std(float(row["post_peak_drop"]) for row in rows)
        mean_step_increment = _safe_mean(float(row["mean_step_emotion_increment"]) for row in rows)
        std_step_increment = _safe_std(float(row["mean_step_emotion_increment"]) for row in rows)
        mean_peak_rise_speed = _safe_mean(float(row["peak_average_rise_speed"]) for row in rows)
        std_peak_rise_speed = _safe_std(float(row["peak_average_rise_speed"]) for row in rows)
        mean_duration = _safe_mean(float(row["duration"]) for row in rows)
        std_duration = _safe_std(float(row["duration"]) for row in rows)
        mean_final_g = _safe_mean(float(row["final_group_negative_emotion"]) for row in rows)
        std_final_g = _safe_std(float(row["final_group_negative_emotion"]) for row in rows)
        mean_final_h = _safe_mean(float(row["final_high_negative_ratio"]) for row in rows)
        std_final_h = _safe_std(float(row["final_high_negative_ratio"]) for row in rows)
        mean_risk_boost = _safe_mean(float(row["risk_boost"]) for row in rows)
        std_risk_boost = _safe_std(float(row["risk_boost"]) for row in rows)
        mean_memory_boost = _safe_mean(float(row["memory_boost"]) for row in rows)
        std_memory_boost = _safe_std(float(row["memory_boost"]) for row in rows)
        mean_share_boost = _safe_mean(float(row["share_boost"]) for row in rows)
        std_share_boost = _safe_std(float(row["share_boost"]) for row in rows)
        summary_rows.append(
            {
                "scenario_type": scenario_type,
                "runs": len(rows),
                "mean_initial_group_negative_emotion": mean_initial,
                "std_initial_group_negative_emotion": std_initial,
                "cv_initial_group_negative_emotion": _safe_cv(mean_initial, std_initial),
                "mean_peak": mean_peak,
                "std_peak": std_peak,
                "cv_peak": _safe_cv(mean_peak, std_peak),
                "mean_post_peak_drop": mean_post_peak_drop,
                "std_post_peak_drop": std_post_peak_drop,
                "cv_post_peak_drop": _safe_cv(mean_post_peak_drop, std_post_peak_drop),
                "mean_mean_step_emotion_increment": mean_step_increment,
                "std_mean_step_emotion_increment": std_step_increment,
                "cv_mean_step_emotion_increment": _safe_cv(mean_step_increment, std_step_increment),
                "mean_peak_average_rise_speed": mean_peak_rise_speed,
                "std_peak_average_rise_speed": std_peak_rise_speed,
                "cv_peak_average_rise_speed": _safe_cv(mean_peak_rise_speed, std_peak_rise_speed),
                "mean_duration": mean_duration,
                "std_duration": std_duration,
                "cv_duration": _safe_cv(mean_duration, std_duration),
                "mean_final_group_negative_emotion": mean_final_g,
                "std_final_group_negative_emotion": std_final_g,
                "cv_final_group_negative_emotion": _safe_cv(mean_final_g, std_final_g),
                "mean_final_high_negative_ratio": mean_final_h,
                "std_final_high_negative_ratio": std_final_h,
                "cv_final_high_negative_ratio": _safe_cv(mean_final_h, std_final_h),
                "mean_risk_boost": mean_risk_boost,
                "std_risk_boost": std_risk_boost,
                "cv_risk_boost": _safe_cv(mean_risk_boost, std_risk_boost),
                "mean_memory_boost": mean_memory_boost,
                "std_memory_boost": std_memory_boost,
                "cv_memory_boost": _safe_cv(mean_memory_boost, std_memory_boost),
                "mean_share_boost": mean_share_boost,
                "std_share_boost": std_share_boost,
                "cv_share_boost": _safe_cv(mean_share_boost, std_share_boost),
            }
        )

    aggregated_time_series = _aggregate_selected_time_series(
        grouped_time_series_rows,
        group_key="scenario_type",
        ordered_groups=list(EXPERIMENT_1_SCENARIO_TYPES),
        metric_keys=EXP1_TIME_SERIES_METRICS,
        metadata_keys=[],
    )

    summary_path = experiment_dir / "exp1_summary.csv"
    time_series_path = experiment_dir / "exp1_time_series.csv"
    interaction_cases_path = experiment_dir / "exp1_agent_interaction_cases.md"
    report_path = experiment_dir / "exp1_report.json"

    summary_table_rows = _combine_run_and_summary_rows(
        [_select_keys(row, EXP1_SUMMARY_RUN_KEYS) for row in run_table_rows],
        [_select_keys(row, EXP1_SUMMARY_AGGREGATE_KEYS) for row in summary_rows],
    )
    time_series_table_rows = _combine_run_and_summary_rows(
        [
            _select_keys(
                row,
                ["scenario_type", "run_index", "random_seed", "step", *EXP1_TIME_SERIES_METRICS],
            )
            for row in grouped_time_series_rows
        ],
        aggregated_time_series,
    )
    _save_csv_atomic(summary_path, summary_table_rows)
    _save_csv_atomic(time_series_path, time_series_table_rows)
    save_text(interaction_cases_path, _build_agent_interaction_cases_markdown(interaction_case_rows))
    _save_json_atomic(
        report_path,
        {
            "experiment_name": "实验一：不同误导表达方式下的群体负面情绪演化",
            "design": {
                "fixed_platform_mechanisms": {
                    "recommendation_strength": base_config.recommendation_strength,
                    "repeat_strength": base_config.repeat_strength,
                    "visibility_strength": base_config.visibility_strength,
                },
                "compared_scenarios": list(EXPERIMENT_1_SCENARIO_TYPES),
                "runs_per_scenario": base_config.experiment_runs_per_scenario,
                "num_agents": base_config.num_agents,
                "num_steps": base_config.num_steps,
                "memory_emotion_weight": base_config.memory_emotion_weight,
                "high_negative_threshold": base_config.high_negative_threshold,
                "medium_high_negative_threshold": base_config.medium_high_negative_threshold,
                "seed_fraction": base_config.seed_fraction,
                "graph_edge_probability": base_config.graph_edge_probability,
                "runtime_llm_enabled": base_config.runtime_llm_enabled,
                "runtime_llm_model": llm_config.model,
                "runtime_llm_base_url": llm_config.base_url,
                "llm_agent_batch_size": base_config.llm_agent_batch_size,
            },
            "runtime_llm_diagnostics": runtime_diagnostic_rows,
            "artifacts": {
                "summary_csv": str(summary_path),
                "time_series_csv": str(time_series_path),
                "agent_interaction_cases_md": str(interaction_cases_path),
            },
            "group_summary": summary_rows,
        },
    )

    return {
        "experiment_dir": str(experiment_dir),
        "summary_path": str(summary_path),
        "time_series_path": str(time_series_path),
        "interaction_cases_path": str(interaction_cases_path),
        "report_path": str(report_path),
    }


def _expected_mean_degree(num_agents: int, edge_probability: float) -> float:
    return max(0.0, float(max(0, num_agents - 1)) * float(edge_probability))


def _scaled_edge_probability(base_config: SimulationConfig, num_agents: int) -> float:
    """保持随机有向网络的期望入度和出度与主实验一致。"""

    if num_agents <= 1:
        return 0.0
    baseline_degree = _expected_mean_degree(
        base_config.num_agents,
        base_config.graph_edge_probability,
    )
    return min(1.0, baseline_degree / float(num_agents - 1))


def _network_scale_metadata(
    num_agents: int,
    edge_probability: float,
    random_seed: int,
    seed_fraction: float,
) -> dict[str, object]:
    graph = build_network("directed_random", num_agents, edge_probability, random_seed)
    return {
        "num_agents": num_agents,
        "graph_edge_probability": edge_probability,
        "expected_mean_degree": _expected_mean_degree(num_agents, edge_probability),
        "realized_mean_out_degree": graph.number_of_edges() / float(max(1, num_agents)),
        "seed_content_count": max(1, int(round(num_agents * seed_fraction))),
    }


def run_experiment_5_agent_scale_robustness(
    base_config: SimulationConfig,
    llm_config: LLMConfig,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict[str, object]:
    """实验五：在保持期望网络度数下检验不同群体规模的结果稳健性。"""

    experiment_dir = base_config.experiments_dir / "exp5_agent_scale_robustness"
    reference_size = base_config.num_agents
    if reference_size != 300:
        raise RuntimeError(
            "实验五固定以300人为网络规模参照。请将 NUM_AGENTS 设为300后运行。"
        )
    agent_scales = base_config.experiment_5_agent_scales
    if reference_size not in agent_scales:
        raise RuntimeError(
            "实验五规模列表必须包含300人参照组。请在 EXPERIMENT_5_AGENT_SCALES 中保留300。"
        )

    scale_probabilities = {
        num_agents: _scaled_edge_probability(base_config, num_agents)
        for num_agents in agent_scales
    }
    ordered_groups = [
        f"{scenario_type}__n{num_agents}"
        for scenario_type in SCENARIO_TYPES
        for num_agents in agent_scales
    ]
    run_rows: list[dict[str, object]] = []
    time_rows: list[dict[str, object]] = []
    results_by_group: dict[str, list[dict[str, object]]] = defaultdict(list)

    for scenario_type in SCENARIO_TYPES:
        for num_agents in agent_scales:
            group_label = f"{scenario_type}__n{num_agents}"
            edge_probability = scale_probabilities[num_agents]
            for run_index in range(base_config.experiment_runs_per_scenario):
                run_seed = int(base_config.random_seed + run_index)
                network_metadata = _network_scale_metadata(
                    num_agents,
                    edge_probability,
                    run_seed,
                    base_config.seed_fraction,
                )
                run_config = base_config.clone_with(
                    num_agents=num_agents,
                    graph_edge_probability=edge_probability,
                    random_seed=run_seed,
                )
                if progress_callback is not None and base_config.progress_detail != "silent":
                    progress_callback(
                        f"[实验五] {scenario_type} | {num_agents}人 | run {run_index + 1}/"
                        f"{base_config.experiment_runs_per_scenario} 开始 | seed={run_seed}"
                    )
                simulation = EmotionDiffusionSimulation(
                    config=run_config,
                    scenario_type=scenario_type,
                    llm_config=llm_config,
                    progress_callback=(
                        None
                        if progress_callback is None or base_config.progress_detail != "detailed"
                        else _build_experiment_progress_callback(
                            progress_callback=progress_callback,
                            scenario_type=scenario_type,
                            run_index=run_index,
                            experiment_name="实验五",
                            group_label=f"{num_agents}人",
                        )
                    ),
                )
                result = simulation.run()
                if progress_callback is not None and base_config.progress_detail != "silent":
                    progress_callback(
                        f"[实验五] {scenario_type} | {num_agents}人 | run {run_index + 1}/"
                        f"{base_config.experiment_runs_per_scenario} 完成 | "
                        f"peak={result.peak:.4f} | final_G={result.group_negative_emotion[-1]:.4f}"
                    )
                summary = result.summary_dict()
                source = "实验五新增运行"
                matching_time_rows = [dict(row) for row in result.time_series]

                summary.update(
                    {
                        "group_label": group_label,
                        "scenario_type": scenario_type,
                        "run_index": run_index,
                        "random_seed": run_seed,
                        "baseline_source": source,
                        **network_metadata,
                    }
                )
                summary["post_peak_drop"] = max(
                    0.0,
                    float(summary["peak"]) - float(summary["final_group_negative_emotion"]),
                )
                run_rows.append(summary)
                results_by_group[group_label].append(summary)

                for time_row in matching_time_rows:
                    # CSV 读取时 step 可能表现为 "0.0"，聚合器要求整数时间步。
                    time_row["step"] = int(float(time_row.get("step", 0)))
                    time_rows.append(
                        {
                            "group_label": group_label,
                            "scenario_type": scenario_type,
                            "run_index": run_index,
                            "random_seed": run_seed,
                            "baseline_source": source,
                            **network_metadata,
                            **time_row,
                        }
                    )

    summary_rows = _build_group_summary_rows(results_by_group, ordered_groups)
    metadata_by_group = {
        row["group_label"]: row
        for row in run_rows
        if int(row["run_index"]) == 0
    }
    for summary_row in summary_rows:
        metadata = metadata_by_group[str(summary_row["group_label"])]
        summary_row.update(
            {
                key: metadata[key]
                for key in (
                    "scenario_type",
                    "num_agents",
                    "graph_edge_probability",
                    "expected_mean_degree",
                    "realized_mean_out_degree",
                    "seed_content_count",
                )
            }
        )

    aggregated_time_series = _aggregate_selected_time_series(
        time_rows,
        group_key="group_label",
        ordered_groups=ordered_groups,
        metric_keys=EXP5_TIME_SERIES_METRICS,
        metadata_keys=["scenario_type", "num_agents", "graph_edge_probability"],
    )

    summary_path = experiment_dir / "exp5_summary.csv"
    time_series_path = experiment_dir / "exp5_time_series.csv"
    report_path = experiment_dir / "exp5_report.json"
    run_keys = [
        "group_label", "scenario_type", "num_agents", "run_index", "random_seed",
        "baseline_source", "graph_edge_probability", "expected_mean_degree",
        "realized_mean_out_degree", "seed_content_count", "peak", "post_peak_drop",
        "final_group_negative_emotion", "final_high_negative_ratio",
    ]
    summary_keys = [
        "group_label", "scenario_type", "num_agents", "graph_edge_probability",
        "expected_mean_degree", "realized_mean_out_degree", "seed_content_count", "runs",
        *[
            key
            for metric in ("peak", "post_peak_drop", "final_group_negative_emotion", "final_high_negative_ratio")
            for key in (f"mean_{metric}", f"std_{metric}", f"cv_{metric}")
        ],
    ]
    _save_csv_atomic(
        summary_path,
        _combine_run_and_summary_rows(
            [_select_keys(row, run_keys) for row in run_rows],
            [_select_keys(row, summary_keys) for row in summary_rows],
        ),
    )
    _save_csv_atomic(
        time_series_path,
        _combine_run_and_summary_rows(
            [
                _select_keys(
                    row,
                    [
                        "group_label", "scenario_type", "num_agents", "run_index", "random_seed",
                        "step", *EXP5_TIME_SERIES_METRICS,
                    ],
                )
                for row in time_rows
            ],
            aggregated_time_series,
        ),
    )
    _save_json_atomic(
        report_path,
        {
            "experiment_name": "实验五：群体规模稳健性检验",
            "design": {
                "agent_scales": list(agent_scales),
                "reference_size": reference_size,
                "reference_source": "实验五独立运行",
                "network_control": "固定期望入度与出度",
                "reference_expected_mean_degree": _expected_mean_degree(
                    base_config.num_agents,
                    base_config.graph_edge_probability,
                ),
                "edge_probabilities": scale_probabilities,
                "seed_fraction": base_config.seed_fraction,
                "runs_per_condition": base_config.experiment_runs_per_scenario,
                "num_steps": base_config.num_steps,
                "runtime_llm_enabled": base_config.runtime_llm_enabled,
                "llm_agent_batch_size": base_config.llm_agent_batch_size,
            },
            "artifacts": {
                "summary_csv": str(summary_path),
                "time_series_csv": str(time_series_path),
            },
            "group_summary": summary_rows,
        },
    )
    return {
        "experiment_dir": str(experiment_dir),
        "summary_path": str(summary_path),
        "time_series_path": str(time_series_path),
        "report_path": str(report_path),
    }


def collect_experiment_6_completed_profiles(
    base_config: SimulationConfig,
) -> dict[str, object]:
    """Rebuild combined Experiment 6 artifacts from completed profile directories."""

    baseline = _load_experiment_1_llm_baseline(base_config)
    experiment_dir = base_config.experiments_dir / "exp6_model_robustness"
    run_rows: list[dict[str, object]] = []
    time_rows: list[dict[str, object]] = []
    results_by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    ordered_groups = [f"primary__{scenario_type}" for scenario_type in SCENARIO_TYPES]

    primary_model = str(baseline["model"])
    primary_base_url = str(baseline["base_url"])
    for scenario_type, rows in dict(baseline["runs_by_scenario"]).items():
        group_label = f"primary__{scenario_type}"
        for source_row in rows:
            summary = dict(source_row)
            summary.update(
                {
                    "group_label": group_label,
                    "profile_id": "primary",
                    "model": primary_model,
                    "base_url": primary_base_url,
                    "scenario_type": scenario_type,
                    "run_index": int(float(source_row["run_index"])),
                    "random_seed": int(float(source_row["random_seed"])),
                    "post_peak_drop": float(source_row.get("post_peak_drop", 0.0) or 0.0),
                }
            )
            run_rows.append(summary)
            results_by_group[group_label].append(summary)
        for source_row in baseline["time_rows"]:
            if source_row.get("scenario_type") != scenario_type:
                continue
            time_rows.append(
                {
                    "group_label": group_label,
                    "profile_id": "primary",
                    "model": primary_model,
                    "scenario_type": scenario_type,
                    "run_index": int(float(source_row["run_index"])),
                    "random_seed": int(float(source_row["random_seed"])),
                    **{
                        key: value
                        for key, value in source_row.items()
                        if key not in {"row_type", "scenario_type", "run_index", "random_seed"}
                    },
                }
            )

    completed_profiles: list[dict[str, object]] = []
    if experiment_dir.exists():
        for profile_dir in sorted(path for path in experiment_dir.iterdir() if path.is_dir()):
            summary_path = profile_dir / "exp6_summary.csv"
            time_series_path = profile_dir / "exp6_time_series.csv"
            if not summary_path.exists() or not time_series_path.exists():
                continue

            all_profile_run_rows = [
                row
                for row in _read_csv_rows(summary_path)
                if row.get("row_type") == "run"
            ]
            if not all_profile_run_rows:
                continue
            profile_id = str(all_profile_run_rows[0].get("profile_id", "")).strip()
            if not profile_id or profile_id != profile_dir.name:
                continue
            profile_run_rows = [
                row for row in all_profile_run_rows if row.get("profile_id") == profile_id
            ]
            profile_time_rows = [
                row
                for row in _read_csv_rows(time_series_path)
                if row.get("row_type") == "run" and row.get("profile_id") == profile_id
            ]
            expected_runs = int(baseline["runs_per_scenario"])
            if any(
                sum(1 for row in profile_run_rows if row.get("scenario_type") == scenario_type) != expected_runs
                for scenario_type in SCENARIO_TYPES
            ):
                continue

            profile_metadata = profile_run_rows[0]
            completed_profiles.append(
                {
                    "profile_id": profile_id,
                    "model": str(profile_metadata.get("model", "")),
                    "base_url": str(profile_metadata.get("base_url", "")),
                    "summary_path": str(summary_path),
                }
            )
            for scenario_type in SCENARIO_TYPES:
                group_label = f"{profile_id}__{scenario_type}"
                ordered_groups.append(group_label)
                scenario_rows = [
                    dict(row)
                    for row in profile_run_rows
                    if row.get("scenario_type") == scenario_type and row.get("group_label") == group_label
                ]
                run_rows.extend(scenario_rows)
                results_by_group[group_label].extend(scenario_rows)
                time_rows.extend(
                    dict(row)
                    for row in profile_time_rows
                    if row.get("scenario_type") == scenario_type and row.get("group_label") == group_label
                )

    summary_rows = _build_group_summary_rows(results_by_group, ordered_groups)
    metadata_by_group = {
        str(row["group_label"]): row
        for row in run_rows
        if int(float(row.get("run_index", -1))) == 0
    }
    for summary_row in summary_rows:
        metadata = metadata_by_group.get(str(summary_row["group_label"]), {})
        summary_row.update(
            {
                "profile_id": metadata.get("profile_id", ""),
                "model": metadata.get("model", ""),
                "base_url": metadata.get("base_url", ""),
                "scenario_type": metadata.get("scenario_type", ""),
            }
        )

    aggregated_time_series = _aggregate_selected_time_series(
        time_rows,
        group_key="group_label",
        ordered_groups=ordered_groups,
        metric_keys=EXP6_TIME_SERIES_METRICS,
        metadata_keys=["profile_id", "model", "scenario_type"],
    )
    summary_path = experiment_dir / "exp6_combined_summary.csv"
    time_series_path = experiment_dir / "exp6_combined_time_series.csv"
    report_path = experiment_dir / "exp6_combined_report.json"
    run_keys = [
        "group_label", "profile_id", "model", "base_url", "scenario_type", "run_index",
        "random_seed", "initial_group_negative_emotion", "peak", "post_peak_drop",
        "mean_step_emotion_increment", "peak_average_rise_speed", "duration",
        "final_group_negative_emotion",
        "final_high_negative_ratio", "runtime_llm_request_count", "runtime_llm_failure_count",
    ]
    summary_keys = [
        "group_label", "profile_id", "model", "base_url", "scenario_type", "runs",
        *[
            key
            for metric in ("peak", "post_peak_drop", "final_group_negative_emotion", "final_high_negative_ratio")
            for key in (f"mean_{metric}", f"std_{metric}", f"cv_{metric}")
        ],
    ]
    _save_csv_atomic(
        summary_path,
        _combine_run_and_summary_rows(
            [_select_keys(row, run_keys) for row in run_rows],
            [_select_keys(row, summary_keys) for row in summary_rows],
        ),
    )
    _save_csv_atomic(
        time_series_path,
        _combine_run_and_summary_rows(
            [
                _select_keys(
                    row,
                    [
                        "group_label", "profile_id", "model", "scenario_type", "run_index",
                        "random_seed", "step", *EXP6_TIME_SERIES_METRICS,
                    ],
                )
                for row in time_rows
            ],
            aggregated_time_series,
        ),
    )
    _save_json_atomic(
        report_path,
        {
            "experiment_name": "实验六：不同运行时LLM的鲁棒性检验",
            "status": "complete",
            "primary_baseline": {
                "profile_id": "primary",
                "model": primary_model,
                "base_url": primary_base_url,
                "source_summary": str(baseline["summary_path"]),
            },
            "completed_profiles": completed_profiles,
            "artifacts": {
                "summary_csv": str(summary_path),
                "time_series_csv": str(time_series_path),
            },
            "group_summary": summary_rows,
        },
    )
    return {
        "experiment_dir": str(experiment_dir),
        "summary_path": str(summary_path),
        "time_series_path": str(time_series_path),
        "report_path": str(report_path),
    }


def _validate_model_robustness_design(
    base_config: SimulationConfig,
    profile: LLMProfile,
) -> None:
    """确保不同模型只在运行时感知与决策层发生差异。"""

    if not base_config.runtime_llm_enabled:
        raise ValueError("实验六要求 LLM_RUNTIME_ENABLED=true。")
    if profile.profile_id == "primary":
        raise ValueError("实验六的 primary 基准必须从实验一读取，不能作为替代模型运行。")


def _preflight_model_robustness_profile(profile: LLMProfile) -> None:
    """Fail before a full experiment when a profile cannot produce structured JSON."""

    client = LLMClient(
        api_key=profile.api_key,
        base_url=profile.base_url,
        model=profile.model,
        timeout=profile.timeout,
    )
    payload = client.generate_json(build_model_preflight_prompt(), temperature=0.1)
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise RuntimeError(
            f"实验六模型预检失败：{profile.profile_id} 无法返回有效 JSON；"
            "未启动该模型的正式重复实验。"
        )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"未找到实验一基准文件：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _load_experiment_1_llm_baseline(
    base_config: SimulationConfig,
) -> dict[str, object]:
    """Load the published primary-model baseline produced by Experiment 1."""

    experiment_dir = base_config.experiments_dir / "exp1_expression_styles"
    summary_path = experiment_dir / "exp1_summary.csv"
    time_series_path = experiment_dir / "exp1_time_series.csv"
    if not summary_path.exists() or not time_series_path.exists():
        raise FileNotFoundError("未找到实验一的汇总表或时间序列表。")

    summary_rows = [
        row for row in _read_csv_rows(summary_path)
        if row.get("row_type") == "run"
    ]
    time_rows = [
        row for row in _read_csv_rows(time_series_path)
        if row.get("row_type") == "run"
    ]
    if not summary_rows or not time_rows:
        raise ValueError("实验一基准缺少原始重复结果，无法作为实验六主模型基准。")

    required_run_fields = {
        "scenario_type",
        "run_index",
        "random_seed",
        "initial_group_negative_emotion",
        "peak",
        "mean_step_emotion_increment",
        "peak_average_rise_speed",
        "final_group_negative_emotion",
        "final_high_negative_ratio",
    }
    missing_run_fields = sorted(required_run_fields.difference(summary_rows[0]))
    if missing_run_fields:
        raise ValueError(
            "实验一基准结果表字段不完整，无法启动实验六："
            f"缺少 {', '.join(missing_run_fields)}。"
        )

    runs_by_scenario: dict[str, list[dict[str, str]]] = {}
    for scenario_type in SCENARIO_TYPES:
        rows = sorted(
            [row for row in summary_rows if row.get("scenario_type") == scenario_type],
            key=lambda row: int(float(row.get("run_index", "-1"))),
        )
        if not rows:
            raise ValueError(f"实验一基准缺少场景 {scenario_type} 的原始重复结果。")
        runs_by_scenario[scenario_type] = rows

    expected_runs = len(next(iter(runs_by_scenario.values())))
    if expected_runs <= 0 or any(len(rows) != expected_runs for rows in runs_by_scenario.values()):
        raise ValueError("实验一基准重复次数不完整，无法与替代模型进行配对比较。")

    return {
        "summary_path": summary_path,
        "summary_rows": summary_rows,
        "time_rows": time_rows,
        "runs_by_scenario": runs_by_scenario,
        "runs_per_scenario": expected_runs,
        "model": str(summary_rows[0].get("runtime_llm_model", "主实验LLM")),
        "base_url": "",
    }


def run_experiment_6_model_robustness(
    base_config: SimulationConfig,
    llm_config: LLMConfig,
    profile: Optional[LLMProfile],
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict[str, object]:
    """Run one alternative model or rebuild the combined Experiment 6 artifacts."""

    if profile is None:
        return collect_experiment_6_completed_profiles(base_config)

    _validate_model_robustness_design(base_config, profile)
    baseline = _load_experiment_1_llm_baseline(base_config)
    _preflight_model_robustness_profile(profile)
    experiment_dir = base_config.experiments_dir / "exp6_model_robustness" / profile.profile_id
    profiles = [profile]
    run_rows: list[dict[str, object]] = []
    time_rows: list[dict[str, object]] = []
    results_by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    ordered_groups = [f"{profile.profile_id}__{scenario_type}" for scenario_type in SCENARIO_TYPES]

    primary_model = str(baseline["model"])
    primary_base_url = str(baseline["base_url"])

    for profile in profiles:
        profile_llm_config = profile.to_llm_config(llm_config)
        for scenario_type in SCENARIO_TYPES:
            group_label = f"{profile.profile_id}__{scenario_type}"
            baseline_runs = dict(baseline["runs_by_scenario"])[scenario_type]
            for baseline_row in baseline_runs:
                run_index = int(float(baseline_row["run_index"]))
                run_seed = int(float(baseline_row["random_seed"]))
                run_config = base_config.clone_with(random_seed=run_seed)
                if progress_callback is not None and base_config.progress_detail != "silent":
                    progress_callback(
                        f"[实验六] {profile.profile_id} | {scenario_type} | run {run_index + 1}/"
                        f"{baseline['runs_per_scenario']} 开始 | seed={run_seed}"
                    )

                simulation = EmotionDiffusionSimulation(
                    config=run_config,
                    scenario_type=scenario_type,
                    llm_config=profile_llm_config,
                    progress_callback=(
                        None
                        if progress_callback is None or base_config.progress_detail != "detailed"
                        else _build_experiment_progress_callback(
                            progress_callback=progress_callback,
                            scenario_type=scenario_type,
                            run_index=run_index,
                            experiment_name="实验六",
                            group_label=profile.profile_id,
                        )
                    ),
                )
                result = simulation.run()
                if progress_callback is not None and base_config.progress_detail != "silent":
                    progress_callback(
                        f"[实验六] {profile.profile_id} | {scenario_type} | run {run_index + 1}/"
                        f"{baseline['runs_per_scenario']} 完成 | "
                        f"peak={result.peak:.4f} | final_G={result.group_negative_emotion[-1]:.4f}"
                    )

                summary = result.summary_dict()
                summary.update(
                    {
                        "group_label": group_label,
                        "profile_id": profile.profile_id,
                        "model": profile.model,
                        "base_url": profile.base_url,
                        "scenario_type": scenario_type,
                        "run_index": run_index,
                        "random_seed": run_seed,
                    }
                )
                summary["post_peak_drop"] = max(
                    0.0,
                    float(summary["peak"]) - float(summary["final_group_negative_emotion"]),
                )
                run_rows.append(summary)
                results_by_group[group_label].append(summary)

                for row in result.time_series:
                    time_rows.append(
                        {
                            "group_label": group_label,
                            "profile_id": profile.profile_id,
                            "model": profile.model,
                            "scenario_type": scenario_type,
                            "run_index": run_index,
                            "random_seed": run_seed,
                            **row,
                        }
                    )

    summary_rows = _build_group_summary_rows(results_by_group, ordered_groups)
    metadata_by_group = {
        str(row["group_label"]): row
        for row in run_rows
        if int(row["run_index"]) == 0
    }
    for summary_row in summary_rows:
        metadata = metadata_by_group[str(summary_row["group_label"])]
        summary_row.update(
            {
                "profile_id": metadata["profile_id"],
                "model": metadata["model"],
                "base_url": metadata["base_url"],
                "scenario_type": metadata["scenario_type"],
            }
        )

    aggregated_time_series = _aggregate_selected_time_series(
        time_rows,
        group_key="group_label",
        ordered_groups=ordered_groups,
        metric_keys=EXP6_TIME_SERIES_METRICS,
        metadata_keys=["profile_id", "model", "scenario_type"],
    )

    summary_path = experiment_dir / "exp6_summary.csv"
    time_series_path = experiment_dir / "exp6_time_series.csv"
    report_path = experiment_dir / "exp6_report.json"
    run_keys = [
        "group_label", "profile_id", "model", "base_url", "scenario_type", "run_index",
        "random_seed", "initial_group_negative_emotion", "peak", "post_peak_drop",
        "mean_step_emotion_increment", "peak_average_rise_speed", "duration",
        "final_group_negative_emotion",
        "final_high_negative_ratio", "runtime_llm_request_count", "runtime_llm_failure_count",
    ]
    summary_keys = [
        "group_label", "profile_id", "model", "base_url", "scenario_type", "runs",
        *[
            key
            for metric in ("peak", "post_peak_drop", "final_group_negative_emotion", "final_high_negative_ratio")
            for key in (f"mean_{metric}", f"std_{metric}", f"cv_{metric}")
        ],
    ]
    _save_csv_atomic(
        summary_path,
        _combine_run_and_summary_rows(
            [_select_keys(row, run_keys) for row in run_rows],
            [_select_keys(row, summary_keys) for row in summary_rows],
        ),
    )
    _save_csv_atomic(
        time_series_path,
        _combine_run_and_summary_rows(
            [
                _select_keys(
                    row,
                    [
                        "group_label", "profile_id", "model", "scenario_type", "run_index",
                        "random_seed", "step", *EXP6_TIME_SERIES_METRICS,
                    ],
                )
                for row in time_rows
            ],
            aggregated_time_series,
        ),
    )
    _save_json_atomic(
        report_path,
        {
            "experiment_name": "实验六：不同运行时LLM的鲁棒性检验",
            "status": "complete",
            "profile_id": profile.profile_id,
            "design": {
                    "purpose": "复用实验一主模型的原始重复结果，并在冻结根帖、评分与互动文本材料条件下替换所有实际暴露智能体的运行时LLM；运行时调用统计仅作为执行记录，不作为结果写入门槛。",
                "primary_baseline": {
                    "profile_id": "primary",
                    "model": primary_model,
                    "base_url": primary_base_url,
                    "source_experiment": "实验一",
                    "source_summary": str(baseline["summary_path"]),
                },
                "alternative_profiles": [
                    {
                        "profile_id": profile.profile_id,
                        "model": profile.model,
                        "base_url": profile.base_url,
                    }
                ],
                "scenarios": list(SCENARIO_TYPES),
                "runs_per_condition": baseline["runs_per_scenario"],
                "runtime_llm_enabled": base_config.runtime_llm_enabled,
                "llm_agent_batch_size": base_config.llm_agent_batch_size,
                "num_agents": base_config.num_agents,
                "num_steps": base_config.num_steps,
            },
            "artifacts": {
                "summary_csv": str(summary_path),
                "time_series_csv": str(time_series_path),
            },
            "group_summary": summary_rows,
        },
    )
    try:
        combined_outputs = collect_experiment_6_completed_profiles(base_config)
    except Exception as error:
        raise RuntimeError(
            f"实验六模型 {profile.profile_id} 已完成并保存独立结果，但组合汇总失败：{error}"
        ) from error
    combined_outputs["profile_artifacts"] = {
        "experiment_dir": str(experiment_dir),
        "summary_path": str(summary_path),
        "time_series_path": str(time_series_path),
        "report_path": str(report_path),
    }
    return combined_outputs


def run_experiment_2_platform_main_effects(
    base_config: SimulationConfig,
    llm_config: LLMConfig,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict[str, object]:
    """实验二：比较三类误导场景下的平台机制过程效应。"""

    experiment_dir = base_config.experiments_dir / "exp2_platform_main_effects"
    scenario_types = list(SCENARIO_TYPES)

    # 固定非目标机制，只改变当前机制力度。
    baseline_platform = {
        "recommendation_strength": 0.40,
        "repeat_strength": 0.25,
        "visibility_strength": 0.35,
    }

    # 三档机制力度用于构造低/中/高对比。
    mechanism_conditions: list[dict[str, object]] = [
        {
            "group_label": "recommendation_low",
            "mechanism_name": "recommendation_strength",
            "mechanism_level": "low",
            "mechanism_family": "recommendation",
            "overrides": {"recommendation_strength": 0.20},
        },
        {
            "group_label": "recommendation_medium",
            "mechanism_name": "recommendation_strength",
            "mechanism_level": "medium",
            "mechanism_family": "recommendation",
            "overrides": {"recommendation_strength": 0.40},
        },
        {
            "group_label": "recommendation_high",
            "mechanism_name": "recommendation_strength",
            "mechanism_level": "high",
            "mechanism_family": "recommendation",
            "overrides": {"recommendation_strength": 0.80},
        },
        {
            "group_label": "repeat_low",
            "mechanism_name": "repeat_strength",
            "mechanism_level": "low",
            "mechanism_family": "repeat",
            "overrides": {"repeat_strength": 0.10},
        },
        {
            "group_label": "repeat_medium",
            "mechanism_name": "repeat_strength",
            "mechanism_level": "medium",
            "mechanism_family": "repeat",
            "overrides": {"repeat_strength": 0.25},
        },
        {
            "group_label": "repeat_high",
            "mechanism_name": "repeat_strength",
            "mechanism_level": "high",
            "mechanism_family": "repeat",
            "overrides": {"repeat_strength": 0.50},
        },
        {
            "group_label": "visibility_low",
            "mechanism_name": "visibility_strength",
            "mechanism_level": "low",
            "mechanism_family": "visibility",
            "overrides": {"visibility_strength": 0.10},
        },
        {
            "group_label": "visibility_medium",
            "mechanism_name": "visibility_strength",
            "mechanism_level": "medium",
            "mechanism_family": "visibility",
            "overrides": {"visibility_strength": 0.35},
        },
        {
            "group_label": "visibility_high",
            "mechanism_name": "visibility_strength",
            "mechanism_level": "high",
            "mechanism_family": "visibility",
            "overrides": {"visibility_strength": 0.70},
        },
    ]
    enabled_mechanism_families = {
        "recommendation": base_config.run_experiment_2_recommendation,
        "repeat": base_config.run_experiment_2_repeat,
        "visibility": base_config.run_experiment_2_visibility,
    }
    mechanism_conditions = [
        condition
        for condition in mechanism_conditions
        if enabled_mechanism_families.get(str(condition["mechanism_family"]), True)
    ]
    if not mechanism_conditions:
        raise ValueError(
            "RUN_EXPERIMENT_2 is enabled, but all Experiment 2 mechanism switches are disabled."
        )
    paired_seed_offsets = {
        "recommendation": 20_000,
        "repeat": 21_000,
        "visibility": 22_000,
    }
    scenario_seed_offsets = {
        "risk_exaggeration": 0,
        "selective_interpretation": 50_000,
        "emotional_wording": 100_000,
    }
    ordered_groups = [
        f"{scenario_type}_{condition['group_label']}"
        for scenario_type in scenario_types
        for condition in mechanism_conditions
    ]

    run_table_rows: list[dict[str, object]] = []
    grouped_time_series_rows: list[dict[str, object]] = []
    results_by_group: dict[str, list[dict[str, object]]] = defaultdict(list)

    for scenario_type in scenario_types:
        for condition in mechanism_conditions:
            group_label = str(condition["group_label"])
            scenario_group_label = f"{scenario_type}_{group_label}"
            mechanism_name = str(condition["mechanism_name"])
            mechanism_level = str(condition["mechanism_level"])
            mechanism_family = str(condition["mechanism_family"])
            overrides = {**baseline_platform, **dict(condition["overrides"])}

            for run_index in range(base_config.experiment_runs_per_scenario):
                run_seed = int(
                    base_config.random_seed
                    + scenario_seed_offsets.get(scenario_type, 0)
                    + paired_seed_offsets[mechanism_family]
                    + run_index
                )
                run_config = _platform_experiment_config(
                    base_config,
                    random_seed=run_seed,
                    overrides=overrides,
                )
                if progress_callback is not None and base_config.progress_detail != "silent":
                    progress_callback(
                        f"[实验二] {scenario_type}/{group_label} run {run_index + 1}/"
                        f"{base_config.experiment_runs_per_scenario} 开始 | seed={run_seed}"
                    )

                show_step_progress = progress_callback is not None and base_config.progress_detail == "detailed"
                simulation = EmotionDiffusionSimulation(
                    config=run_config,
                    scenario_type=scenario_type,
                    llm_config=llm_config,
                    progress_callback=(
                        None
                        if not show_step_progress
                        else _build_experiment_progress_callback(
                            progress_callback=progress_callback,
                            scenario_type=scenario_type,
                            run_index=run_index,
                            experiment_name="实验二",
                            group_label=group_label,
                        )
                    ),
                )
                result = simulation.run()
                run_metadata = {
                    "scenario_type": scenario_type,
                    "group_label": group_label,
                    "scenario_group_label": scenario_group_label,
                    "mechanism_name": mechanism_name,
                    "mechanism_level": mechanism_level,
                    "mechanism_family": mechanism_family,
                    "run_index": run_index,
                    "random_seed": run_seed,
                    "recommendation_strength": run_config.recommendation_strength,
                    "repeat_strength": run_config.repeat_strength,
                    "visibility_strength": run_config.visibility_strength,
                }
                run_diagnostics = _build_exp2_run_diagnostics(
                    result.time_series,
                    run_config.num_agents,
                    run_metadata,
                )
                if progress_callback is not None and base_config.progress_detail != "silent":
                    progress_callback(
                        f"[实验二] {scenario_type}/{group_label} run {run_index + 1}/"
                        f"{base_config.experiment_runs_per_scenario} 完成 | "
                        f"candidate_reach={run_diagnostics['candidate_reach_total']:.0f} | "
                        f"interaction_intensity={run_diagnostics['unit_interaction_intensity']:.3f}"
                    )

                run_table_rows.append(run_diagnostics)
                results_by_group[scenario_group_label].append(run_diagnostics)

                for row in result.time_series:
                    grouped_time_series_rows.append(
                        {
                            "scenario_group_label": scenario_group_label,
                            "group_label": group_label,
                            "scenario_type": scenario_type,
                            "mechanism_name": mechanism_name,
                            "mechanism_level": mechanism_level,
                            "mechanism_family": mechanism_family,
                            "run_index": run_index,
                            "random_seed": run_seed,
                            "recommendation_strength": run_config.recommendation_strength,
                            "repeat_strength": run_config.repeat_strength,
                            "visibility_strength": run_config.visibility_strength,
                            **row,
                        }
                    )

    summary_rows = _build_exp2_group_summary_rows(results_by_group, ordered_groups)
    aggregated_time_series = _aggregate_selected_time_series(
        grouped_time_series_rows,
        group_key="scenario_group_label",
        ordered_groups=ordered_groups,
        metric_keys=EXP2_TIME_SERIES_METRICS,
        metadata_keys=[
            "group_label",
            "scenario_type",
            "mechanism_name",
            "mechanism_level",
            "mechanism_family",
            "recommendation_strength",
            "repeat_strength",
            "visibility_strength",
        ],
    )

    summary_path = experiment_dir / "exp2_summary.csv"
    time_series_path = experiment_dir / "exp2_time_series.csv"
    report_path = experiment_dir / "exp2_report.json"

    summary_table_rows = _combine_run_and_summary_rows(
        [_select_keys(row, EXP2_RUN_KEYS) for row in run_table_rows],
        summary_rows,
    )
    time_series_table_rows = _combine_run_and_summary_rows(
        [
            _select_keys(
                row,
                [
                    "scenario_group_label",
                    "group_label",
                    "scenario_type",
                    "mechanism_name",
                    "mechanism_level",
                    "mechanism_family",
                    "run_index",
                    "random_seed",
                    "step",
                    *EXP2_TIME_SERIES_METRICS,
                ],
            )
            for row in grouped_time_series_rows
        ],
        aggregated_time_series,
    )
    _save_csv_atomic(summary_path, summary_table_rows)
    _save_csv_atomic(time_series_path, time_series_table_rows)
    _save_json_atomic(
        report_path,
        {
            "experiment_name": "实验二：平台机制主效应实验",
            "design": {
                "scenario_types": scenario_types,
                "output_focus": "实验二以过程变量为主，分别观察推荐强化对暴露总量、重复暴露对情绪记忆、网络可见性对单位用户互动强度的影响。",
                "purpose": "比较三类误导场景下平台机制强度变化对对应传播过程变量的影响。",
                "control_strategy": "同一场景、同一机制内部采用配对随机种子，低/中/高条件共享同一网络结构、agent初始状态和种子节点；非目标平台参数固定为中等基准值。",
                "baseline_platform": baseline_platform,
                "enabled_mechanism_families": enabled_mechanism_families,
                "exposure_process": (
                    "候选内容来自首轮投放、上一轮邻居互动和历史重复接触；"
                    "推荐、重复暴露与网络可见性共同进入既有暴露概率公式。"
                ),
                "conditions": mechanism_conditions,
                "runs_per_condition": base_config.experiment_runs_per_scenario,
                "num_agents": base_config.num_agents,
                "num_steps": base_config.num_steps,
            },
            "artifacts": {
                "summary_csv": str(summary_path),
                "time_series_csv": str(time_series_path),
            },
            "group_summary": summary_rows,
        },
    )

    return {
        "experiment_dir": str(experiment_dir),
        "summary_path": str(summary_path),
        "time_series_path": str(time_series_path),
        "report_path": str(report_path),
    }


def run_experiment_3_scenario_platform_interaction(
    base_config: SimulationConfig,
    llm_config: LLMConfig,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict[str, object]:
    """实验三：误导信息类型与平台综合放大条件的交互效应。"""

    experiment_dir = base_config.experiments_dir / "exp3_scenario_platform_interaction"
    platform_conditions: list[dict[str, object]] = [
        {
            "platform_level": "baseline_platform",
            "overrides": {
                "recommendation_strength": 0.40,
                "repeat_strength": 0.25,
                "visibility_strength": 0.35,
                "graph_edge_probability": base_config.graph_edge_probability,
            },
        },
        {
            "platform_level": "low_platform_amplification",
            "overrides": {
                "recommendation_strength": 0.50,
                "repeat_strength": 0.28,
                "visibility_strength": 0.38,
                "graph_edge_probability": base_config.graph_edge_probability,
            },
        },
        {
            "platform_level": "medium_platform_amplification",
            "overrides": {
                "recommendation_strength": 0.55,
                "repeat_strength": 0.32,
                "visibility_strength": 0.42,
                "graph_edge_probability": base_config.graph_edge_probability,
            },
        },
        {
            "platform_level": "high_platform_amplification",
            "overrides": {
                "recommendation_strength": 0.65,
                "repeat_strength": 0.38,
                "visibility_strength": 0.48,
                "graph_edge_probability": base_config.graph_edge_probability,
            },
        },
    ]

    ordered_groups: list[str] = [
        f"{scenario_type}__{condition['platform_level']}"
        for scenario_type in SCENARIO_TYPES
        for condition in platform_conditions
    ]
    run_table_rows: list[dict[str, object]] = []
    grouped_time_series_rows: list[dict[str, object]] = []
    results_by_group: dict[str, list[dict[str, object]]] = defaultdict(list)

    for scenario_index, scenario_type in enumerate(SCENARIO_TYPES):
        for platform_index, condition in enumerate(platform_conditions):
            platform_level = str(condition["platform_level"])
            group_label = f"{scenario_type}__{platform_level}"
            overrides = dict(condition["overrides"])

            for run_index in range(base_config.experiment_runs_per_scenario):
                run_seed = int(
                    base_config.random_seed
                    + 30_000
                    + scenario_index * 10_000
                    + run_index
                )
                run_config = _platform_experiment_config(
                    base_config,
                    random_seed=run_seed,
                    overrides=overrides,
                )
                if progress_callback is not None and base_config.progress_detail != "silent":
                    progress_callback(
                        f"[实验三] {group_label} run {run_index + 1}/"
                        f"{base_config.experiment_runs_per_scenario} 开始 | seed={run_seed}"
                    )

                show_step_progress = progress_callback is not None and base_config.progress_detail == "detailed"
                simulation = EmotionDiffusionSimulation(
                    config=run_config,
                    scenario_type=scenario_type,
                    llm_config=llm_config,
                    progress_callback=(
                        None
                        if not show_step_progress
                        else _build_experiment_progress_callback(
                            progress_callback=progress_callback,
                            scenario_type=scenario_type,
                            run_index=run_index,
                            experiment_name="实验三",
                            group_label=group_label,
                        )
                    ),
                )
                result = simulation.run()
                trace_metadata = {
                    "experiment": "experiment_3",
                    "group_label": group_label,
                    "platform_level": platform_level,
                    "run_index": run_index,
                    "random_seed": run_seed,
                }
                if progress_callback is not None and base_config.progress_detail != "silent":
                    progress_callback(
                        f"[实验三] {group_label} run {run_index + 1}/"
                        f"{base_config.experiment_runs_per_scenario} 完成 | "
                        f"peak={result.peak:.4f} | final_G={result.group_negative_emotion[-1]:.4f} | "
                        f"final_H={result.high_negative_ratio[-1]:.4f}"
                    )

                summary = result.summary_dict()
                summary.update(
                    {
                        "group_label": group_label,
                        "scenario_type": scenario_type,
                        "platform_level": platform_level,
                        "run_index": run_index,
                        "random_seed": run_seed,
                        "recommendation_strength": run_config.recommendation_strength,
                        "repeat_strength": run_config.repeat_strength,
                        "visibility_strength": run_config.visibility_strength,
                        "graph_edge_probability": run_config.graph_edge_probability,
                    }
                )
                summary["post_peak_drop"] = max(
                    0.0,
                    float(summary["peak"]) - float(summary["final_group_negative_emotion"]),
                )
                run_table_rows.append(summary)
                results_by_group[group_label].append(summary)

                for row in result.time_series:
                    grouped_time_series_rows.append(
                        {
                            "group_label": group_label,
                            "scenario_type": scenario_type,
                            "platform_level": platform_level,
                            "run_index": run_index,
                            "random_seed": run_seed,
                            **row,
                        }
                    )

    summary_rows = _build_group_summary_rows(results_by_group, ordered_groups)
    aggregated_time_series = _aggregate_selected_time_series(
        grouped_time_series_rows,
        group_key="group_label",
        ordered_groups=ordered_groups,
        metric_keys=EXP3_TIME_SERIES_METRICS,
        metadata_keys=["scenario_type", "platform_level"],
    )

    summary_path = experiment_dir / "exp3_summary.csv"
    time_series_path = experiment_dir / "exp3_time_series.csv"
    report_path = experiment_dir / "exp3_report.json"

    summary_table_rows = _combine_run_and_summary_rows(
        [
            _select_keys(
                row,
                [
                    "group_label",
                    "scenario_type",
                    "platform_level",
                    "run_index",
                    "random_seed",
                    "peak",
                    "post_peak_drop",
                    "peak_average_rise_speed",
                ],
            )
            for row in run_table_rows
        ],
        [
            _select_keys(
                row,
                [
                    "group_label",
                    "scenario_type",
                    "platform_level",
                    "recommendation_strength",
                    "repeat_strength",
                    "visibility_strength",
                    "runs",
                    *[
                        key
                        for metric in ("peak", "post_peak_drop", "peak_average_rise_speed")
                        for key in (f"mean_{metric}", f"std_{metric}", f"cv_{metric}")
                    ],
                ],
            )
            for row in summary_rows
        ],
    )
    time_series_table_rows = _combine_run_and_summary_rows(
        [
            _select_keys(
                row,
                [
                    "group_label",
                    "scenario_type",
                    "platform_level",
                    "run_index",
                    "random_seed",
                    "step",
                    *EXP3_TIME_SERIES_METRICS,
                ],
            )
            for row in grouped_time_series_rows
        ],
        aggregated_time_series,
    )
    _save_csv_atomic(summary_path, summary_table_rows)
    _save_csv_atomic(time_series_path, time_series_table_rows)
    _save_json_atomic(
        report_path,
        {
            "experiment_name": "实验三：误导信息类型与平台综合放大交互实验",
            "design": {
                "purpose": "比较三类误导信息在五档平台综合放大条件下的群体负面情绪峰值、峰值增量与达峰平均上升速度差异。",
                "scenarios": list(SCENARIO_TYPES),
                "platform_conditions": platform_conditions,
                "runs_per_condition": base_config.experiment_runs_per_scenario,
                "num_agents": base_config.num_agents,
                "num_steps": base_config.num_steps,
            },
            "artifacts": {
                "summary_csv": str(summary_path),
                "time_series_csv": str(time_series_path),
            },
            "group_summary": summary_rows,
        },
    )

    return {
        "experiment_dir": str(experiment_dir),
        "summary_path": str(summary_path),
        "time_series_path": str(time_series_path),
        "report_path": str(report_path),
    }


def run_experiment_4_sensitivity_analysis(
    base_config: SimulationConfig,
    llm_config: LLMConfig,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict[str, object]:
    """实验四：参数敏感性实验。"""

    experiment_dir = base_config.experiments_dir / "exp4_sensitivity_analysis"
    conditions: list[dict[str, object]] = []

    beta_values = [0.14, 0.16, 0.18, 0.20, 0.22]
    gamma_values = [0.16, 0.18, 0.20, 0.22, 0.24]
    if base_config.run_experiment_4_beta_gamma:
        for beta_value in beta_values:
            for gamma_value in gamma_values:
                conditions.append(
                    {
                        "group_label": f"emotion_beta_{beta_value:.2f}_gamma_{gamma_value:.2f}",
                        "analysis_part": "emotion_mechanism",
                        "scenario_type": "selective_interpretation",
                        "overrides": {
                            "beta_stimulus": beta_value,
                            "gamma_feedback": gamma_value,
                        },
                    }
                )

    memory_weight_values = [0.20, 0.22, 0.24, 0.26, 0.28]
    if base_config.run_experiment_4_memory_weight:
        for memory_weight in memory_weight_values:
            conditions.append(
                {
                    "group_label": f"memory_weight_{memory_weight:.2f}",
                    "analysis_part": "memory_weight",
                    "scenario_type": "selective_interpretation",
                    "overrides": {
                        "memory_emotion_weight": memory_weight,
                    },
                }
            )

    emotion_decay_values = [0.36, 0.38, 0.40, 0.42, 0.44]
    if base_config.run_experiment_4_emotion_decay:
        for emotion_decay in emotion_decay_values:
            conditions.append(
                {
                    "group_label": f"emotion_decay_{emotion_decay:.2f}",
                    "analysis_part": "emotion_decay",
                        "scenario_type": "selective_interpretation",
                        "overrides": {
                            "emotion_decay": emotion_decay,
                        },
                }
            )

    content_age_values = [3, 4, 5, 6, 7]
    if base_config.run_experiment_4_content_age:
        for content_age in content_age_values:
            conditions.append(
                {
                    "group_label": f"content_age_{content_age}",
                    "analysis_part": "content_age",
                    "scenario_type": "selective_interpretation",
                    "overrides": {
                        "max_content_age": content_age,
                    },
                }
            )

    network_density_values = [0.04, 0.07, 0.10, 0.13, 0.16]
    if base_config.run_experiment_4_network_density:
        for edge_probability in network_density_values:
            conditions.append(
                {
                    "group_label": f"network_density_{edge_probability:.2f}",
                    "analysis_part": "network_density",
                        "scenario_type": "selective_interpretation",
                    "overrides": {
                        "graph_edge_probability": edge_probability,
                    },
                }
            )

    if not conditions:
        raise ValueError("RUN_EXPERIMENT_4 is enabled, but all experiment 4 sub-experiment switches are disabled.")

    paired_seed_offsets = {
        "emotion_mechanism": 40_000,
        "memory_weight": 41_000,
        "emotion_decay": 42_000,
        "content_age": 43_000,
        "network_density": 44_000,
    }
    ordered_groups = [str(condition["group_label"]) for condition in conditions]
    run_table_rows: list[dict[str, object]] = []
    results_by_group: dict[str, list[dict[str, object]]] = defaultdict(list)

    for condition in conditions:
        group_label = str(condition["group_label"])
        analysis_part = str(condition["analysis_part"])
        scenario_type = str(condition["scenario_type"])
        overrides = dict(condition["overrides"])

        for run_index in range(base_config.experiment_runs_per_scenario):
            run_seed = int(base_config.random_seed + paired_seed_offsets[analysis_part] + run_index)
            run_config = _platform_experiment_config(
                base_config,
                random_seed=run_seed,
                overrides=overrides,
            )
            if progress_callback is not None and base_config.progress_detail != "silent":
                progress_callback(
                    f"[实验四] {group_label} run {run_index + 1}/"
                    f"{base_config.experiment_runs_per_scenario} 开始 | seed={run_seed}"
                )

            show_step_progress = progress_callback is not None and base_config.progress_detail == "detailed"
            simulation = EmotionDiffusionSimulation(
                config=run_config,
                scenario_type=scenario_type,
                llm_config=llm_config,
                progress_callback=(
                    None
                    if not show_step_progress
                    else _build_experiment_progress_callback(
                        progress_callback=progress_callback,
                        scenario_type=scenario_type,
                        run_index=run_index,
                        experiment_name="实验四",
                        group_label=group_label,
                    )
                ),
            )
            result = simulation.run()
            if progress_callback is not None and base_config.progress_detail != "silent":
                progress_callback(
                    f"[实验四] {group_label} run {run_index + 1}/"
                    f"{base_config.experiment_runs_per_scenario} 完成 | "
                    f"peak={result.peak:.4f} | final_G={result.group_negative_emotion[-1]:.4f} | "
                    f"final_H={result.high_negative_ratio[-1]:.4f}"
                )

            summary = result.summary_dict()
            summary.update(
                {
                    "group_label": group_label,
                    "analysis_part": analysis_part,
                    "scenario_type": scenario_type,
                    "run_index": run_index,
                    "random_seed": run_seed,
                    "beta_stimulus": run_config.beta_stimulus,
                    "gamma_feedback": run_config.gamma_feedback,
                    "memory_emotion_weight": run_config.memory_emotion_weight,
                    "emotion_decay": run_config.emotion_decay,
                    "repeat_strength": run_config.repeat_strength,
                    "visibility_strength": run_config.visibility_strength,
                    "recommendation_strength": run_config.recommendation_strength,
                    "max_content_age": run_config.max_content_age,
                    "graph_edge_probability": run_config.graph_edge_probability,
                }
            )
            run_table_rows.append(summary)
            results_by_group[group_label].append(summary)

    summary_rows = _build_group_summary_rows(results_by_group, ordered_groups)

    summary_path = experiment_dir / "exp4_summary.csv"
    time_series_path = experiment_dir / "exp4_time_series.csv"
    report_path = experiment_dir / "exp4_report.json"

    summary_table_rows = _combine_run_and_summary_rows(
        [
            _select_keys(
                row,
                [
                    "group_label",
                    "analysis_part",
                    "scenario_type",
                    "run_index",
                    "random_seed",
                    "beta_stimulus",
                    "gamma_feedback",
                    "memory_emotion_weight",
                    "emotion_decay",
                    "max_content_age",
                    "graph_edge_probability",
                    "peak",
                ],
            )
            for row in run_table_rows
        ],
        [
            _select_keys(
                row,
                [
                    "group_label",
                    "analysis_part",
                    "scenario_type",
                    "beta_stimulus",
                    "gamma_feedback",
                    "memory_emotion_weight",
                    "emotion_decay",
                    "max_content_age",
                    "graph_edge_probability",
                    "runs",
                    *[
                        key
                        for metric in ("peak",)
                        for key in (f"mean_{metric}", f"std_{metric}", f"cv_{metric}")
                    ],
                ],
            )
            for row in summary_rows
        ],
    )
    _save_csv_atomic(summary_path, summary_table_rows)
    # Experiment 4 compares parameter-response results rather than trajectories.
    time_series_path.unlink(missing_ok=True)
    _save_json_atomic(
        report_path,
        {
            "experiment_name": "实验四：参数敏感性实验",
            "design": {
                "purpose": "检验情绪机制参数和平台机制参数变化对群体负面情绪扩散结果的影响。",
                "enabled_subexperiments": {
                    "beta_gamma": base_config.run_experiment_4_beta_gamma,
                    "memory_weight": base_config.run_experiment_4_memory_weight,
                    "emotion_decay": base_config.run_experiment_4_emotion_decay,
                    "content_age": base_config.run_experiment_4_content_age,
                    "network_density": base_config.run_experiment_4_network_density,
                },
                "control_strategy": "同一敏感性子实验内的全部参数条件使用配对随机种子，共享网络结构和代理初始状态。",
                "emotion_mechanism": {
                    "scenario_type": "selective_interpretation",
                    "beta_stimulus": beta_values,
                    "gamma_feedback": gamma_values,
                    "main_metric": "mean_peak",
                },
                "memory_weight_sensitivity": {
                    "scenario_type": "selective_interpretation",
                    "memory_emotion_weight": memory_weight_values,
                    "baseline_value": base_config.memory_emotion_weight,
                    "main_metrics": ["mean_peak"],
                },
                "emotion_decay_sensitivity": {
                    "scenario_type": "selective_interpretation",
                    "emotion_decay": emotion_decay_values,
                    "baseline_value": base_config.emotion_decay,
                    "main_metrics": ["mean_peak"],
                },
                "content_age_sensitivity": {
                    "scenario_type": "selective_interpretation",
                    "max_content_age": content_age_values,
                    "baseline_value": base_config.max_content_age,
                    "main_metrics": ["mean_peak"],
                },
                "network_density_sensitivity": {
                    "scenario_type": "selective_interpretation",
                    "graph_edge_probability": network_density_values,
                    "baseline_value": base_config.graph_edge_probability,
                    "main_metrics": ["mean_peak"],
                },
                "runs_per_condition": base_config.experiment_runs_per_scenario,
                "num_agents": base_config.num_agents,
                "num_steps": base_config.num_steps,
            },
            "artifacts": {
                "summary_csv": str(summary_path),
            },
            "group_summary": summary_rows,
        },
    )

    return {
        "experiment_dir": str(experiment_dir),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
    }


def _experiment_7_conditions(base_config: SimulationConfig) -> list[dict[str, object]]:
    """Build one-factor scene-parameter conditions without changing other scenarios."""

    specifications = (
        (
            "risk_boost",
            "risk_exaggeration",
            "risk_boost",
            "风险强化参数",
            base_config.experiment_7_risk_boost_values,
        ),
        (
            "memory_boost",
            "selective_interpretation",
            "memory_boost",
            "记忆强化参数",
            base_config.experiment_7_memory_boost_values,
        ),
        (
            "share_boost",
            "emotional_wording",
            "share_boost",
            "互动强化参数",
            base_config.experiment_7_share_boost_values,
        ),
    )
    study_stimuli = load_experimental_stimuli(EXPERIMENTAL_STIMULI_PATH)
    conditions: list[dict[str, object]] = []
    for analysis_part, scenario_type, parameter_name, parameter_label, values in specifications:
        study_parameters = aggregate_strategy_parameters(study_stimuli.get(scenario_type, []))
        base_value = float(study_parameters[parameter_name])
        normalized_values = tuple(float(value) for value in values)
        if not normalized_values:
            raise ValueError(f"实验七 {parameter_label} 未配置敏感性取值。")
        reference_center = normalized_values[len(normalized_values) // 2]
        shifted_values = tuple(
            min(1.0, max(0.0, base_value + value - reference_center))
            for value in normalized_values
        )
        for value in shifted_values:
            conditions.append(
                {
                    "group_label": f"{analysis_part}_{value:.4f}",
                    "analysis_part": analysis_part,
                    "scenario_type": scenario_type,
                    "parameter_name": parameter_name,
                    "parameter_label": parameter_label,
                    "parameter_value": value,
                    "baseline_value": base_value,
                    "is_baseline": abs(value - base_value) < 1e-9,
                }
            )
    return conditions


def _build_experiment_7_scenario_override(
    scenario_type: str,
    parameter_name: str,
    parameter_value: float,
) -> ScenarioConfig:
    study_stimuli = load_experimental_stimuli(EXPERIMENTAL_STIMULI_PATH)
    parameters = aggregate_strategy_parameters(study_stimuli.get(scenario_type, []))
    values = {
        "memory_boost": parameters["memory_boost"],
        "share_boost": parameters["share_boost"],
        "risk_boost": parameters["risk_boost"],
    }
    values[parameter_name] = float(parameter_value)
    return ScenarioConfig(scene_name=scenario_type, **values)


def _build_experiment_7_run_row(
    summary: dict[str, object],
    time_rows: list[dict[str, object]],
    condition: dict[str, object],
    *,
    run_index: int,
    random_seed: int,
    num_agents: int,
    baseline_source: str,
) -> dict[str, object]:
    peak = float(summary.get("peak", 0.0))
    final_emotion = float(summary.get("final_group_negative_emotion", 0.0))
    textual_interactions = sum(
        float(row.get("derived_comment_count", 0.0))
        + float(row.get("derived_reply_count", 0.0))
        for row in time_rows
    )
    return {
        "group_label": condition["group_label"],
        "analysis_part": condition["analysis_part"],
        "scenario_type": condition["scenario_type"],
        "parameter_name": condition["parameter_name"],
        "parameter_label": condition["parameter_label"],
        "parameter_value": condition["parameter_value"],
        "baseline_value": condition["baseline_value"],
        "is_baseline": condition["is_baseline"],
        "baseline_source": baseline_source,
        "run_index": run_index,
        "random_seed": random_seed,
        "initial_group_negative_emotion": float(summary.get("initial_group_negative_emotion", 0.0)),
        "peak": peak,
        "final_group_negative_emotion": final_emotion,
        "final_high_negative_ratio": float(summary.get("final_high_negative_ratio", 0.0)),
        "post_peak_drop": max(0.0, peak - final_emotion),
        "emotion_retention_ratio": final_emotion / peak if peak > 1e-12 else 0.0,
        "unit_interaction_intensity": textual_interactions / max(1, num_agents),
    }


def _build_experiment_7_summary_rows(
    results_by_group: dict[str, list[dict[str, object]]],
    ordered_groups: list[str],
) -> list[dict[str, object]]:
    metric_names = (
        "peak",
        "final_group_negative_emotion",
        "final_high_negative_ratio",
        "post_peak_drop",
        "emotion_retention_ratio",
        "unit_interaction_intensity",
    )
    metadata_keys = (
        "analysis_part",
        "scenario_type",
        "parameter_name",
        "parameter_label",
        "parameter_value",
        "baseline_value",
        "is_baseline",
        "baseline_source",
    )
    summary_rows: list[dict[str, object]] = []
    for group_label in ordered_groups:
        rows = results_by_group.get(group_label, [])
        if not rows:
            continue
        summary_row: dict[str, object] = {"group_label": group_label, "runs": len(rows)}
        for key in metadata_keys:
            summary_row[key] = rows[0][key]
        for metric in metric_names:
            values = [float(row[metric]) for row in rows]
            metric_mean = _safe_mean(values)
            metric_std = _safe_std(values)
            summary_row[f"mean_{metric}"] = metric_mean
            summary_row[f"std_{metric}"] = metric_std
            summary_row[f"cv_{metric}"] = _safe_cv(metric_mean, metric_std)
        summary_rows.append(summary_row)
    return summary_rows


def _append_experiment_7_time_rows(
    destination: list[dict[str, object]],
    time_rows: list[dict[str, object]],
    run_row: dict[str, object],
) -> None:
    for row in time_rows:
        destination.append(
            {
                "group_label": run_row["group_label"],
                "analysis_part": run_row["analysis_part"],
                "scenario_type": run_row["scenario_type"],
                "parameter_name": run_row["parameter_name"],
                "parameter_label": run_row["parameter_label"],
                "parameter_value": run_row["parameter_value"],
                "baseline_value": run_row["baseline_value"],
                "is_baseline": run_row["is_baseline"],
                "baseline_source": run_row["baseline_source"],
                "run_index": run_row["run_index"],
                "random_seed": run_row["random_seed"],
                "step": _as_step(row.get("step", 0)),
                "group_negative_emotion": float(row.get("group_negative_emotion", 0.0)),
                "high_negative_ratio": float(row.get("high_negative_ratio", 0.0)),
            }
        )


def run_experiment_7_strategy_parameter_sensitivity(
    base_config: SimulationConfig,
    llm_config: LLMConfig,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict[str, object]:
    """Experiment 7: local one-factor sensitivity of the three scene-level parameters."""

    experiment_dir = base_config.experiments_dir / "exp7_strategy_parameter_sensitivity"
    conditions = _experiment_7_conditions(base_config)
    ordered_groups = [str(condition["group_label"]) for condition in conditions]
    run_rows: list[dict[str, object]] = []
    time_rows: list[dict[str, object]] = []
    results_by_group: dict[str, list[dict[str, object]]] = defaultdict(list)

    conditions_by_part: dict[str, list[dict[str, object]]] = defaultdict(list)
    for condition in conditions:
        conditions_by_part[str(condition["analysis_part"])].append(condition)

    paired_seed_offsets = {
        "risk_boost": 70_000,
        "memory_boost": 71_000,
        "share_boost": 72_000,
    }
    for analysis_part, part_conditions in conditions_by_part.items():
        baseline_condition = next(condition for condition in part_conditions if bool(condition["is_baseline"]))
        scenario_type = str(baseline_condition["scenario_type"])
        run_specs = [
            (
                run_index,
                int(base_config.random_seed + paired_seed_offsets[analysis_part] + run_index),
            )
            for run_index in range(base_config.experiment_runs_per_scenario)
        ]

        for condition in part_conditions:
            group_label = str(condition["group_label"])
            for run_index, random_seed in run_specs:
                run_config = base_config.clone_with(random_seed=random_seed)
                scenario_override = _build_experiment_7_scenario_override(
                    scenario_type,
                    str(condition["parameter_name"]),
                    float(condition["parameter_value"]),
                )
                if progress_callback is not None and base_config.progress_detail != "silent":
                    progress_callback(
                        f"[实验七] {group_label} run {run_index + 1}/{len(run_specs)} 开始 | "
                        f"seed={random_seed}"
                    )
                simulation = EmotionDiffusionSimulation(
                    config=run_config,
                    scenario_type=scenario_type,
                    scenario_override=scenario_override,
                    llm_config=llm_config,
                )
                result = simulation.run()
                result_summary = result.summary_dict()
                result_summary["num_agents"] = run_config.num_agents
                run_row = _build_experiment_7_run_row(
                    result_summary,
                    result.time_series,
                    condition,
                    run_index=run_index,
                    random_seed=random_seed,
                    num_agents=run_config.num_agents,
                    baseline_source="simulated",
                )
                run_rows.append(run_row)
                results_by_group[group_label].append(run_row)
                _append_experiment_7_time_rows(time_rows, result.time_series, run_row)
                if progress_callback is not None and base_config.progress_detail != "silent":
                    progress_callback(
                        f"[实验七] {group_label} run {run_index + 1}/{len(run_specs)} 完成 | "
                        f"peak={run_row['peak']:.4f} | final_G={run_row['final_group_negative_emotion']:.4f}"
                    )

    summary_rows = _build_experiment_7_summary_rows(results_by_group, ordered_groups)
    aggregated_time_series = _aggregate_selected_time_series(
        time_rows,
        group_key="group_label",
        ordered_groups=ordered_groups,
        metric_keys=["group_negative_emotion", "high_negative_ratio"],
        metadata_keys=[
            "analysis_part",
            "scenario_type",
            "parameter_name",
            "parameter_label",
            "parameter_value",
            "baseline_value",
            "is_baseline",
            "baseline_source",
        ],
    )

    summary_path = experiment_dir / "exp7_summary.csv"
    time_series_path = experiment_dir / "exp7_time_series.csv"
    report_path = experiment_dir / "exp7_report.json"
    _save_csv_atomic(summary_path, _combine_run_and_summary_rows(run_rows, summary_rows))
    _save_csv_atomic(
        time_series_path,
        _combine_run_and_summary_rows(time_rows, aggregated_time_series),
    )
    _save_json_atomic(
        report_path,
        {
            "experiment_name": "实验七：误导策略特征参数局部敏感性分析",
            "design": {
                "baseline_policy": "实验七独立运行；各子实验内部的全部参数水平共享新的配对随机种子。",
                "runtime_llm_enabled": base_config.runtime_llm_enabled,
                "conditions": conditions,
                "runs_per_group": {
                    group_label: len(rows)
                    for group_label, rows in results_by_group.items()
                },
            },
            "artifacts": {
                "summary_csv": str(summary_path),
                "time_series_csv": str(time_series_path),
            },
            "group_summary": summary_rows,
        },
    )
    return {
        "experiment_dir": str(experiment_dir),
        "summary_path": str(summary_path),
        "time_series_path": str(time_series_path),
        "report_path": str(report_path),
    }


def _format_simulation_progress(prefix: str, event: str, payload: dict[str, object]) -> str:
    """把仿真事件格式化为可读日志。"""

    detail = str(payload.get("progress_detail", "brief")).lower()
    if detail == "silent":
        return ""

    if event == "simulation_start":
        return (
            f"{prefix} 开始 | agents={payload['num_agents']} | steps={payload['num_steps']} | "
            f"seed_nodes={payload['seed_nodes']} | event={payload['source_event_id']}"
        )
    if event == "step_start":
        return f"{prefix} Step {int(payload['step']) + 1}/{payload['total_steps']} 开始"
    if event == "step_interaction":
        return (
            f"{prefix} Step {int(payload['step']) + 1}/{payload['total_steps']} "
            "交互计算中"
        )
    if event == "agent_done":
        if detail != "detailed":
            return ""
        return (
            f"{prefix} Step {int(payload['step']) + 1}/{payload['total_steps']} "
            f"交互计算中 | 已完成 {payload['finished_agents']}/{payload['total_agents']} 个 agent | "
            f"并发数={payload.get('parallel_workers', 1)}"
        )
    if event == "step_end":
        if detail != "detailed":
            return (
                f"{prefix} Step {int(payload['step']) + 1}/{payload['total_steps']} 结束 | "
                f"G={float(payload['group_negative_emotion']):.4f} | "
                f"H={float(payload['high_negative_ratio']):.4f} | "
                f"V={float(payload['group_exposure_rate']):.4f} | "
                f"Q={float(payload['group_activation_rate']):.4f} | "
                f"I={float(payload.get('interaction_rate', 0.0)):.4f}"
            )
        return (
            f"{prefix} Step {int(payload['step']) + 1}/{payload['total_steps']} 结束 | "
            f"G={float(payload['group_negative_emotion']):.4f} | "
            f"H={float(payload['high_negative_ratio']):.4f} | "
            f"V={float(payload['group_exposure_rate']):.4f} | "
            f"Q={float(payload['group_activation_rate']):.4f} | "
            f"I={float(payload.get('interaction_rate', 0.0)):.4f} | "
            f"exposed={int(payload.get('exposure_count', 0))} | "
            f"interacted={int(payload.get('interaction_count', 0))}"
        )
    if event == "simulation_end":
        return (
            f"{prefix} 结束 | peak={float(payload['peak']):.4f} | "
            f"final_G={float(payload['final_group_negative_emotion']):.4f}"
        )
    return f"{prefix} {event}: {payload}"
