from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from config import FIGURES_DIR, OUTPUTS_DIR


EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"
TABLES_DIR = OUTPUTS_DIR / "tables"

SCENARIO_ORDER = ["risk_exaggeration", "selective_interpretation", "emotional_wording"]
EXP1_SCENARIO_ORDER = [*SCENARIO_ORDER, "compound_amplification"]
SCENARIO_LABELS = {
    "risk_exaggeration": "风险夸大",
    "selective_interpretation": "选择性解释",
    "emotional_wording": "情绪化措辞",
    "compound_amplification": "复合强化",
}
LEVEL_ORDER = ["low", "medium", "high"]
LEVEL_LABELS = {"low": "低强度", "medium": "中强度", "high": "高强度"}
PLATFORM_ORDER = [
    "baseline_platform",
    "low_platform_amplification",
    "medium_platform_amplification",
    "high_platform_amplification",
]
EMOTION_RESPONSE_ORDER = ["anxiety", "anger", "suspicion", "none"]
EMOTION_RESPONSE_LABELS = {
    "anxiety": "焦虑/恐惧",
    "anger": "愤怒/厌恶",
    "suspicion": "怀疑/不安",
    "none": "无明显负面情绪",
}
PLATFORM_LABELS = {
    "baseline_platform": "基准平台",
    "low_platform_amplification": "轻度放大",
    "medium_platform_amplification": "中等平台放大",
    "high_platform_amplification": "高平台放大",
}


def _setup_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "figure.dpi": 140,
            "savefig.dpi": 600,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
        }
    )


def _save_publication_figure(fig: plt.Figure, output_path: Path) -> None:
    """Export one PNG artifact per figure to keep experiment outputs compact."""

    fig.savefig(output_path, bbox_inches="tight")


EXP2_LEVEL_COLORS = {"low": "#7FA8C9", "medium": "#E7B866", "high": "#C7685A"}
EXP1_SCENARIO_COLORS = {
    "risk_exaggeration": "#E9605B",
    "selective_interpretation": "#F2A12A",
    "emotional_wording": "#5487B8",
    "compound_amplification": "#7B6AA2",
}
EXP1_SCENARIO_STYLES = {
    "risk_exaggeration": {"linestyle": "-", "marker": "o"},
    "selective_interpretation": {"linestyle": "--", "marker": "s"},
    "emotional_wording": {"linestyle": "-.", "marker": "^"},
    "compound_amplification": {"linestyle": ":", "marker": "D"},
}
EMOTION_COMPOSITION_COLORS = {
    "anxiety": "#7FA8C9",
    "anger": "#E58D7E",
    "suspicion": "#AF92CF",
    "none": "#C6D0D7",
}
EXP5_SCALE_COLORS = {
    200: "#7EAD83",
    300: "#E0A458",
    400: "#BF6B62",
    500: "#8D78B7",
}
EXP6_PROFILE_COLORS = ["#5B8DB8", "#D9854C", "#759E6A", "#9A7BB8", "#B85C75"]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    _ensure_dir(path.parent)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _group_rows(rows: Iterable[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key, "")), []).append(row)
    for group_rows in grouped.values():
        group_rows.sort(key=lambda item: _float(item, "step"))
    return grouped


def _split_typed_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Separate raw repetitions from aggregate rows in the compact output tables."""

    if not any(row.get("row_type") for row in rows):
        return [], rows
    return (
        [row for row in rows if row.get("row_type") == "run"],
        [row for row in rows if row.get("row_type") == "summary"],
    )


def _series(rows: list[dict[str, str]], value_key: str) -> tuple[list[float], list[float]]:
    ordered = sorted(rows, key=lambda item: _float(item, "step"))
    return [_float(row, "step") for row in ordered], [_float(row, value_key) for row in ordered]


def _sample_std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _matches(row: dict[str, str], filters: dict[str, str]) -> bool:
    return all(str(row.get(key, "")) == str(value) for key, value in filters.items())


def _step_std(
    raw_rows: list[dict[str, str]],
    value_key: str,
    filters: dict[str, str],
) -> dict[int, float]:
    buckets: dict[int, list[float]] = defaultdict(list)
    for row in raw_rows:
        if _matches(row, filters):
            buckets[int(_float(row, "step"))].append(_float(row, value_key))
    return {step: _sample_std(values) for step, values in buckets.items()}


def _series_with_std(
    rows: list[dict[str, str]],
    mean_key: str,
    std_key: str,
    raw_rows: list[dict[str, str]],
    raw_key: str,
    filters: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read stored SD, with a raw-series fallback for historical outputs."""

    x_values, mean_values = _series(rows, mean_key)
    stored_std = [_float(row, std_key) for row in sorted(rows, key=lambda row: _float(row, "step"))]
    if raw_rows:
        raw_std = _step_std(raw_rows, raw_key, filters)
        stored_std = [raw_std.get(int(step), value) for step, value in zip(x_values, stored_std)]
    return (
        np.asarray(x_values, dtype=float),
        np.asarray(mean_values, dtype=float),
        np.asarray(stored_std, dtype=float),
    )


def _draw_exp1_trajectory_with_sd(
    axis: plt.Axes,
    rows: list[dict[str, str]],
    raw_rows: list[dict[str, str]],
    *,
    mean_key: str,
    std_key: str,
    raw_key: str,
    ylabel: str,
) -> None:
    """Draw one Experiment 1 trajectory panel as mean +/- one sample SD."""

    grouped = _group_rows(rows, "scenario_type")
    for scenario_type in EXP1_SCENARIO_ORDER:
        scenario_rows = grouped.get(scenario_type, [])
        if not scenario_rows:
            continue
        x_values, mean_values, std_values = _series_with_std(
            scenario_rows,
            mean_key,
            std_key,
            raw_rows,
            raw_key,
            {"scenario_type": scenario_type},
        )
        color = EXP1_SCENARIO_COLORS[scenario_type]
        style = EXP1_SCENARIO_STYLES[scenario_type]
        axis.fill_between(
            x_values,
            np.clip(mean_values - std_values, 0.0, 1.0),
            np.clip(mean_values + std_values, 0.0, 1.0),
            color=color,
            alpha=0.29,
            linewidth=0.25,
            edgecolor=color,
            zorder=1,
        )
        axis.plot(
            x_values,
            mean_values,
            color=color,
            linewidth=1.75,
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=3.8,
            markeredgecolor="#303030",
            markeredgewidth=0.35,
            label=SCENARIO_LABELS[scenario_type],
            zorder=2,
        )
    axis.set_xlabel("仿真步", fontsize=10.5)
    axis.set_ylabel(ylabel, fontsize=10.5)
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", color="#E5E9ED", linewidth=0.7)
    axis.set_axisbelow(True)


def _plot_exp1_trajectory_figure(
    rows: list[dict[str, str]],
    raw_rows: list[dict[str, str]],
    *,
    mean_key: str,
    std_key: str,
    raw_key: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Render one title-free Experiment 1 trajectory with its full-process SD band."""

    _setup_matplotlib()
    _ensure_dir(output_path.parent)
    fig, axis = plt.subplots(figsize=(7.35, 4.85))
    _draw_exp1_trajectory_with_sd(
        axis,
        rows,
        raw_rows,
        mean_key=mean_key,
        std_key=std_key,
        raw_key=raw_key,
        ylabel=ylabel,
    )
    axis.legend(loc="upper left", frameon=False, fontsize=9.5, handlelength=2.7)
    fig.tight_layout()
    _save_publication_figure(fig, output_path)
    plt.close(fig)


def _plot_exp1_dynamic_emotion_composition(
    raw_rows: list[dict[str, str]],
    output_path: Path,
    summary_rows: list[dict[str, str]] | None = None,
) -> None:
    """Render the run-balanced perceived-emotion composition of all normal users."""

    _setup_matplotlib()
    _ensure_dir(output_path.parent)
    fig, axes = plt.subplots(4, 1, figsize=(8.2, 10.2), sharex=True, sharey=True)
    emotion_specs = [
        ("anxiety", EMOTION_RESPONSE_LABELS["anxiety"]),
        ("anger", EMOTION_RESPONSE_LABELS["anger"]),
        ("suspicion", EMOTION_RESPONSE_LABELS["suspicion"]),
        ("none", EMOTION_RESPONSE_LABELS["none"]),
    ]
    for axis, scenario_type in zip(axes, EXP1_SCENARIO_ORDER):
        scenario_rows = [row for row in raw_rows if row.get("scenario_type") == scenario_type]
        if not scenario_rows and summary_rows:
            scenario_rows = [row for row in summary_rows if row.get("scenario_type") == scenario_type]
        by_step_run: dict[int, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
        for row in scenario_rows:
            total_key = "emotion_distribution_total" if raw_rows else "mean_emotion_distribution_total"
            if _float(row, total_key) <= 0.0:
                continue
            by_step_run[int(_float(row, "step"))][str(row.get("run_index", ""))].append(row)

        steps = sorted(by_step_run)
        if not steps:
            axis.text(0.5, 0.5, "缺少全体用户情绪结构序列", transform=axis.transAxes, ha="center", va="center")
            continue
        composition = {key: [] for key, _ in emotion_specs}
        # The first three cumulative boundaries fully determine a four-part composition.
        boundary_mean = [[], [], []]
        boundary_lower = [[], [], []]
        boundary_upper = [[], [], []]
        for step in steps:
            run_rows = [rows[0] for rows in by_step_run[step].values() if rows]
            run_distributions: list[list[float]] = []
            for row in run_rows:
                prefix = "perceived_" if raw_rows else "mean_perceived_"
                values = [_float(row, f"{prefix}{key}_mean") for key, _ in emotion_specs]
                total = sum(values)
                if total > 0.0:
                    run_distributions.append([value / total for value in values])
            if not run_distributions:
                for key, _ in emotion_specs:
                    composition[key].append(np.nan)
                continue
            averaged = np.mean(np.asarray(run_distributions, dtype=float), axis=0)
            for index, (key, _) in enumerate(emotion_specs):
                composition[key].append(float(averaged[index]))

            cumulative = np.cumsum(np.asarray(run_distributions, dtype=float)[:, :3], axis=1)
            for boundary_index in range(3):
                center = float(np.mean(cumulative[:, boundary_index]))
                spread = float(np.std(cumulative[:, boundary_index], ddof=1)) if len(run_distributions) > 1 else 0.0
                boundary_mean[boundary_index].append(center)
                boundary_lower[boundary_index].append(
                    max(0.0, center - spread)
                )
                boundary_upper[boundary_index].append(
                    min(1.0, center + spread)
                )

        values = [np.asarray(composition[key], dtype=float) for key, _ in emotion_specs]
        axis.stackplot(
            steps,
            values,
            colors=[EMOTION_COMPOSITION_COLORS[key] for key, _ in emotion_specs],
            labels=[label for _, label in emotion_specs],
            alpha=0.92,
            linewidth=0.55,
            edgecolor="white",
        )
        if any(len(values) > 1 for values in by_step_run.values()):
            for center, lower, upper in zip(boundary_mean, boundary_lower, boundary_upper):
                axis.fill_between(
                    steps,
                    lower,
                    upper,
                    color="#243746",
                    alpha=0.44,
                    linewidth=0.0,
                    zorder=3,
                )
                axis.plot(steps, lower, color="#17212B", linewidth=1.15, linestyle="--", alpha=0.98, zorder=4)
                axis.plot(steps, upper, color="#17212B", linewidth=1.15, linestyle="--", alpha=0.98, zorder=4)
                axis.plot(steps, center, color="#FFFFFF", linewidth=1.10, alpha=0.98, zorder=5)
        axis.set_title(SCENARIO_LABELS[scenario_type], fontsize=11.5, fontweight="bold", loc="left", pad=5)
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", color="#E5E9ED", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.set_ylabel("全体用户平均占比", fontsize=10)

    axes[-1].set_xlabel("仿真步", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.008), fontsize=9.4)
    fig.tight_layout(rect=[0.03, 0.055, 1, 0.96])
    _save_publication_figure(fig, output_path)
    plt.close(fig)


def _cumulative_metric_mean_std(
    raw_rows: list[dict[str, str]],
    value_key: str,
    filters: dict[str, str],
    scale: float = 1.0,
) -> tuple[float, float]:
    totals: dict[str, float] = defaultdict(float)
    for row in raw_rows:
        if _matches(row, filters):
            totals[str(row.get("run_index", ""))] += _float(row, value_key)
    values = [value / scale for value in totals.values()]
    return (float(np.mean(values)), _sample_std(values)) if values else (0.0, 0.0)


def _unit_interaction_mean_std(
    raw_rows: list[dict[str, str]],
    filters: dict[str, str],
) -> tuple[float, float]:
    totals: dict[str, float] = defaultdict(float)
    agent_counts: dict[str, float] = defaultdict(float)
    for row in raw_rows:
        if not _matches(row, filters):
            continue
        run_index = str(row.get("run_index", ""))
        totals[run_index] += _float(row, "derived_comment_count") + _float(row, "derived_reply_count")
        active_rate = _float(row, "active_rate")
        if active_rate > 0.0:
            agent_counts[run_index] = max(
                agent_counts[run_index],
                _float(row, "active_count") / active_rate,
            )
    values = [total / max(1.0, agent_counts[run_index]) for run_index, total in totals.items()]
    return (float(np.mean(values)), _sample_std(values)) if values else (0.0, 0.0)


def _area(rows: list[dict[str, str]], value_key: str) -> float:
    if not rows:
        return 0.0
    values = [_float(row, value_key) for row in rows]
    return float(sum(values) / len(values))


def _metric_value(row: dict[str, str], *keys: str) -> float:
    for key in keys:
        if key in row and row.get(key) not in ("", None):
            return _float(row, key)
    return 0.0


def _plot_lines(
    rows: Iterable[dict[str, str]],
    group_key: str,
    value_key: str,
    group_order: list[str],
    group_labels: dict[str, str],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    _setup_matplotlib()
    _ensure_dir(output_path.parent)
    grouped = _group_rows(rows, group_key)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for group in group_order:
        group_rows = grouped.get(group, [])
        if not group_rows:
            continue
        x_values, y_values = _series(group_rows, value_key)
        ax.plot(x_values, y_values, marker="o", linewidth=2.0, label=group_labels.get(group, group))
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("时间步")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_bar_from_values(values: list[float], labels: list[str], title: str, ylabel: str, output_path: Path) -> None:
    _setup_matplotlib()
    _ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bars = ax.bar(labels, values, color=["#7aa6c2", "#e5b567", "#c86b5b"], width=0.58)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.30)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _draw_exp3_grouped_bar(
    axis: plt.Axes,
    groups: list[str],
    series: list[tuple[str, list[float]]],
    error_series: list[list[float]] | None,
    ylabel: str,
    panel_label: str,
    panel_title: str,
) -> None:
    """Draw one scientific grouped-bar panel for the Experiment 3 composite figure."""

    x = np.arange(len(groups))
    series_count = max(1, len(series))
    width = min(0.17, 0.76 / series_count)
    series_colors = {
        PLATFORM_LABELS["baseline_platform"]: "#4C78A8",
        PLATFORM_LABELS["low_platform_amplification"]: "#F2A541",
        PLATFORM_LABELS["medium_platform_amplification"]: "#59A14F",
        PLATFORM_LABELS["high_platform_amplification"]: "#D9504E",
    }
    for index, (name, values) in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * width
        errors = error_series[index] if error_series and index < len(error_series) else None
        bars = axis.bar(
            x + offset,
            values,
            width=width,
            label=name,
            color=series_colors.get(name, "#7F8C8D"),
            edgecolor="white",
            linewidth=0.7,
            yerr=errors,
            error_kw={"ecolor": "#4D4D4D", "elinewidth": 0.75, "capsize": 2.2, "capthick": 0.75},
        )
        for item_index, bar in enumerate(bars):
            error = errors[item_index] if errors else 0.0
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + error,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=8.2,
            )
    axis.set_xticks(x)
    axis.set_xticklabels(groups, fontsize=9.5)
    axis.set_ylabel(ylabel, fontsize=10)
    axis.set_title(panel_title, fontsize=10.5, fontweight="bold", pad=8)
    axis.text(-0.16, 1.05, panel_label, transform=axis.transAxes, ha="left", va="bottom", fontsize=12, fontweight="bold")
    axis.grid(axis="y", color="#E5E9ED", linewidth=0.7)
    axis.set_axisbelow(True)


def _plot_heatmap(
    matrix: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    title: str,
    colorbar_label: str,
    output_path: Path,
) -> None:
    _setup_matplotlib()
    _ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    image = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_title(title, fontsize=14, fontweight="bold")
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(col_index, row_index, f"{matrix[row_index, col_index]:.3f}", ha="center", va="center", fontsize=9)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_experiment_1(outputs: dict[str, object], figures_dir: Path) -> dict[str, str]:
    _ensure_dir(figures_dir)
    time_series_path = Path(str(outputs["time_series_path"]))
    raw_rows, rows = _split_typed_rows(_read_csv(time_series_path))
    generated: dict[str, str] = {}

    figure_1 = figures_dir / "figure_1_exp1_group_negative_emotion.png"
    _plot_exp1_trajectory_figure(
        rows,
        raw_rows,
        mean_key="mean_group_negative_emotion",
        std_key="std_group_negative_emotion",
        raw_key="group_negative_emotion",
        ylabel="群体平均负面情绪 G(t)",
        output_path=figure_1,
    )
    generated["figure_1"] = str(figure_1)

    figure_2 = figures_dir / "figure_2_exp1_high_negative_ratio.png"
    _plot_exp1_trajectory_figure(
        rows,
        raw_rows,
        mean_key="mean_high_negative_ratio",
        std_key="std_high_negative_ratio",
        raw_key="high_negative_ratio",
        ylabel="高负面情绪节点占比 H(t)",
        output_path=figure_2,
    )
    generated["figure_2"] = str(figure_2)

    figure_3 = figures_dir / "figure_3_exp1_dynamic_emotion_composition.png"
    _plot_exp1_dynamic_emotion_composition(raw_rows, figure_3, rows)
    generated["figure_3"] = str(figure_3)

    for obsolete_name in (
        "figure_1_exp1_emotion_trajectories.png",
        "figure_1_exp1_emotion_trajectories.svg",
        "figure_1_exp1_emotion_trajectories.pdf",
        "figure_1_exp1_group_negative_emotion.svg",
        "figure_1_exp1_group_negative_emotion.pdf",
        "figure_2_exp1_high_negative_ratio.svg",
        "figure_2_exp1_high_negative_ratio.pdf",
        "figure_3_exp1_dynamic_emotion_composition.svg",
        "figure_3_exp1_dynamic_emotion_composition.pdf",
        "figure_3_exp1_interaction_action_counts.png",
        "figure_3_exp1_interaction_action_counts.svg",
        "figure_3_exp1_interaction_action_counts.pdf",
        "figure_4_exp1_emotion_response_analysis.png",
        "figure_4_exp1_emotion_response_analysis.svg",
        "figure_4_exp1_emotion_response_analysis.pdf",
    ):
        (figures_dir / obsolete_name).unlink(missing_ok=True)

    return generated


def _mean_and_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), _sample_std(values)


def _result_metric_mean_std(
    raw_rows: list[dict[str, str]],
    scenario_type: str,
    platform_level: str,
    metric_key: str,
) -> tuple[float, float]:
    values = [
        _float(row, metric_key)
        for row in raw_rows
        if row.get("scenario_type") == scenario_type and row.get("platform_level") == platform_level
    ]
    return _mean_and_std(values)


def _paired_delta_mean_std(
    raw_rows: list[dict[str, str]],
    scenario_type: str,
    platform_level: str,
    baseline_level: str,
    metric_key: str,
) -> tuple[float, float]:
    def collect(level: str) -> dict[str, float]:
        return {
            str(row.get("run_index", "")): _float(row, metric_key)
            for row in raw_rows
            if row.get("scenario_type") == scenario_type and row.get("platform_level") == level
        }

    baseline = collect(baseline_level)
    target = collect(platform_level)
    values = [target[index] - baseline[index] for index in sorted(set(target) & set(baseline))]
    return _mean_and_std(values)


def _plot_experiment_3_summary(
    rows: list[dict[str, str]],
    time_rows: list[dict[str, str]],
    raw_summary_rows: list[dict[str, str]],
    raw_time_rows: list[dict[str, str]],
    figures_dir: Path,
) -> dict[str, str]:
    _ensure_dir(figures_dir)
    available_platforms = {row.get("platform_level", "") for row in rows}
    if not set(PLATFORM_ORDER).issubset(available_platforms):
        return {}

    scenario_labels = [SCENARIO_LABELS[item] for item in SCENARIO_ORDER]
    peak_series: list[tuple[str, list[float]]] = []
    peak_errors: list[list[float]] = []
    for platform in PLATFORM_ORDER:
        values: list[float] = []
        errors: list[float] = []
        for scenario in SCENARIO_ORDER:
            mean_value, std_value = _result_metric_mean_std(raw_summary_rows, scenario, platform, "peak")
            values.append(mean_value)
            errors.append(std_value)
        peak_series.append((PLATFORM_LABELS[platform], values))
        peak_errors.append(errors)
    delta_order = PLATFORM_ORDER[1:]
    delta_series: list[tuple[str, list[float]]] = []
    delta_errors: list[list[float]] = []
    for platform in delta_order:
        values: list[float] = []
        errors: list[float] = []
        for scenario in SCENARIO_ORDER:
            mean_value, std_value = _paired_delta_mean_std(
                raw_summary_rows,
                scenario,
                platform,
                PLATFORM_ORDER[0],
                "peak",
            )
            values.append(mean_value)
            errors.append(std_value)
        delta_series.append((PLATFORM_LABELS[platform], values))
        delta_errors.append(errors)
    average_rise_speed_series: list[tuple[str, list[float]]] = []
    average_rise_speed_errors: list[list[float]] = []
    for platform in PLATFORM_ORDER:
        values: list[float] = []
        errors: list[float] = []
        for scenario in SCENARIO_ORDER:
            mean_value, std_value = _result_metric_mean_std(
                raw_summary_rows,
                scenario,
                platform,
                "peak_average_rise_speed",
            )
            values.append(mean_value)
            errors.append(std_value)
        average_rise_speed_series.append((PLATFORM_LABELS[platform], values))
        average_rise_speed_errors.append(errors)
    figure_8 = figures_dir / "figure_8_exp3_platform_amplification_comparison.png"
    _setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.85))
    panel_specs = (
        ("A", peak_series, peak_errors, "群体负面情绪峰值", "群体负面情绪峰值"),
        ("B", delta_series, delta_errors, "相对基准平台的峰值增量", "相对基准平台的峰值增量"),
        ("C", average_rise_speed_series, average_rise_speed_errors, "平均上升速度", "平均上升速度"),
    )
    for axis, (panel_label, series, errors, ylabel, panel_title) in zip(axes, panel_specs):
        _draw_exp3_grouped_bar(axis, scenario_labels, series, errors, ylabel, panel_label, panel_title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.01), frameon=False, fontsize=9.5)
    fig.tight_layout(rect=[0.02, 0.10, 1, 1])
    _save_publication_figure(fig, figure_8)
    plt.close(fig)

    for obsolete_name in (
        "figure_8_exp3_peak_grouped_bar.png",
        "figure_9_exp3_delta_peak_grouped_bar.png",
        "figure_10_exp3_average_rise_speed_grouped_bar.png",
    ):
        (figures_dir / obsolete_name).unlink(missing_ok=True)

    return {
        "figure_8_exp3_platform_amplification_comparison": str(figure_8),
    }


def plot_experiment_3(outputs: dict[str, object], figures_dir: Path) -> dict[str, str]:
    raw_summary_rows, rows = _split_typed_rows(_read_csv(str(outputs["summary_path"])))
    raw_time_rows, time_rows = _split_typed_rows(_read_csv(str(outputs.get("time_series_path", ""))))
    return _plot_experiment_3_summary(rows, time_rows, raw_summary_rows, raw_time_rows, figures_dir)


def plot_experiment_5_agent_scale(outputs: dict[str, object], figures_dir: Path) -> dict[str, str]:
    """绘制群体规模稳健性检验的轨迹与终局指标。"""

    _setup_matplotlib()
    _ensure_dir(figures_dir)
    raw_summary_rows, summary_rows = _split_typed_rows(_read_csv(str(outputs["summary_path"])))
    raw_time_rows, time_rows = _split_typed_rows(_read_csv(str(outputs["time_series_path"])))
    del raw_summary_rows  # 图表直接使用汇总统计量与逐轮原始序列。
    if not summary_rows or not time_rows:
        return {}

    scales = sorted({_float(row, "num_agents") for row in summary_rows})
    scales = [int(value) for value in scales]
    figure_15 = figures_dir / "figure_15_exp5_agent_scale_trajectories.png"
    fig, axes = plt.subplots(1, len(scales), figsize=(4.55 * len(scales), 4.25), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, num_agents in zip(axes, scales):
        for scenario_type in SCENARIO_ORDER:
            group_label = f"{scenario_type}__n{num_agents}"
            group_rows = [row for row in time_rows if row.get("group_label") == group_label]
            if not group_rows:
                continue
            x_values, mean_values, std_values = _series_with_std(
                group_rows,
                "mean_group_negative_emotion",
                "std_group_negative_emotion",
                raw_time_rows,
                "group_negative_emotion",
                {"group_label": group_label},
            )
            color = EXP1_SCENARIO_COLORS[scenario_type]
            style = EXP1_SCENARIO_STYLES[scenario_type]
            axis.fill_between(
                x_values,
                np.clip(mean_values - std_values, 0.0, 1.0),
                np.clip(mean_values + std_values, 0.0, 1.0),
                color=color,
                alpha=0.12,
                linewidth=0.0,
            )
            axis.plot(
                x_values,
                mean_values,
                color=color,
                linewidth=1.7,
                label=SCENARIO_LABELS[scenario_type],
                **style,
            )
        axis.set_title(f"{num_agents}人", fontsize=12, fontweight="bold")
        axis.set_xlabel("仿真步")
        axis.grid(axis="y", color="#E5E9ED", linewidth=0.7)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("群体平均负面情绪 G(t)")
    axes[0].set_ylim(0.0, 1.0)
    axes[-1].legend(loc="best", fontsize=9)
    fig.tight_layout()
    _save_publication_figure(fig, figure_15)
    plt.close(fig)

    figure_16 = figures_dir / "figure_16_exp5_agent_scale_summary_metrics.png"
    metric_specs = [
        ("mean_peak", "std_peak", "群体负面情绪峰值"),
        ("mean_final_group_negative_emotion", "std_final_group_negative_emotion", "末轮群体负面情绪"),
        ("mean_final_high_negative_ratio", "std_final_high_negative_ratio", "末轮高负面情绪节点占比"),
    ]
    scenario_colors = [EXP1_SCENARIO_COLORS[scenario] for scenario in SCENARIO_ORDER]
    x_values = np.arange(len(scales), dtype=float)
    width = 0.22
    fig, axes = plt.subplots(1, len(metric_specs), figsize=(14.2, 4.35))
    for axis, (metric_key, std_key, title) in zip(axes, metric_specs):
        for index, scenario_type in enumerate(SCENARIO_ORDER):
            values: list[float] = []
            errors: list[float] = []
            for num_agents in scales:
                row = next(
                    (
                        item
                        for item in summary_rows
                        if item.get("scenario_type") == scenario_type
                        and int(_float(item, "num_agents")) == num_agents
                    ),
                    {},
                )
                values.append(_metric_value(row, metric_key))
                errors.append(_metric_value(row, std_key))
            offset = (index - 1) * width
            axis.bar(
                x_values + offset,
                values,
                width=width,
                color=scenario_colors[index],
                label=SCENARIO_LABELS[scenario_type],
                yerr=errors,
                error_kw={"ecolor": "#4D4D4D", "elinewidth": 0.8, "capsize": 2.2, "capthick": 0.8},
            )
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.set_xticks(x_values)
        axis.set_xticklabels([str(value) for value in scales])
        axis.set_xlabel("智能体数量")
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", color="#E5E9ED", linewidth=0.7)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("均值 ± 标准差")
    axes[-1].legend(loc="best", fontsize=8.5)
    fig.tight_layout()
    _save_publication_figure(fig, figure_16)
    plt.close(fig)
    return {
        "figure_15_exp5_agent_scale_trajectories": str(figure_15),
        "figure_16_exp5_agent_scale_summary_metrics": str(figure_16),
    }


def plot_experiment_6_model_robustness(outputs: dict[str, object], figures_dir: Path) -> dict[str, str]:
    """绘制不同运行时 LLM 下的过程轨迹与核心结果对比。"""

    _setup_matplotlib()
    _ensure_dir(figures_dir)
    _, summary_rows = _split_typed_rows(_read_csv(str(outputs["summary_path"])))
    raw_time_rows, time_rows = _split_typed_rows(_read_csv(str(outputs["time_series_path"])))
    if not summary_rows or not time_rows:
        return {}

    profile_ids = list(dict.fromkeys(row.get("profile_id", "") for row in summary_rows if row.get("profile_id")))
    profile_labels = {
        profile_id: next(
            (
                row.get("model", "").strip()
                for row in summary_rows
                if row.get("profile_id") == profile_id and row.get("model", "").strip()
            ),
            profile_id,
        )
        for profile_id in profile_ids
    }

    figure_17 = figures_dir / "figure_17_exp6_model_robustness_trajectories.png"
    fig, axes = plt.subplots(1, len(profile_ids), figsize=(4.85 * len(profile_ids), 4.25), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, profile_id in zip(axes, profile_ids):
        for scenario_type in SCENARIO_ORDER:
            group_label = f"{profile_id}__{scenario_type}"
            group_rows = [row for row in time_rows if row.get("group_label") == group_label]
            if not group_rows:
                continue
            x_values, mean_values, std_values = _series_with_std(
                group_rows,
                "mean_group_negative_emotion",
                "std_group_negative_emotion",
                raw_time_rows,
                "group_negative_emotion",
                {"group_label": group_label},
            )
            color = EXP1_SCENARIO_COLORS[scenario_type]
            style = EXP1_SCENARIO_STYLES[scenario_type]
            axis.fill_between(
                x_values,
                np.clip(mean_values - std_values, 0.0, 1.0),
                np.clip(mean_values + std_values, 0.0, 1.0),
                color=color,
                alpha=0.12,
                linewidth=0.0,
            )
            axis.plot(
                x_values,
                mean_values,
                color=color,
                linewidth=1.7,
                label=SCENARIO_LABELS[scenario_type],
                **style,
            )
        axis.set_title(profile_labels[profile_id], fontsize=11, fontweight="bold")
        axis.set_xlabel("仿真步")
        axis.grid(axis="y", color="#E5E9ED", linewidth=0.7)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("群体平均负面情绪 G(t)")
    axes[0].set_ylim(0.0, 1.0)
    axes[-1].legend(loc="best", fontsize=9)
    fig.tight_layout()
    _save_publication_figure(fig, figure_17)
    plt.close(fig)

    figure_18 = figures_dir / "figure_18_exp6_model_robustness_summary_metrics.png"
    metric_specs = [
        ("mean_peak", "std_peak", "群体负面情绪峰值"),
        ("mean_final_group_negative_emotion", "std_final_group_negative_emotion", "末轮群体负面情绪"),
        ("mean_final_high_negative_ratio", "std_final_high_negative_ratio", "末轮高负面情绪节点占比"),
    ]
    x_values = np.arange(len(profile_ids), dtype=float)
    width = 0.22
    fig, axes = plt.subplots(1, len(metric_specs), figsize=(14.2, 4.35))
    for axis, (metric_key, std_key, title) in zip(axes, metric_specs):
        for index, scenario_type in enumerate(SCENARIO_ORDER):
            values: list[float] = []
            errors: list[float] = []
            for profile_id in profile_ids:
                row = next(
                    (
                        item
                        for item in summary_rows
                        if item.get("profile_id") == profile_id
                        and item.get("scenario_type") == scenario_type
                    ),
                    {},
                )
                values.append(_metric_value(row, metric_key))
                errors.append(_metric_value(row, std_key))
            offset = (index - 1) * width
            axis.bar(
                x_values + offset,
                values,
                width=width,
                color=EXP1_SCENARIO_COLORS[scenario_type],
                label=SCENARIO_LABELS[scenario_type],
                yerr=errors,
                error_kw={"ecolor": "#4D4D4D", "elinewidth": 0.8, "capsize": 2.2, "capthick": 0.8},
            )
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.set_xticks(x_values)
        axis.set_xticklabels([profile_labels[profile_id] for profile_id in profile_ids])
        axis.set_xlabel("运行时LLM")
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", color="#E5E9ED", linewidth=0.7)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("均值 ± 标准差")
    axes[-1].legend(loc="best", fontsize=8.5)
    fig.tight_layout()
    _save_publication_figure(fig, figure_18)
    plt.close(fig)
    return {
        "figure_17_exp6_model_robustness_trajectories": str(figure_17),
        "figure_18_exp6_model_robustness_summary_metrics": str(figure_18),
    }


def _draw_exp4_peak_response(
    axis: plt.Axes,
    rows: list[dict[str, str]],
    analysis_part: str,
    parameter_key: str,
    parameter_label: str,
) -> None:
    """Draw the peak-response curve with one sample-SD error bar per parameter value."""

    response_rows = [
        row
        for row in rows
        if row.get("analysis_part") == analysis_part and row.get(parameter_key) not in ("", None)
    ]
    response_rows.sort(key=lambda row: _float(row, parameter_key))
    x_values = np.asarray([_float(row, parameter_key) for row in response_rows], dtype=float)
    mean_values = np.asarray([_metric_value(row, "mean_peak") for row in response_rows], dtype=float)
    std_values = np.asarray([_metric_value(row, "std_peak") for row in response_rows], dtype=float)
    if not len(x_values):
        axis.axis("off")
        return

    axis.errorbar(
        x_values,
        mean_values,
        yerr=std_values,
        color="#3B78A8",
        marker="o",
        markersize=3.8,
        linewidth=1.7,
        capsize=3.4,
        capthick=0.8,
        elinewidth=0.8,
    )
    axis.set_xticks(x_values)
    axis.set_xticklabels([f"{value:g}" for value in x_values])
    axis.set_xlabel(parameter_label)
    axis.set_ylabel("群体负面情绪峰值")
    axis.set_ylim(bottom=0.0)
    axis.grid(axis="y", color="#E5E9ED", linewidth=0.7)
    axis.set_axisbelow(True)


def _plot_exp4_peak_response_figure(
    rows: list[dict[str, str]],
    analysis_part: str,
    parameter_key: str,
    parameter_label: str,
    output_path: Path,
) -> None:
    _setup_matplotlib()
    _ensure_dir(output_path.parent)
    fig, axis = plt.subplots(figsize=(7.8, 4.7))
    _draw_exp4_peak_response(
        axis,
        rows,
        analysis_part,
        parameter_key,
        parameter_label,
    )
    fig.tight_layout()
    _save_publication_figure(fig, output_path)
    plt.close(fig)


def _plot_exp4_beta_gamma_peak_heatmaps(rows: list[dict[str, str]], output_path: Path) -> None:
    """Show the beta-gamma peak response surface as one publication-ready heatmap."""

    emotion_rows = [row for row in rows if row.get("analysis_part") == "emotion_mechanism"]
    beta_values = sorted({_float(row, "beta_stimulus") for row in emotion_rows})
    gamma_values = sorted({_float(row, "gamma_feedback") for row in emotion_rows})
    if not beta_values or not gamma_values:
        return

    lookup = {(_float(row, "gamma_feedback"), _float(row, "beta_stimulus")): row for row in emotion_rows}
    mean_matrix = np.zeros((len(gamma_values), len(beta_values)), dtype=float)
    for row_index, gamma_value in enumerate(gamma_values):
        for column_index, beta_value in enumerate(beta_values):
            row = lookup.get((gamma_value, beta_value), {})
            mean_matrix[row_index, column_index] = _metric_value(row, "mean_peak")

    _setup_matplotlib()
    _ensure_dir(output_path.parent)
    fig, axis = plt.subplots(figsize=(6.25, 4.85))
    image = axis.imshow(mean_matrix, cmap="YlOrRd", aspect="auto")
    axis.set_xticks(range(len(beta_values)))
    axis.set_xticklabels([f"{value:.2f}" for value in beta_values])
    axis.set_yticks(range(len(gamma_values)))
    axis.set_yticklabels([f"{value:.2f}" for value in gamma_values])
    axis.set_xlabel("信息刺激权重 β")
    axis.set_ylabel("互动反馈权重 γ")
    midpoint = (float(np.min(mean_matrix)) + float(np.max(mean_matrix))) / 2.0
    for row_index in range(mean_matrix.shape[0]):
        for column_index in range(mean_matrix.shape[1]):
            value = mean_matrix[row_index, column_index]
            text_color = "#FFFFFF" if value >= midpoint else "#2B2B2B"
            axis.text(column_index, row_index, f"{value:.3f}", ha="center", va="center", fontsize=8.5, color=text_color)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.05, pad=0.04)
    colorbar.set_label("群体负面情绪峰值", fontsize=9.5)
    fig.tight_layout()
    _save_publication_figure(fig, output_path)
    plt.close(fig)


def _plot_experiment_4_sensitivity_figures(
    rows: list[dict[str, str]],
    time_rows: list[dict[str, str]],
    raw_time_rows: list[dict[str, str]],
    figures_dir: Path,
) -> dict[str, str]:
    _ensure_dir(figures_dir)
    for old_name in (
        "figure_4a_exp4_beta_gamma_final_g_heatmap.png",
        "figure_4a_exp4_beta_gamma_peak_heatmap.png",
        "figure_4c_exp4_memory_weight_sensitivity.png",
        "figure_4d_exp4_emotion_decay_sensitivity.png",
        "figure_7_exp4_beta_gamma_peak_heatmap.png",
    ):
        (figures_dir / old_name).unlink(missing_ok=True)

    generated: dict[str, str] = {}
    emotion_rows = [row for row in rows if row.get("analysis_part") == "emotion_mechanism"]
    beta_values = sorted({_float(row, "beta_stimulus") for row in emotion_rows})
    gamma_values = sorted({_float(row, "gamma_feedback") for row in emotion_rows})
    if beta_values and gamma_values:
        lookup = {(_float(row, "gamma_feedback"), _float(row, "beta_stimulus")): row for row in emotion_rows}
        matrix = np.zeros((len(gamma_values), len(beta_values)), dtype=float)
        for row_index, gamma in enumerate(gamma_values):
            for col_index, beta in enumerate(beta_values):
                matrix[row_index, col_index] = _metric_value(lookup.get((gamma, beta), {}), "mean_peak")
        figure_11 = figures_dir / "figure_11_exp4_beta_gamma_peak_heatmap.png"
        _plot_heatmap(
            matrix,
            [f"{value:.2f}" for value in beta_values],
            [f"{value:.2f}" for value in gamma_values],
            "图11 β与γ组合下群体负面情绪峰值热力图",
            "群体负面情绪峰值",
            figure_11,
        )
        generated["figure_11"] = str(figure_11)

    decay_rows = [
        row
        for row in rows
        if row.get("analysis_part") == "emotion_decay"
        and row.get("emotion_decay") not in ("", None)
    ]
    if decay_rows:
        figure_12 = figures_dir / "figure_12_exp4_emotion_decay_peak_sensitivity.png"
        _plot_exp4_process_sensitivity_figure(
            time_rows,
            raw_time_rows,
            "emotion_decay",
            "emotion_decay",
            "情绪衰减率",
            0.40,
            "图12 情绪衰减率不同取值下群体负面情绪全过程变化",
            figure_12,
        )
        generated["figure_12"] = str(figure_12)

    memory_rows = [
        row
        for row in rows
        if row.get("analysis_part") == "memory_weight"
        and row.get("memory_emotion_weight") not in ("", None)
    ]
    if memory_rows:
        figure_13 = figures_dir / "figure_13_exp4_memory_weight_peak_sensitivity.png"
        _plot_exp4_process_sensitivity_figure(
            time_rows,
            raw_time_rows,
            "memory_weight",
            "memory_emotion_weight",
            "情绪记忆权重",
            0.24,
            "图13 情绪记忆权重不同取值下群体负面情绪全过程变化",
            figure_13,
        )
        generated["figure_13"] = str(figure_13)

    content_age_rows = [
        row
        for row in rows
        if row.get("analysis_part") == "content_age"
        and row.get("max_content_age") not in ("", None)
    ]
    network_density_rows = [
        row
        for row in rows
        if row.get("analysis_part") == "network_density"
        and row.get("graph_edge_probability") not in ("", None)
    ]
    if content_age_rows or network_density_rows:
        figure_14 = figures_dir / "figure_14_exp4_boundary_condition_sensitivity.png"
        _setup_matplotlib()
        _ensure_dir(figures_dir)
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
        if content_age_rows:
            _draw_exp4_process_trajectories(
                axes[0], time_rows, raw_time_rows, "content_age", "max_content_age", "最大内容寿命", 5.0
            )
            axes[0].set_title("内容最大寿命敏感性", fontsize=12, fontweight="bold")
        else:
            axes[0].axis("off")

        if network_density_rows:
            _draw_exp4_process_trajectories(
                axes[1], time_rows, raw_time_rows, "network_density", "graph_edge_probability", "网络边密度", 0.10
            )
            axes[1].set_title("网络边密度敏感性", fontsize=12, fontweight="bold")
        else:
            axes[1].axis("off")

        fig.suptitle("图14 传播边界条件敏感性分析", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(figure_14, bbox_inches="tight")
        plt.close(fig)
        generated["figure_14"] = str(figure_14)
    return generated


def _paired_peak_change_series(
    run_rows: list[dict[str, str]],
    analysis_part: str,
    parameter_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return mean and sample-SD peak changes paired to the central parameter value."""

    relevant_rows = [
        row
        for row in run_rows
        if row.get("analysis_part") == analysis_part
        and row.get(parameter_key) not in ("", None)
        and row.get("peak") not in ("", None)
    ]
    values = sorted({_float(row, parameter_key) for row in relevant_rows})
    if len(values) < 3 or len(values) % 2 == 0:
        return None

    baseline_value = values[len(values) // 2]
    baseline_by_run = {
        int(_float(row, "run_index")): _float(row, "peak")
        for row in relevant_rows
        if np.isclose(_float(row, parameter_key), baseline_value)
    }
    if not baseline_by_run:
        return None

    step_size = min(right - left for left, right in zip(values, values[1:]))
    positions: list[float] = []
    mean_changes: list[float] = []
    std_changes: list[float] = []
    for value in values:
        paired_changes = [
            _float(row, "peak") - baseline_by_run[int(_float(row, "run_index"))]
            for row in relevant_rows
            if np.isclose(_float(row, parameter_key), value)
            and int(_float(row, "run_index")) in baseline_by_run
        ]
        if not paired_changes:
            return None
        positions.append(round((value - baseline_value) / step_size))
        mean_changes.append(float(np.mean(paired_changes)))
        std_changes.append(_sample_std(paired_changes))

    return (
        np.asarray(positions, dtype=float),
        np.asarray(mean_changes, dtype=float),
        np.asarray(std_changes, dtype=float),
    )


def _plot_exp4_emotion_maintenance_peak_sensitivity(
    run_rows: list[dict[str, str]],
    output_path: Path,
) -> bool:
    """Compare decay and memory effects on peak emotion with one shared axis."""

    series_specs = (
        ("emotion_decay", "emotion_decay", "情绪衰减率", "#3B78A8", "o", "-"),
        ("memory_weight", "memory_emotion_weight", "情绪记忆权重", "#D9822B", "s", "--"),
    )
    resolved_series = []
    for analysis_part, parameter_key, label, color, marker, linestyle in series_specs:
        series = _paired_peak_change_series(run_rows, analysis_part, parameter_key)
        if series is None:
            return False
        resolved_series.append((series, label, color, marker, linestyle))

    _setup_matplotlib()
    _ensure_dir(output_path.parent)
    fig, axis = plt.subplots(figsize=(7.8, 4.7))
    values_for_limits: list[float] = []
    for (positions, mean_changes, std_changes), label, color, marker, linestyle in resolved_series:
        axis.errorbar(
            positions,
            mean_changes,
            yerr=std_changes,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.8,
            markersize=4.4,
            markeredgecolor="#3E3E3E",
            markeredgewidth=0.55,
            capsize=3.1,
            capthick=0.8,
            elinewidth=0.8,
        )
        values_for_limits.extend((mean_changes - std_changes).tolist())
        values_for_limits.extend((mean_changes + std_changes).tolist())

    lower, upper = min(values_for_limits), max(values_for_limits)
    padding = max((upper - lower) * 0.12, 0.015)
    axis.set_xlim(-2.2, 2.2)
    axis.set_ylim(lower - padding, upper + padding)
    axis.set_xticks([-2, -1, 0, 1, 2])
    axis.set_xlabel("相对基准的参数调节步长（每步 0.02）")
    axis.set_ylabel("群体负面情绪峰值变化 ΔG_peak")
    axis.grid(axis="y", color="#E5E9ED", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.legend(loc="best", fontsize=9)
    fig.tight_layout()
    _save_publication_figure(fig, output_path)
    plt.close(fig)
    return True


def _plot_experiment_4_peak_sensitivity_figures(
    rows: list[dict[str, str]],
    figures_dir: Path,
    run_rows: list[dict[str, str]],
) -> dict[str, str]:
    """Plot Experiment 4 as parameter-response peak comparisons only."""

    _ensure_dir(figures_dir)
    generated: dict[str, str] = {}

    figure_11 = figures_dir / "figure_11_exp4_beta_gamma_peak_heatmap.png"
    _plot_exp4_beta_gamma_peak_heatmaps(rows, figure_11)
    if figure_11.exists():
        generated["figure_11"] = str(figure_11)

    for obsolete_name in (
        "figure_12_exp4_emotion_decay_peak_sensitivity.png",
        "figure_13_exp4_memory_weight_peak_sensitivity.png",
    ):
        (figures_dir / obsolete_name).unlink(missing_ok=True)

    figure_12 = figures_dir / "figure_12_exp4_emotion_maintenance_peak_sensitivity.png"
    if _plot_exp4_emotion_maintenance_peak_sensitivity(run_rows, figure_12):
        generated["figure_12"] = str(figure_12)

    boundary_specs = (
        ("content_age", "max_content_age", "内容最大寿命"),
        ("network_density", "graph_edge_probability", "网络边密度"),
    )
    if any(any(row.get("analysis_part") == part for row in rows) for part, *_ in boundary_specs):
        (figures_dir / "figure_14_exp4_boundary_condition_sensitivity.png").unlink(missing_ok=True)
        figure_13 = figures_dir / "figure_13_exp4_boundary_condition_sensitivity.png"
        _setup_matplotlib()
        fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7), sharey=True)
        for panel_label, axis, (analysis_part, parameter_key, parameter_label) in zip(("A", "B"), axes, boundary_specs):
            if not any(row.get("analysis_part") == analysis_part for row in rows):
                axis.axis("off")
                continue
            _draw_exp4_peak_response(
                axis,
                rows,
                analysis_part,
                parameter_key,
                parameter_label,
            )
            axis.text(0.02, 0.98, panel_label, transform=axis.transAxes, ha="left", va="top", fontsize=12, fontweight="bold")
        # Both boundary-condition response series stay above 0.40; crop unused space.
        axes[0].set_ylim(bottom=0.40)
        fig.tight_layout()
        _save_publication_figure(fig, figure_13)
        plt.close(fig)
        generated["figure_13"] = str(figure_13)

    return generated


def plot_experiment_4(outputs: dict[str, object], figures_dir: Path) -> dict[str, str]:
    _ensure_dir(figures_dir)
    run_rows, rows = _split_typed_rows(_read_csv(str(outputs["summary_path"])))
    return _plot_experiment_4_peak_sensitivity_figures(rows, figures_dir, run_rows)


def _experiment_7_rows_by_part(rows: list[dict[str, str]], analysis_part: str) -> list[dict[str, str]]:
    return sorted(
        [row for row in rows if row.get("analysis_part") == analysis_part],
        key=lambda row: _float(row, "parameter_value"),
    )


def _plot_experiment_7_trajectory(
    summary_rows: list[dict[str, str]],
    time_rows: list[dict[str, str]],
    analysis_part: str,
    parameter_label: str,
    output_path: Path,
) -> bool:
    relevant_summary = _experiment_7_rows_by_part(summary_rows, analysis_part)
    if not relevant_summary:
        return False

    _setup_matplotlib()
    figure, axis = plt.subplots(figsize=(7.6, 4.55))
    colors = plt.cm.viridis(np.linspace(0.15, 0.88, len(relevant_summary)))
    for color, summary_row in zip(colors, relevant_summary):
        group_label = str(summary_row.get("group_label", ""))
        group_rows = sorted(
            [row for row in time_rows if row.get("group_label") == group_label],
            key=lambda row: _float(row, "step"),
        )
        if not group_rows:
            continue
        x_values = np.asarray([_float(row, "step") for row in group_rows], dtype=float)
        y_values = np.asarray([_metric_value(row, "mean_group_negative_emotion") for row in group_rows], dtype=float)
        std_values = np.asarray([_metric_value(row, "std_group_negative_emotion") for row in group_rows], dtype=float)
        parameter_value = _float(summary_row, "parameter_value")
        is_baseline = str(summary_row.get("is_baseline", "")).lower() == "true"
        axis.fill_between(
            x_values,
            np.maximum(0.0, y_values - std_values),
            np.minimum(1.0, y_values + std_values),
            color=color,
            alpha=0.12,
            linewidth=0.0,
        )
        axis.plot(
            x_values,
            y_values,
            color=color,
            linewidth=2.2 if is_baseline else 1.45,
            linestyle="-" if is_baseline else "--",
            label=f"{parameter_value:.2f}" + ("（基准）" if is_baseline else ""),
        )
    axis.set_xlabel("仿真步", fontsize=10)
    axis.set_ylabel("群体平均负面情绪", fontsize=10)
    axis.set_title(f"{parameter_label}不同水平下群体负面情绪轨迹", fontsize=12, fontweight="bold")
    axis.grid(axis="y", color="#E5E9ED", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.legend(title=parameter_label, ncol=3, fontsize=8, title_fontsize=8, loc="upper left")
    figure.tight_layout()
    _save_publication_figure(figure, output_path)
    plt.close(figure)
    return True


def _plot_experiment_7_risk_peak(
    rows: list[dict[str, str]],
    output_path: Path,
) -> bool:
    relevant_rows = _experiment_7_rows_by_part(rows, "risk_boost")
    if not relevant_rows:
        return False
    _setup_matplotlib()
    figure, axis = plt.subplots(figsize=(6.6, 4.25))
    x_values = np.asarray([_float(row, "parameter_value") for row in relevant_rows])
    y_values = np.asarray([_metric_value(row, "mean_peak") for row in relevant_rows])
    errors = np.asarray([_metric_value(row, "std_peak") for row in relevant_rows])
    axis.errorbar(
        x_values,
        y_values,
        yerr=errors,
        color="#B54B46",
        marker="o",
        markersize=5.2,
        markeredgecolor="#333333",
        markeredgewidth=0.5,
        linewidth=1.8,
        capsize=3.2,
        capthick=0.8,
    )
    axis.fill_between(x_values, y_values - errors, y_values + errors, color="#E6A19D", alpha=0.20, linewidth=0.0)
    axis.set_xlabel("风险强化参数", fontsize=10)
    axis.set_ylabel("群体负面情绪峰值", fontsize=10)
    axis.set_xticks(x_values)
    axis.grid(axis="y", color="#E5E9ED", linewidth=0.7)
    axis.set_axisbelow(True)
    figure.tight_layout()
    _save_publication_figure(figure, output_path)
    plt.close(figure)
    return True


def _plot_experiment_7_memory_retention(
    rows: list[dict[str, str]],
    output_path: Path,
) -> bool:
    relevant_rows = _experiment_7_rows_by_part(rows, "memory_boost")
    if not relevant_rows:
        return False
    _setup_matplotlib()
    figure, axis = plt.subplots(figsize=(6.6, 4.25))
    x_values = np.arange(len(relevant_rows), dtype=float)
    values = np.asarray([_metric_value(row, "mean_emotion_retention_ratio") for row in relevant_rows])
    errors = np.asarray([_metric_value(row, "std_emotion_retention_ratio") for row in relevant_rows])
    colors = plt.cm.Purples(np.linspace(0.45, 0.86, len(relevant_rows)))
    bars = axis.bar(
        x_values,
        values,
        yerr=errors,
        color=colors,
        edgecolor="#FFFFFF",
        linewidth=0.8,
        error_kw={"ecolor": "#4C4C4C", "elinewidth": 0.8, "capsize": 3.0, "capthick": 0.8},
    )
    for bar, row, value, error in zip(bars, relevant_rows, values, errors):
        if str(row.get("is_baseline", "")).lower() == "true":
            bar.set_edgecolor("#2F2F2F")
            bar.set_linewidth(1.15)
        axis.text(bar.get_x() + bar.get_width() / 2, value + error + 0.008, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    axis.set_xticks(x_values)
    axis.set_xticklabels([f"{_float(row, 'parameter_value'):.2f}" for row in relevant_rows])
    axis.set_xlabel("记忆强化参数", fontsize=10)
    axis.set_ylabel("情绪保留程度（末轮/峰值）", fontsize=10)
    axis.set_ylim(bottom=max(0.0, float(np.min(values - errors)) - 0.05), top=min(1.04, float(np.max(values + errors)) + 0.06))
    axis.grid(axis="y", color="#E5E9ED", linewidth=0.7)
    axis.set_axisbelow(True)
    figure.tight_layout()
    _save_publication_figure(figure, output_path)
    plt.close(figure)
    return True


def _plot_experiment_7_share_interaction(
    rows: list[dict[str, str]],
    output_path: Path,
) -> bool:
    relevant_rows = _experiment_7_rows_by_part(rows, "share_boost")
    if not relevant_rows:
        return False
    _setup_matplotlib()
    figure, axis = plt.subplots(figsize=(6.6, 4.25))
    x_values = np.asarray([_float(row, "parameter_value") for row in relevant_rows])
    y_values = np.asarray([_metric_value(row, "mean_unit_interaction_intensity") for row in relevant_rows])
    errors = np.asarray([_metric_value(row, "std_unit_interaction_intensity") for row in relevant_rows])
    baseline_mask = np.asarray([str(row.get("is_baseline", "")).lower() == "true" for row in relevant_rows])
    axis.errorbar(
        x_values,
        y_values,
        yerr=errors,
        fmt="none",
        ecolor="#4E7C4F",
        elinewidth=0.9,
        capsize=3.2,
        capthick=0.9,
        zorder=2,
    )
    axis.scatter(x_values[~baseline_mask], y_values[~baseline_mask], s=48, color="#75A877", edgecolor="#365B38", linewidth=0.6, zorder=3)
    axis.scatter(x_values[baseline_mask], y_values[baseline_mask], s=74, color="#75A877", edgecolor="#202020", linewidth=1.0, zorder=4)
    axis.plot(x_values, y_values, color="#75A877", linewidth=1.25, alpha=0.72, zorder=1)
    axis.set_xlabel("互动强化参数", fontsize=10)
    axis.set_ylabel("单位用户互动强度", fontsize=10)
    axis.set_xticks(x_values)
    axis.grid(axis="y", color="#E5E9ED", linewidth=0.7)
    axis.set_axisbelow(True)
    figure.tight_layout()
    _save_publication_figure(figure, output_path)
    plt.close(figure)
    return True


def plot_experiment_7_strategy_parameter_sensitivity(
    outputs: dict[str, object],
    figures_dir: Path,
) -> dict[str, str]:
    """Render three trajectory and three response figures for Experiment 7."""

    _ensure_dir(figures_dir)
    raw_summary_rows, summary_rows = _split_typed_rows(_read_csv(str(outputs["summary_path"])))
    raw_time_rows, time_rows = _split_typed_rows(_read_csv(str(outputs["time_series_path"])))
    generated: dict[str, str] = {}
    trajectory_specs = (
        ("risk_boost", "风险强化参数", "figure_15_exp7_risk_boost_trajectory.png", "figure_15"),
        ("memory_boost", "记忆强化参数", "figure_16_exp7_memory_boost_trajectory.png", "figure_16"),
        ("share_boost", "互动强化参数", "figure_17_exp7_share_boost_trajectory.png", "figure_17"),
    )
    for analysis_part, parameter_label, filename, output_key in trajectory_specs:
        output_path = figures_dir / filename
        if _plot_experiment_7_trajectory(summary_rows, time_rows, analysis_part, parameter_label, output_path):
            generated[output_key] = str(output_path)

    response_specs = (
        ("figure_18", _plot_experiment_7_risk_peak, "figure_18_exp7_risk_boost_peak_response.png"),
        ("figure_19", _plot_experiment_7_memory_retention, "figure_19_exp7_memory_boost_retention_response.png"),
        ("figure_20", _plot_experiment_7_share_interaction, "figure_20_exp7_share_boost_interaction_response.png"),
    )
    for output_key, plotter, filename in response_specs:
        output_path = figures_dir / filename
        if plotter(summary_rows, output_path):
            generated[output_key] = str(output_path)
    return generated


def _plot_exp2_figure5_recommendation_exposure(
    rows: list[dict[str, str]],
    figures_dir: Path,
    raw_rows: list[dict[str, str]] | None = None,
) -> dict[str, str]:
    recommendation_rows = [
        row
        for row in rows
        if row.get("mechanism_name") == "recommendation_strength"
        and row.get("mechanism_level") in LEVEL_ORDER
        and row.get("mean_candidate_agent_content_pairs") not in ("", None)
    ]
    if not recommendation_rows:
        return {}

    values_by_level: dict[str, list[float]] = {}
    errors_by_level: dict[str, list[float]] = {}
    for level in LEVEL_ORDER:
        values_by_level[level] = [
            sum(
                _float(row, "mean_candidate_agent_content_pairs")
                for row in recommendation_rows
                if row.get("scenario_type") == scenario
                and row.get("mechanism_level") == level
            )
            for scenario in SCENARIO_ORDER
        ]
        errors_by_level[level] = []
        if raw_rows:
            values_by_level[level] = []
            for scenario in SCENARIO_ORDER:
                mean_value, std_value = _cumulative_metric_mean_std(
                    raw_rows,
                    "candidate_agent_content_pairs",
                    {
                        "scenario_type": scenario,
                        "mechanism_name": "recommendation_strength",
                        "mechanism_level": level,
                    },
                    scale=10000.0,
                )
                values_by_level[level].append(mean_value * 10000.0)
                errors_by_level[level].append(std_value)

    figure_path = figures_dir / "figure_5_exp2_recommendation_candidate_reach.png"
    _setup_matplotlib()
    _ensure_dir(figures_dir)
    x = np.arange(len(SCENARIO_ORDER), dtype=float)
    width = 0.22
    fig, ax = plt.subplots(figsize=(7.2, 4.35))
    for offset, level in zip([-width, 0.0, width], LEVEL_ORDER):
        values = [value / 10000.0 for value in values_by_level[level]]
        errors = errors_by_level[level]
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=EXP2_LEVEL_COLORS[level],
            edgecolor="white",
            linewidth=0.8,
            label=LEVEL_LABELS[level],
            yerr=errors if raw_rows else None,
            error_kw={"ecolor": "#555555", "elinewidth": 0.65, "capsize": 2.2, "capthick": 0.65},
        )
        for bar, error in zip(bars, errors if raw_rows else [0.0] * len(bars)):
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + error + max(values) * 0.012,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[scenario] for scenario in SCENARIO_ORDER])
    ax.set_ylabel("候选触达规模（万次）", fontsize=10)
    ax.set_xlabel("误导信息场景", fontsize=10, labelpad=8)
    ax.grid(axis="y", color="#E5E9ED", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=9, frameon=False)
    fig.tight_layout(rect=[0.02, 0.06, 1, 1])
    _save_publication_figure(fig, figure_path)
    plt.close(fig)
    return {"figure_5_recommendation_candidate_reach": str(figure_path)}


def _plot_exp2_figure6_repeat_memory(
    rows: list[dict[str, str]],
    figures_dir: Path,
    raw_rows: list[dict[str, str]] | None = None,
) -> dict[str, str]:
    repeat_rows = [
        row
        for row in rows
        if row.get("mechanism_name") == "repeat_strength"
        and row.get("mechanism_level") in LEVEL_ORDER
        and row.get("mean_memory_emotion") not in ("", None)
    ]
    if not repeat_rows:
        return {}

    figure_path = figures_dir / "figure_6_exp2_repeat_strength_memory_emotion.png"
    _setup_matplotlib()
    _ensure_dir(figures_dir)
    colors = {"low": "#5F8FB8", "medium": "#E7B866", "high": "#4E9A51"}
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.9), sharey=True)
    for ax, scenario in zip(axes, SCENARIO_ORDER):
        scenario_rows = [row for row in repeat_rows if row.get("scenario_type") == scenario]
        for level in LEVEL_ORDER:
            level_rows = [row for row in scenario_rows if row.get("mechanism_level") == level]
            if not level_rows:
                continue
            x_values, y_values = _series(level_rows, "mean_memory_emotion")
            if raw_rows:
                std_by_step = _step_std(
                    raw_rows,
                    "mean_memory_emotion",
                    {
                        "scenario_type": scenario,
                        "mechanism_name": "repeat_strength",
                        "mechanism_level": level,
                    },
                )
                error = np.asarray([std_by_step.get(int(step), 0.0) for step in x_values])
                ax.fill_between(
                    x_values,
                    np.maximum(0.0, np.asarray(y_values) - error),
                    np.minimum(1.0, np.asarray(y_values) + error),
                    color=colors[level],
                    alpha=0.16,
                    linewidth=0.0,
                    zorder=1,
                )
            ax.plot(
                x_values,
                y_values,
                marker="o",
                markersize=2.6,
                linewidth=1.55,
                color=colors[level],
                label=LEVEL_LABELS[level],
            )
        ax.set_title(SCENARIO_LABELS[scenario], fontsize=10.5, fontweight="bold")
        ax.set_xlabel("时间步", fontsize=9)
        ax.grid(axis="y", color="#E5E9ED", linewidth=0.65)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("情绪记忆水平", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.tight_layout(rect=[0.02, 0.08, 1, 1])
    _save_publication_figure(fig, figure_path)
    plt.close(fig)
    return {"figure_6_repeat_strength_memory_emotion": str(figure_path)}


def _plot_exp2_figure7_visibility_interaction(
    summary_rows: list[dict[str, str]],
    figures_dir: Path,
) -> dict[str, str]:
    visibility_rows = [
        row
        for row in summary_rows
        if row.get("mechanism_name") == "visibility_strength"
        and row.get("mechanism_level") in LEVEL_ORDER
        and row.get("mean_unit_interaction_intensity") not in ("", None)
    ]
    if not visibility_rows:
        return {}

    interaction_values_by_level: dict[str, list[float]] = {level: [] for level in LEVEL_ORDER}
    interaction_errors_by_level: dict[str, list[float]] = {level: [] for level in LEVEL_ORDER}
    for level in LEVEL_ORDER:
        for scenario in SCENARIO_ORDER:
            level_rows = [
                row
                for row in visibility_rows
                if row.get("mechanism_level") == level
                and row.get("scenario_type") == scenario
            ]
            row = level_rows[0] if level_rows else {}
            interaction_values_by_level[level].append(_float(row, "mean_unit_interaction_intensity"))
            interaction_errors_by_level[level].append(_float(row, "std_unit_interaction_intensity"))

    _setup_matplotlib()
    _ensure_dir(figures_dir)
    figure_path = figures_dir / "figure_7_exp2_visibility_user_interaction_intensity.png"
    x = np.arange(len(SCENARIO_ORDER), dtype=float)
    width = 0.22
    fig, ax = plt.subplots(figsize=(7.2, 4.35))
    for offset, level in zip([-width, 0.0, width], LEVEL_ORDER):
        bars = ax.bar(
            x + offset,
            interaction_values_by_level[level],
            color=EXP2_LEVEL_COLORS[level],
            edgecolor="white",
            linewidth=0.8,
            width=width,
            label=LEVEL_LABELS[level],
            yerr=interaction_errors_by_level[level],
            error_kw={"ecolor": "#555555", "elinewidth": 0.65, "capsize": 2.2, "capthick": 0.65},
        )
        for bar, error in zip(bars, interaction_errors_by_level[level]):
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + error,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[scenario] for scenario in SCENARIO_ORDER])
    ax.set_ylabel("单位用户互动强度", fontsize=10)
    ax.set_xlabel("误导信息场景", fontsize=10, labelpad=8)
    ax.grid(axis="y", color="#E5E9ED", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=9, frameon=False)
    fig.tight_layout(rect=[0.02, 0.06, 1, 1])
    _save_publication_figure(fig, figure_path)
    plt.close(fig)
    return {"figure_7_visibility_user_interaction_intensity": str(figure_path)}


def plot_experiment_2(outputs: dict[str, object], figures_dir: Path) -> dict[str, str]:
    _ensure_dir(figures_dir)
    generated: dict[str, str] = {}
    time_series_path = Path(str(outputs.get("time_series_path", "")))
    raw_rows, time_rows = _split_typed_rows(_read_csv(time_series_path))
    _, summary_rows = _split_typed_rows(_read_csv(str(outputs.get("summary_path", ""))))
    generated.update(
        _plot_exp2_figure5_recommendation_exposure(
            time_rows,
            figures_dir,
            raw_rows,
        )
    )
    generated.update(
        _plot_exp2_figure6_repeat_memory(
            time_rows,
            figures_dir,
            raw_rows,
        )
    )
    generated.update(_plot_exp2_figure7_visibility_interaction(summary_rows, figures_dir))
    return generated


def expand_experiment_2() -> dict[str, str]:
    exp_dir = EXPERIMENTS_DIR / "exp2_platform_main_effects"
    figures_dir = FIGURES_DIR / "exp2"

    generated: dict[str, str] = {}
    raw_rows, time_rows = _split_typed_rows(_read_csv(exp_dir / "exp2_time_series.csv"))
    _, summary_rows = _split_typed_rows(_read_csv(exp_dir / "exp2_summary.csv"))
    generated.update(_plot_exp2_figure5_recommendation_exposure(time_rows, figures_dir, raw_rows))
    generated.update(_plot_exp2_figure6_repeat_memory(time_rows, figures_dir, raw_rows))
    generated.update(_plot_exp2_figure7_visibility_interaction(summary_rows, figures_dir))
    return generated


def expand_experiment_3() -> dict[str, str]:
    exp_dir = EXPERIMENTS_DIR / "exp3_scenario_platform_interaction"
    figures_dir = FIGURES_DIR / "exp3"
    raw_summary_rows, rows = _split_typed_rows(_read_csv(exp_dir / "exp3_summary.csv"))
    raw_time_rows, time_rows = _split_typed_rows(_read_csv(exp_dir / "exp3_time_series.csv"))
    if not rows:
        return {}
    return _plot_experiment_3_summary(rows, time_rows, raw_summary_rows, raw_time_rows, figures_dir)


def expand_experiment_4() -> dict[str, str]:
    exp_dir = EXPERIMENTS_DIR / "exp4_sensitivity_analysis"
    figures_dir = FIGURES_DIR / "exp4"
    run_rows, rows = _split_typed_rows(_read_csv(exp_dir / "exp4_summary.csv"))
    if not rows:
        return {}
    return _plot_experiment_4_peak_sensitivity_figures(rows, figures_dir, run_rows)


def generate_expanded_outputs() -> dict[str, str]:
    outputs: dict[str, str] = {}
    outputs.update(expand_experiment_2())
    outputs.update(expand_experiment_3())
    outputs.update(expand_experiment_4())
    return outputs
