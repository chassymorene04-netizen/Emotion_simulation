from __future__ import annotations

from datetime import datetime

from config import DEFAULT_SCENARIO, LLMConfig, SimulationConfig, load_environment, load_llm_profiles
from experiments import (
    _build_agent_state_trace_markdown,
    _format_simulation_progress,
    run_experiment_1_expression_styles,
    run_experiment_2_platform_main_effects,
    run_experiment_3_scenario_platform_interaction,
    run_experiment_4_sensitivity_analysis,
    run_experiment_5_agent_scale_robustness,
    run_experiment_6_model_robustness,
    run_experiment_7_strategy_parameter_sensitivity,
    save_csv,
    save_json,
    save_text,
)
from plotting import (
    plot_experiment_1,
    plot_experiment_2,
    plot_experiment_3,
    plot_experiment_4,
    plot_experiment_5_agent_scale,
    plot_experiment_6_model_robustness,
    plot_experiment_7_strategy_parameter_sensitivity,
)
from simulation import EmotionDiffusionSimulation


DEBUG_TIME_SERIES_KEYS = [
    "step",
    "group_negative_emotion",
    "high_negative_ratio",
    "group_dispersion",
    "mean_memory_emotion",
]


def _select_existing(row: dict[str, object], keys: list[str]) -> dict[str, object]:
    return {key: row[key] for key in keys if key in row}


def _log_progress(message: str) -> None:
    """统一输出带时间戳的进度日志。"""

    if not message:
        return
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def main() -> None:
    """运行单次调试仿真与实验一输出。"""

    load_environment()
    simulation_config = SimulationConfig.from_env()
    simulation_config.ensure_directories()
    llm_config = LLMConfig.from_env()
    scenario_type = DEFAULT_SCENARIO
    debug_config = simulation_config.clone_with(
        num_agents=simulation_config.debug_num_agents,
        num_steps=simulation_config.debug_num_steps,
        parallel_workers=simulation_config.debug_parallel_workers,
    )

    print("========== LLM生成误导信息驱动的多智能体情绪扩散仿真 ==========")
    print(f"当前模型：{llm_config.model}")
    print(f"Base URL：{llm_config.base_url}")
    enabled_model_profiles = simulation_config.enabled_model_robustness_profiles
    print(
        "正式实验运行时代理模式："
        f"{'统一LLM-agent架构（事件触发推理）' if simulation_config.runtime_llm_enabled and llm_config.api_key else '规则模式'}"
    )
    if simulation_config.runtime_llm_enabled and llm_config.api_key:
        print(f"运行时LLM批量大小：{simulation_config.llm_agent_batch_size}")
    print(f"调试场景：{scenario_type}")
    print(
        "调试规模："
        f"{debug_config.num_agents} agents × {debug_config.num_steps} steps | "
        f"并发数={debug_config.parallel_workers} | "
        f"运行时LLM={'开启' if debug_config.runtime_llm_enabled else '关闭'}"
    )
    print(
        "正式实验规模："
        f"{simulation_config.num_agents} agents × {simulation_config.num_steps} steps | "
        f"并发数={simulation_config.parallel_workers}"
    )
    print(f"实验一每个场景重复次数：{simulation_config.experiment_runs_per_scenario}")
    print(
        "实验开关："
        f"实验一={'开启' if simulation_config.run_experiment_1 else '关闭'} | "
        f"实验二={'开启' if simulation_config.run_experiment_2 else '关闭'} | "
        f"实验三={'开启' if simulation_config.run_experiment_3 else '关闭'} | "
        f"实验四={'开启' if simulation_config.run_experiment_4 else '关闭'} | "
        f"实验五={'开启' if simulation_config.run_experiment_5_agent_scale else '关闭'} | "
        f"实验六={'、'.join(enabled_model_profiles) if enabled_model_profiles else '关闭'} | "
        f"实验七={'开启' if simulation_config.run_experiment_7_strategy_parameter_sensitivity else '关闭'}"
    )
    print(f"日志详细度：{simulation_config.progress_detail}")
    print(f"实验刺激数据：{simulation_config.experimental_stimuli_path}")
    print(f"输出目录：{simulation_config.output_dir}")
    print("------------------------------------------------")

    simulation = EmotionDiffusionSimulation(
        config=debug_config,
        scenario_type=scenario_type,
        llm_config=llm_config,
        progress_callback=lambda event, payload: _log_progress(
            _format_simulation_progress(
                prefix="[调试运行]",
                event=event,
                payload=payload,
            )
        ),
    )
    result = simulation.run()

    time_series_path = simulation_config.raw_dir / "debug_single_run_time_series.csv"
    summary_path = simulation_config.raw_dir / "debug_single_run_summary.json"
    agent_state_trace_path = simulation_config.raw_dir / "debug_sampled_agent_state_trace.md"
    save_csv(time_series_path, [_select_existing(row, DEBUG_TIME_SERIES_KEYS) for row in result.time_series])
    save_json(summary_path, result.summary_dict())
    save_text(agent_state_trace_path, _build_agent_state_trace_markdown(result.sampled_agent_state_rows))

    print(
        "调试完成 | "
        f"scenario={result.scenario_type} | "
        f"seed_nodes={result.seed_nodes} | "
        f"peak={result.peak:.4f} | "
        f"final_G={result.group_negative_emotion[-1]:.4f} | "
        f"final_H={result.high_negative_ratio[-1]:.4f}"
    )

    experiment_outputs: dict[str, dict[str, object]] = {}
    if simulation_config.run_experiment_1:
        experiment_outputs["experiment_1"] = run_experiment_1_expression_styles(
            base_config=simulation_config,
            llm_config=llm_config,
            progress_callback=_log_progress,
        )
        experiment_outputs["experiment_1"]["figures"] = plot_experiment_1(
            experiment_outputs["experiment_1"],
            simulation_config.figures_dir / "exp1",
        )
    if simulation_config.run_experiment_2:
        experiment_outputs["experiment_2"] = run_experiment_2_platform_main_effects(
            base_config=simulation_config,
            llm_config=llm_config,
            progress_callback=_log_progress,
        )
        experiment_outputs["experiment_2"]["figures"] = plot_experiment_2(
            experiment_outputs["experiment_2"],
            simulation_config.figures_dir / "exp2",
        )
    if simulation_config.run_experiment_3:
        experiment_outputs["experiment_3"] = run_experiment_3_scenario_platform_interaction(
            base_config=simulation_config,
            llm_config=llm_config,
            progress_callback=_log_progress,
        )
        experiment_outputs["experiment_3"]["figures"] = plot_experiment_3(
            experiment_outputs["experiment_3"],
            simulation_config.figures_dir / "exp3",
        )
    if simulation_config.run_experiment_4:
        experiment_outputs["experiment_4"] = run_experiment_4_sensitivity_analysis(
            base_config=simulation_config,
            llm_config=llm_config,
            progress_callback=_log_progress,
        )
        experiment_outputs["experiment_4"]["figures"] = plot_experiment_4(
            experiment_outputs["experiment_4"],
            simulation_config.figures_dir / "exp4",
        )
    if simulation_config.run_experiment_7_strategy_parameter_sensitivity:
        experiment_outputs["experiment_7"] = run_experiment_7_strategy_parameter_sensitivity(
            base_config=simulation_config,
            llm_config=llm_config,
            progress_callback=_log_progress,
        )
        experiment_outputs["experiment_7"]["figures"] = plot_experiment_7_strategy_parameter_sensitivity(
            experiment_outputs["experiment_7"],
            simulation_config.figures_dir / "exp7",
        )
    if simulation_config.run_experiment_5_agent_scale:
        experiment_outputs["experiment_5"] = run_experiment_5_agent_scale_robustness(
            base_config=simulation_config,
            llm_config=llm_config,
            progress_callback=_log_progress,
        )
        experiment_outputs["experiment_5"]["figures"] = plot_experiment_5_agent_scale(
            experiment_outputs["experiment_5"],
            simulation_config.figures_dir / "exp5",
        )
    if enabled_model_profiles:
        profiles = load_llm_profiles(
            enabled_model_profiles,
            llm_config,
        )
        for profile in profiles:
            experiment_outputs["experiment_6"] = run_experiment_6_model_robustness(
                base_config=simulation_config,
                llm_config=llm_config,
                profile=profile,
                progress_callback=_log_progress,
            )
            experiment_outputs["experiment_6"]["figures"] = plot_experiment_6_model_robustness(
                experiment_outputs["experiment_6"],
                simulation_config.figures_dir / "exp6",
            )
            _log_progress(f"[实验六] {profile.profile_id} 已独立保存")

    print(f"调试时间序列已保存：{time_series_path}")
    print(f"调试摘要已保存：{summary_path}")
    print(f"调试用户状态报告已保存：{agent_state_trace_path}")
    if not experiment_outputs:
        print("未运行正式实验。请在 .env 中开启对应实验开关；实验六使用 RUN_EXPERIMENT_6_<模型标识>。")
    for experiment_name, outputs in experiment_outputs.items():
        print(f"{experiment_name} 输出已保存：")
        print(f"- 结果汇总（重复行与统计行）：{outputs['summary_path']}")
        if "time_series_path" in outputs:
            print(f"- 过程时间序列（重复行与统计行）：{outputs['time_series_path']}")
        if "interaction_cases_path" in outputs:
            print(f"- Agent interaction cases: {outputs['interaction_cases_path']}")
        if "agent_state_trace_path" in outputs:
            print(f"- Agent state trace: {outputs['agent_state_trace_path']}")
        if "indicator_path" in outputs:
            print(f"- 指标体系表：{outputs['indicator_path']}")
        print(f"- 实验报告：{outputs['report_path']}")
        figures = outputs.get("figures")
        if isinstance(figures, dict):
            for figure_name, figure_path in figures.items():
                print(f"- {figure_name}：{figure_path}")


if __name__ == "__main__":
    main()
