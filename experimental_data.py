from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class SourceEvent:
    source_event_id: str
    source_event_summary: str


def default_source_events() -> List[SourceEvent]:
    """Return the neutral event backgrounds used by the study."""

    summaries = [
        "某市一所高校近日发布校园健康提示，提醒学生在季节性呼吸道疾病高发期间注意个人防护。学校建议学生保持宿舍通风，出现发热、咳嗽、咽痛等症状时及时向辅导员报备，并根据校医院建议进行休息或就诊。校方同时表示，目前教学安排保持正常，相关部门将根据实际情况动态调整公共区域消毒和健康提醒频次。",
        "某社区卫生服务中心发布通知，将在本周内开放季节性流感疫苗预约接种服务。居民可通过线上平台或现场登记方式预约，接种对象主要包括老年人、儿童、慢性病患者及有接种需求的普通居民。社区工作人员提醒，疫苗接种遵循自愿原则，居民可根据自身身体状况和医生建议决定是否接种。",
        "某市多家医院近期呼吸道门诊就诊人数有所增加，主要症状包括发热、咳嗽、鼻塞和咽痛等。医院方面表示，秋冬季节呼吸道疾病就诊量波动较为常见，已根据门诊情况增设导诊人员并优化候诊流程。医务人员建议市民保持良好作息，注意手部卫生，如症状持续或加重，应及时就医。",
        "某地公共交通枢纽近日发布管理提示，表示将根据季节性传染病防控需要，适当增加候车大厅、卫生间、电梯按钮等公共区域的清洁和消毒频次。同时，工作人员提醒乘客在人员密集区域注意个人防护，保持良好卫生习惯。相关措施属于常规公共卫生管理安排，交通运行秩序保持正常。",
    ]
    return [
        SourceEvent(f"event_{index + 1:03d}", summary)
        for index, summary in enumerate(summaries)
    ]


def load_experimental_stimuli(path: Path) -> Dict[str, List[Dict[str, object]]]:
    """Load the versioned, read-only stimuli used in the reported experiments."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"实验刺激数据不可读取：{path}") from error
    scenarios = raw.get("scenarios") if isinstance(raw, dict) else None
    if not isinstance(scenarios, dict):
        raise ValueError(f"实验刺激数据结构错误：{path}")
    return {
        str(scenario_type): [dict(row) for row in rows if isinstance(row, dict)]
        for scenario_type, rows in scenarios.items()
        if isinstance(rows, list)
    }


def experimental_stimulus_event_id(
    stimuli: Dict[str, List[Dict[str, object]]],
    scenario_type: str,
) -> Optional[str]:
    rows = stimuli.get(scenario_type, [])
    if not rows:
        return None
    event_id = rows[0].get("source_event_id")
    return str(event_id).strip() if event_id else None


PROMPT_LEAK_MARKERS = (
    "只输出",
    "JSON",
    "字段",
    "提示词",
    "任务",
    "要求：",
    "请围绕",
    "scenario_type",
    "action_type",
    "generated_text",
)


def is_clean_action_text(text: str) -> bool:
    value = str(text or "").strip()
    if not value or len(value) > 180:
        return False
    return not any(marker in value for marker in PROMPT_LEAK_MARKERS)
