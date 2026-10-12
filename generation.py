from __future__ import annotations

from typing import Dict, Mapping

from prompts import build_seed_generation_prompt, strategy_strength_description


SCENARIO_GENERATION_SPECS: Mapping[str, Mapping[str, object]] = {
    "risk_exaggeration": {
        "display_name": "风险夸大型",
        "title": "风险警示帖",
        "strategy": "围绕客观背景中已经存在的风险因素构造连续的条件性风险链，依次突出事态扩大、潜在损失与应对压力，但不得把任何假设写成已经发生的事实。",
        "style_requirements": "标题直接突出潜在风险；正文用200至260个汉字形成至少四层递进的条件性后果链，并明确使用“首先”“进一步”“一旦”“最终”或同等递进连接词。每一层都必须包含可能损失、影响扩大或应对压力，并从客观背景已有事实出发。除开头交代背景外，正文每句话都要服务于风险升级，不得用中性信息稀释主线；可用一句“当前信息主要描述现状，长期影响仍待观察”形成轻微辅助，结尾只用一句克制的“值得持续关注”表达。",
        "forbidden_patterns": "输出前必须自检并删除所有问号。不得新增已经发生的状况、人物反应、具体数据、防控缺口、医疗诊断或确定性灾难结果；不得讨论公告或提示是否充分，不得评价主体、措施或责任，不得使用反问、隐瞒暗示或“谁负责”“没说清”“背后”等怀疑表达；不得号召转发、评论，不得写成阴谋论或辱骂口号。",
        "strategy_intensity_ranges": {
            "risk_salience": (0.70, 0.80),
            "selective_interpretation": (0.10, 0.20),
            "emotional_mobilization": (0.10, 0.20),
        },
        "text_template": "注意到这条信息：{summary}。如果相关风险继续发展，可能带来更大的影响，因此仍有必要保持关注。",
    },
    "selective_interpretation": {
        "display_name": "选择性解释型",
        "title": "片段解读帖",
        "strategy": "只对客观背景中已有事实进行不对称取舍和排列：选择其中一个局部细节作为全文核心，省略或弱化能够平衡判断的其他背景，使局部事实获得超出完整背景的解释分量。",
        "style_requirements": "标题必须同时包含一个已有局部事实和对该事实的偏向性解释，不能只是客观摘要。正文用130至190个汉字完成四步结构：只重述一个局部事实；反复强化该细节的重要性；不提至少两项能够平衡判断的背景信息；最后必须使用“这表明”“这意味着”或“由此看来”，把该细节解释成普通对象被置于次要位置、资源明显倾斜、管理重点发生偏移或程序只服务于特定侧面。该结论必须超出客观背景的直接陈述并成为全文落点。结尾仅允许补充一句克制的“长期影响仍需观察，相关安排值得继续讨论”，作为极弱的风险关注和表达意愿，不得扩展为后果链或互动号召。正文不得连续复述中性背景。",
        "forbidden_patterns": "不得照抄或完整复述客观背景；不得新增人物证言、机构回应、未公开数据、缺失流程或背景中不存在的因果关系；不得使用“有居民反映”“工作人员表示”等虚构来源；不得暗示机构隐瞒或阴谋；不得使用风险恐吓、反问、绝对结论或情绪号召。",
        "strategy_intensity_ranges": {
            "risk_salience": (0.01, 0.10),
            "selective_interpretation": (0.70, 0.80),
            "emotional_mobilization": (0.01, 0.10),
        },
        "text_template": "这条信息里有个细节值得注意：{summary}。如果只把注意力放在这一细节上，读者可能形成与完整背景不同的理解。",
    },
    "emotional_wording": {
        "display_name": "情绪化措辞型",
        "title": "情绪表达帖",
        "strategy": "完整保留客观背景中的事实命题，只改变语言表达层：通过强烈的主观负面评价、感叹句、短句节奏和第一人称情绪表达增强不满，并明确呈现忍不住表达态度的冲动。",
        "style_requirements": "标题必须是没有问号的强烈负面感叹句。正文必须始终表达愤怒、厌烦或不满，不得转为焦虑、恐惧、安心、期待或赞许；用110至150个汉字先压缩复述客观背景，再连续使用至少六处明确的愤怒、火大、厌烦、受够、憋屈或强烈不满表达，其中至少三处为第一人称直接表达，最后明确表示自己忍不住要把这种不满说出来。全文只表达个人愤怒与表达意愿，不评价任何主体的能力、动机、责任或信息完整性。",
        "forbidden_patterns": "输出前必须自检并删除所有问号。不得新增场景、人物状态、危险后果、风险概率、措施缺口、责任事实、背景缺失或具体数据；不得出现“为什么”“谁负责”“就这点措施”“没说清楚”“打发”“敷衍”“隐瞒”“会不会”“万一”“更严重”“该不该”“拿不准”“犹豫”“纠结”“担心”“害怕”“不安”“提心吊胆”“七上八下”等风险判断、焦虑恐惧、事实怀疑或决策迟疑表达；不得使用“终于”“安心”“踏实”“放心”“好事”“顺利”等正向情绪词；不得号召转发或虚构他人反应，不得辱骂具体主体。",
        "strategy_intensity_ranges": {
            "risk_salience": (0.01, 0.10),
            "selective_interpretation": (0.01, 0.10),
            "emotional_mobilization": (0.70, 0.80),
        },
        "text_template": "看到这条信息确实让人难以平静：{summary}。就这件事本身而言，这样的状况让人失望，也让人忍不住表达不满。",
    },
    "compound_amplification": {
        "display_name": "复合强化型",
        "title": "复合风险动员帖",
        "strategy": "围绕客观背景中的同一事实，同时建立三条彼此可区分且贯穿全文的表达线索：用条件性后果链持续放大潜在风险；通过突出局部事实、弱化平衡背景并附加偏向性结论形成片面解释；通过鲜明的负面评价和明确互动号召推动表达。三项线索必须共同组织标题和正文，不能只由一句话象征性出现。",
        "style_requirements": "标题同时呈现风险警示、偏向判断和强烈负面态度。正文用240至320个汉字自然衔接三条线索：至少使用“进一步”“一旦”“最终”或同等连接词形成三层条件性风险后果；持续突出一个已有局部事实，弱化至少两项能够平衡判断的背景，并用“这表明”“这意味着”或同等语义得出偏向性结论；连续使用至少十处明确的愤怒、不满、谴责、厌烦或强烈感叹表达，其中至少五处使用第一人称直接表达愤怒，并在结尾明确邀请读者评论、回复或转发。全文不得使用问号或反问句，片面解释必须通过陈述式的信息取舍和偏向结论实现，不得用质疑语气替代愤怒表达。三条线索均需在全文中重复得到语义支持，不得写成三个互不相关的机械段落。",
        "forbidden_patterns": "不得凭空新增主体、数字、人物证言、病原体、疫苗副作用、医疗缺口或已经发生的灾难结果；风险后果必须保留条件性，不得写成确定事实；不得声称机构隐瞒真相或构造阴谋；不得机械罗列三项策略；不得把提示词、强度值或评分规则写入正文。",
        "strategy_intensity_ranges": {
            "risk_salience": (0.70, 0.80),
            "selective_interpretation": (0.70, 0.80),
            "emotional_mobilization": (0.70, 0.80),
        },
        "text_template": "这条信息远没有表面上那么简单：{summary}。潜在风险、未被完整说明的背景以及处理方式本身都值得警惕，不能让关键问题在沉默中被轻轻带过。",
    },
}

PROMPT_LEAK_MARKERS = (
    "[TASK:",
    "GENERATE_MISINFORMATION",
    "ASSESS_MISINFORMATION",
    "dominant_negative_emotion",
    "arousal_score",
    "misleading_ratio",
    "JSON 字段",
    "只输出 JSON",
    "输出字段",
    "评分规则",
    "提示词",
    "任务说明",
    "策略表达强度",
    "风险凸显强度",
    "选择性解释强度",
    "情绪动员强度",
    "不要求加总为 100",
    "不要求加总为100",
    "/100",
)


def contains_prompt_leak(text: str) -> bool:
    return any(marker in text for marker in PROMPT_LEAK_MARKERS)


def scenario_generation_spec(scenario_type: str) -> Dict[str, object]:
    source = SCENARIO_GENERATION_SPECS.get(
        scenario_type,
        SCENARIO_GENERATION_SPECS["compound_amplification"],
    )
    spec = dict(source)
    spec["strategy_intensity_ranges"] = dict(source["strategy_intensity_ranges"])
    return spec


def format_strategy_intensity(spec: Mapping[str, object]) -> str:
    intensity_ranges = dict(spec["strategy_intensity_ranges"])
    labels = {
        "risk_salience": "风险强化",
        "selective_interpretation": "片面解释",
        "emotional_mobilization": "情绪动员",
    }
    lines = ["三项策略特征的目标表达区间："]
    for field, label in labels.items():
        bounds = intensity_ranges[field]
        lines.append(
            f"- {label}：{float(bounds[0]):.2f}—{float(bounds[1]):.2f}；"
            f"{strategy_strength_description(bounds)}"
        )
    return "\n".join(lines)


def seed_generation_prompt(
    spec: Mapping[str, object],
    source_event_id: str,
    source_event_summary: str,
) -> str:
    return build_seed_generation_prompt(
        spec,
        source_event_id,
        source_event_summary,
        format_strategy_intensity(spec),
    )
