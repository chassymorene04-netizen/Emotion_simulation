from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, TYPE_CHECKING

import networkx as nx
import numpy as np

from config import SimulationConfig
from formulas import (
    clip_01,
    exposure_probability,
    recommendation_signal,
    repeated_exposure_signal,
    safe_mean,
    social_visibility_signal,
)
from models import ActionEvent, ContentItem, FeedBundle

if TYPE_CHECKING:
    from agent import Agent


def build_network(
    network_type: str,
    num_agents: int,
    edge_probability: float,
    random_seed: int,
) -> nx.DiGraph:
    """构建仿真网络。"""

    if network_type == "directed_random":
        graph = nx.gnp_random_graph(
            n=num_agents,
            p=edge_probability,
            seed=random_seed,
            directed=True,
        )
        graph.remove_edges_from(nx.selfloop_edges(graph))
        return graph

    if network_type in {"small_world", "scale_free"}:
        raise NotImplementedError(f"网络类型 {network_type} 尚未实现。")

    raise ValueError(f"未知网络类型：{network_type}")


@dataclass
class _ScoredContent:
    item: ContentItem
    score: float
    recommendation: float
    repeated: float
    visibility: float
    style_repeat_ratio: float
    topic_repeat_ratio: float


def _neighbor_action_strength(
    content: ContentItem,
    neighbor_ids: Iterable[int],
    recent_actions_by_content: Dict[str, List[ActionEvent]],
    action_weights: Dict[str, float],
) -> float:
    strengths: List[float] = []
    for action in recent_actions_by_content.get(content.content_id, []):
        if action.actor_id not in neighbor_ids:
            continue
        strengths.append(action_weights.get(action.action_type, 0.0) * action.action_intensity)
    return clip_01(safe_mean(strengths))


def _resolve_root_post_item(
    selected_item: ContentItem,
    content_lookup: Optional[Dict[str, ContentItem]],
) -> Optional[ContentItem]:
    root_id = selected_item.root_id()
    if content_lookup is None:
        return selected_item if selected_item.content_id == root_id else None
    root_item = content_lookup.get(root_id)
    if root_item is not None:
        return root_item
    return selected_item if selected_item.content_id == root_id else None


def _resolve_interaction_context_item(
    selected_item: ContentItem,
    root_post_item: Optional[ContentItem],
    candidate_content: List[ContentItem],
) -> Optional[ContentItem]:
    if root_post_item is not None and selected_item.content_id != root_post_item.content_id:
        return selected_item

    thread_root_id = selected_item.root_id()
    thread_items = [
        item
        for item in candidate_content
        if item.root_id() == thread_root_id
        and (root_post_item is None or item.content_id != root_post_item.content_id)
    ]
    if not thread_items:
        return None
    return max(thread_items, key=lambda item: (item.created_step, item.thread_depth, item.content_id))


def build_agent_feed(
    agent: Agent,
    graph: nx.DiGraph,
    candidate_content: List[ContentItem],
    recent_actions_by_content: Dict[str, List[ActionEvent]],
    config: SimulationConfig,
    step: int,
    forced_content_ids: Optional[Set[str]] = None,
    content_lookup: Optional[Dict[str, ContentItem]] = None,
) -> FeedBundle:
    """从已具备传播资格的内容中形成单条重点暴露情境。"""

    candidates: List[_ScoredContent] = []
    neighbor_ids = {int(node_id) for node_id in graph.predecessors(agent.node_id)}
    rng = np.random.default_rng(config.random_seed + (step + 1) * 10_000 + int(agent.node_id))

    for item in candidate_content:
        age = step - item.created_step
        if age < 0 or age > config.max_content_age:
            continue

        recommendation_value = recommendation_signal(
            recommendation_strength=config.recommendation_strength,
            hotness_score=item.hotness_score,
            arousal_score=item.arousal_score,
            misleading_ratio=item.misleading_ratio,
        )
        style_repeat_ratio = agent.style_repeat_ratio(item.scenario_type)
        topic_repeat_ratio = agent.topic_repeat_ratio(item.source_event_id)
        repeated_value = repeated_exposure_signal(
            style_repeat_ratio=style_repeat_ratio,
            topic_repeat_ratio=topic_repeat_ratio,
            repeat_strength=config.repeat_strength,
        )
        visibility_value = social_visibility_signal(
            neighbor_interaction_signal=_neighbor_action_strength(
                content=item,
                neighbor_ids=neighbor_ids,
                recent_actions_by_content=recent_actions_by_content,
                action_weights=config.action_weights,
            ),
            visibility_strength=config.visibility_strength,
        )
        exposure_value = exposure_probability(
            recommendation_signal_value=recommendation_value,
            repeated_exposure_signal_value=repeated_value,
            visibility_signal_value=visibility_value,
        )
        is_forced = item.content_id in (forced_content_ids or set())
        if not is_forced and rng.random() > exposure_value:
            continue

        candidates.append(
            _ScoredContent(
                item=item,
                score=exposure_value,
                recommendation=recommendation_value,
                repeated=repeated_value,
                visibility=visibility_value,
                style_repeat_ratio=style_repeat_ratio,
                topic_repeat_ratio=topic_repeat_ratio,
            )
        )

    if not candidates:
        return FeedBundle(
            agent_id=agent.node_id,
            items=[],
            recommendation_signal=0.0,
            repeated_signal=0.0,
            visibility_signal=0.0,
            misinfo_exposed=False,
            style_repeat_ratio=0.0,
            topic_repeat_ratio=0.0,
            thread_ids=[],
            root_post_item=None,
            interaction_context_item=None,
        )

    selected = max(candidates, key=lambda row: (row.score, row.item.created_step))
    root_post_item = _resolve_root_post_item(selected.item, content_lookup)
    interaction_context_item = _resolve_interaction_context_item(
        selected.item,
        root_post_item,
        candidate_content,
    )

    return FeedBundle(
        agent_id=agent.node_id,
        items=[selected.item],
        recommendation_signal=selected.recommendation,
        repeated_signal=selected.repeated,
        visibility_signal=selected.visibility,
        misinfo_exposed=selected.item.is_misinformation(config.misinformation_threshold),
        style_repeat_ratio=selected.style_repeat_ratio,
        topic_repeat_ratio=selected.topic_repeat_ratio,
        thread_ids=[selected.item.root_id()],
        root_post_item=root_post_item,
        interaction_context_item=interaction_context_item,
    )
