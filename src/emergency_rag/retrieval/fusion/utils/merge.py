"""融合内部共用的候选合并工具：保留召回观察，不改写输入对象。"""

from emergency_rag.retrieval.models import Candidate


def merge_candidate(merged: dict[str, Candidate], candidate: Candidate) -> None:
    if candidate.unit_id not in merged:
        merged[candidate.unit_id] = candidate.model_copy(deep=True)
        return
    existing = merged[candidate.unit_id]
    if existing.rule_id != candidate.rule_id or existing.text != candidate.text:
        raise ValueError(f"同一 unit_id 对应不同规则内容：{candidate.unit_id}")
    for source in candidate.sources:
        if source not in existing.sources:
            existing.sources.append(source)
    observations = existing.metadata.setdefault("retrieval", [])
    for observation in candidate.metadata.get("retrieval", []):
        if observation not in observations:
            observations.append(observation.copy())
