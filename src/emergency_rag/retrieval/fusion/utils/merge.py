"""融合内部共用的候选合并工具：保留召回观察，不改写输入对象。"""

from emergency_rag.retrieval.models import Candidate


def merge_candidate(merged: dict[str, Candidate], candidate: Candidate) -> None:
    # 首次出现：深拷贝后写入，避免修改调用方对象
    if candidate.unit_id not in merged:
        merged[candidate.unit_id] = candidate.model_copy(deep=True)
        return

    existing = merged[candidate.unit_id]

    # 同一 unit_id 必须对应相同规则与文本
    if existing.rule_id != candidate.rule_id or existing.text != candidate.text:
        raise ValueError(f"同一 unit_id 对应不同规则内容：{candidate.unit_id}")

    # 合并来源，按已有顺序去重
    for source in candidate.sources:
        if source not in existing.sources:
            existing.sources.append(source)

    # 合并召回观察；copy 避免共享内部对象
    observations = existing.metadata.setdefault("retrieval", [])
    for observation in candidate.metadata.get("retrieval", []):
        if observation not in observations:
            observations.append(observation.copy())