from emergency_rag.data.models import Rule, SearchUnit

from .base import Chunker


class RuleChunker(Chunker):
    name = "rule"

    def chunk(self, records: list[Rule]) -> list[SearchUnit]:
        return [
            SearchUnit(unit_id=f"rule:{rule.rule_id}", rule_id=rule.rule_id, text=rule.text)
            for rule in records
        ]
