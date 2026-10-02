from .base import UnitBuilder, Rule, SearchUnit


class RuleUnitBuilder(UnitBuilder):
    name = "rule"

    def build(self, rules: list[Rule]) -> list[SearchUnit]:
        return [
            SearchUnit(
                unit_id=f"rule:{rule.rule_id}", rule_id=rule.rule_id,
                text=rule.text, index_text=rule.text,
            )
            for rule in rules
        ]
