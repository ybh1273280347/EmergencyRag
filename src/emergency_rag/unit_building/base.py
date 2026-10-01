from abc import ABC, abstractmethod

from emergency_rag.data.models import Rule, SearchUnit


class UnitBuilder(ABC):
    """规则到检索单元的构建接口，负责单元划分与索引文本增强。"""

    # 策略名用于离线产物目录；不同划分或增强实验应提供不同名称。
    name: str

    @abstractmethod
    def build(self, rules: list[Rule]) -> list[SearchUnit]:
        raise NotImplementedError
