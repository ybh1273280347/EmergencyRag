from abc import ABC, abstractmethod

from emergency_rag.data.models import Rule, SearchUnit


class Chunker(ABC):
    """知识记录到检索单元的分块接口。"""

    # 英文策略名用于离线产物目录；不同分块实验应提供不同名称。
    name: str

    @abstractmethod
    def chunk(self, records: list[Rule]) -> list[SearchUnit]:
        raise NotImplementedError
