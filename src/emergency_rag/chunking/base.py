from abc import ABC, abstractmethod

from emergency_rag.data.models import Rule, SearchUnit


class Chunker(ABC):
    """知识记录到检索单元的分块接口。"""

    @abstractmethod
    def chunk(self, records: list[Rule]) -> list[SearchUnit]:
        raise NotImplementedError
