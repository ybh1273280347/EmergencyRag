from abc import ABC, abstractmethod
from emergency_rag.data.indexed_dataset import IndexedDataset
from emergency_rag.retrieval.models import Candidate


class Retriever(ABC):
    """独立召回接口；预算在初始化时确定，数据集由 Pipeline 统一传入。"""

    name: str

    @abstractmethod
    def retrieve(self, query: str, dataset: IndexedDataset) -> list[Candidate]:
        raise NotImplementedError
