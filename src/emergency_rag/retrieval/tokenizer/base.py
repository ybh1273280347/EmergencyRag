from abc import ABC, abstractmethod


class TextTokenizer(ABC):
    """索引和查询共享的文本分词接口。"""

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        raise NotImplementedError
