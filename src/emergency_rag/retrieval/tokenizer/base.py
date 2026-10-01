from abc import ABC, abstractmethod


class TextTokenizer(ABC):
    """索引和查询共享的文本分词接口。"""

    # 英文策略名参与数据集目录；不同分词配置应提供不同名称。
    name: str

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        raise NotImplementedError
