import logging

import jieba

from .base import TextTokenizer

class JiebaTokenizer(TextTokenizer):
    """独立中文分词规则；查询和索引共享同一实现，不使用英文停用词。"""

    def __init__(self) -> None:
        jieba.setLogLevel(logging.WARNING)
        self.segmenter = jieba.Tokenizer()

    def tokenize(self, text: str) -> list[str]:
        return [
            token.lower()
            for token in self.segmenter.cut(text, cut_all=False)
            if any(character.isalnum() for character in token)
        ]
