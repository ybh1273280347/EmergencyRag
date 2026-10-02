"""单写入进程使用的 JSON 文件缓存，启动时读取，完整写入后原子替换。"""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


class JsonFileCache:
    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        if not isinstance(self.data, dict):
            raise ValueError(f"缓存文件必须为 JSON 对象：{self.path}")

    def save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent, delete=False) as output:
                temporary = Path(output.name)
                json.dump(data, output, ensure_ascii=False, allow_nan=False)
            os.replace(temporary, self.path)
            # 写入失败时保留之前的磁盘内容和内存状态，失败结果不能成为缓存命中。
            self.data = data
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
