"""BM25S 的导入边界，将第三方导入诊断移到 stderr。"""

import sys
from contextlib import redirect_stdout

# BM25S 0.2.x 在 Windows 导入 benchmark 时 print resource 缺失提示。
# 只将第三方导入诊断移到 stderr，不修改库或屏蔽后续业务输出。
with redirect_stdout(sys.stderr):
    import bm25s
    from bm25s.tokenization import Tokenizer
