# 整段与逐句取最大值的纯文本检索

固定当前 P → PH → E 记忆；检索不读图，不调用生成模型，不改写记忆。

对问题 q 和事件 E，使用同一 embedding 模型、同一向量空间：

```text
caption_score = cosine(embedding(q), embedding(E.caption))
line_score = max(cosine(embedding(q), embedding(line)) for each nonempty line)
score(E, q) = max(caption_score, line_score)
```

默认返回得分最高的一个 E，可指定 top_k。每个 E 只返回一次；相同分数按事件开始时间、ID 稳定排序。整段和逐句分数相同，命中来源记为整段。

返回完整 `caption`，保留真实换行，以及所属 PH/P 的 `id/title/summary`。父节点只提供上下文，不参与相似度计算、重排或扩展候选。无父节点的 root E 正常检索，`phase` 和 `episode` 为 null。没有返回帧或图片；`render_context()` 可生成供后续回答模型使用的纯文字。

附带 `caption_score`、`line_score`、`matched` 供排查。`matched.line_index` 为原 caption 中的零起始行号，空白行不参加检索。命中一行不代表它已有精确的帧或时间对齐；`start/end` 仍是整个事件的证据时间范围。相似度不是事实置信度。

## 使用已有本地 embedding 模型

```bash
python -m pip install -r requirements-retrieval.txt
python scripts/search_caption_memory.py \
  --memory outputs/memory/video-id/memory.json \
  --embedding-model /absolute/path/to/local-text-embedding-model \
  --query '展示测试仪之前展示了什么工具？' \
  --top-k 1
```

使用本地 Sentence Transformers 格式的稠密文本模型，默认 CPU，不自动下载模型。应选择支持实际问题和 caption 语种的模型。模型要求检索指令时，按该模型文档显式设置 `--query-prompt` 和 `--document-prompt`；默认均为空。两条文档检索路径始终使用相同的 document prompt。

适配器依据 [Sentence Transformers 官方接口](https://sbert.net/docs/package_reference/sentence_transformer/model.html) 实现。编码前检查包括提示词与特殊 token 在内的长度；超过模型上限直接拒绝，不静默截断或改为分块平均，确保“整段”含义不变。可替换为自有 `encode_documents(texts)` / `encode_query(text)` 适配器，无须修改排序逻辑。

## 流式版本与使用边界

- 索引对应一个不可变 v10 或 v11 快照。事件续写、阶段合并或父摘要更新后，重新构建索引；原索引不会随外部字典修改而悄悄变化。第一版仅在索引内复用完全相同文本的向量，尚未实现跨快照增量缓存。
- `--query-time` 早于快照 `observed_until` 时拒绝检索。必须加载当时的记忆版本，不能用最终摘要回答较早的问题，仅过滤事件时间无法修复未来摘要泄露。
- 快照早于查询时刻时可检索已有历史，但这不证明记忆在查询时刻仍然有效。
- 此命令只输出检索到的文字。带历史工具的问答和独立计分使用 [OVO EPM 入口](ovo-qa.md)。
- 工程测试使用可控向量或模拟模型，不是语义检索成绩。真实记忆结构 smoke 仅验证字段和父链兼容性。

```bash
PYTHONPATH=src python -m unittest tests.test_text_retrieval tests.test_text_embedding -v
```
