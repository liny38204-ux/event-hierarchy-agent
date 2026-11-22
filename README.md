# Event Hierarchy Agent

免训练的流式视频记忆与问答框架。当前 v11 以多模态模型写入带时间窗的事件 caption，再把小事件整理成“大事件 → 阶段 → 合并事件”。提问时，回答器先看最近四帧，证据不足时可以检索历史文字、展开阶段、聚焦时间窗或查看原帧拼图。

| 环节 | 入口 |
| --- | --- |
| 从视频前缀构建 v11 记忆 | [单视频建库](docs/streaming-memory-v11.md) · [writer 提示词](src/event_memory/prompts_caption_hierarchy.py) · [滚动管理器](src/event_memory/rolling_phase.py) |
| 用本地 8B 或视觉 API 模型回答问题 | [运行 OVO EPM 问答与计分](docs/ovo-qa.md) · [检索控制器](src/event_memory/agent_retrieval.py) |
| 检查事件 caption 的向量相似度 | [纯文本检索](docs/text-retrieval.md) |

v11 在缓存累计 5 个事件或距上次整理已过 30 秒时触发复核，时间条件在新事件到达时检查。开放阶段的合并事件会带入下一轮。帧引用与观察时间随事件保存，可供回答器回看。短窗和历史控制器都要求显式的布尔字段 `evidence_sufficient`；尚有工具预算时，填入候选选项且证据不足会继续查找。预算耗尽后才进入强制作答。

## v10 历史入口

以下说明用于重放已有的 v10 小事件事务。

```text
大事件 P：简短描述一个整体活动
└── 阶段 PH：简短描述一段具体活动
    ├── 进行中：多个小事件 E，各自保存 caption 和直接帧引用
    └── 结束后：全部小事件合为一个 E，保留完整有序帧集合

待组织区 roots：暂时不能确认阶段归属的小事件
```

## v10 行为

- 累计 **5 个新事件**或 **30 秒视频时间**后触发管理器复核。时间条件在事件事务到达时检查；这是调度阈值，不是缓存硬容量。
- 管理器按时间消费待组织事件的连续前缀，可以划分多个阶段，并保留不确定的后缀继续等待。
- P 表示整体活动，PH 表示具体子区域、步骤或局部活动；共享宽泛主题不能单独成为延续同一 PH 的依据。
- 确认新阶段后关闭前序阶段。只有已关闭阶段才执行一次整体合并；开放阶段继续积累小事件。视频结束只触发最后复核，不强制关闭阶段。
- 合并后删除旧事件及旧 caption，新事件直接引用原成员的全部帧。活动记忆不保存 source 事件库、旧事件别名库或持续状态库。
- caption 写入与阶段结束整理使用逐行观察格式：一个字符串，每行一句主要观察，保留主语、必要细节、时间顺序和不确定性。换行不增加 E 节点，P/PH 仍为简短概括；HTML 和 Markdown 导出保留换行。
- 模型决策经过字段、时间、成员顺序、帧哈希和可见证据校验；无效决策重试，失败不提交部分状态。

v10 的重放入口接收已经写好的小事件事务。行式 caption 的示例见 [docs/caption-lines.md](docs/caption-lines.md)。

## 快速检查

Python 3.10+。核心存储和管理逻辑仅依赖标准库；单元测试使用模拟模型。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
PYTHONPATH=src:scripts python -m unittest tests.test_buffered_phase tests.test_buffered_runner tests.test_phase_end_collapse tests.test_backend_receipt
```

包含图像前端的完整便携测试：

```bash
python -m pip install -r requirements-image-filter.txt
PYTHONPATH=src:scripts python -m unittest discover -s tests -v
```

不需要 GPU 的合成示例，展示等待、归组、阶段结束合并和最终记忆导出：

```bash
python scripts/demo_buffered_phase.py --output outputs/demo
python scripts/export_memory_browser.py --run outputs/demo/run --output outputs/demo/memory_browser.html
```

示例使用脚本化的模型回复与生成的简单图形，不是实际模型推理或语义质量评测。

## v10 模型重放

当前后端冻结使用 `Qwen/Qwen3.8-27B`，版本见 `configs/qwen38_model.lock.json`。最近诊断记录的环境是 Torch 2.9.1+cu128、Transformers 5.19.0；后端要求恰好两张可见 GPU，每张分配上限 27 GiB，BF16、SDPA、贪心生成，不允许 CPU/disk offload。两张 24 GB 卡不能直接套用这一配置。

```bash
python -m pip install -r requirements-gpu.txt
python scripts/download_qwen38.py \
  --lock configs/qwen38_model.lock.json --destination models/qwen38
CUDA_VISIBLE_DEVICES=0,1 python scripts/build_buffered_phase.py \
  --input data/events.json --frames-run data/frames \
  --model models/qwen38 --output outputs/memory
```

模型下载脚本会校验固定版本的大小与哈希，并生成后端所需的 `verified_model.json`。请自行选择空闲设备和足够的存储空间；仓库不包含服务器地址、调度器或模型权重。

输入格式与因果约束见 [docs/input-format.md](docs/input-format.md)。运行会生成 `memory.json`、`memory_tree.md`、帧副本、调用记录及事务日志；这些产物默认被 Git 忽略。`scripts/recover_buffered_phase.py --help` 提供中断恢复：仅在提示词、阶段、证据及推理配置匹配时复用有序调用前缀，并区分缓存与新推理。

## v10 独立图像与 caption 前端

`scripts/build_caption_memory.py --help` 提供视频前缀采样、图像去重、因果边界检测和 caption 写入。默认图像过滤沿用 VAM 的模糊度、像素差和 SSIM 门控；**不做 embedding 去重**。CLIP embedding 用于候选片段边界检测，不能与去重混淆。

这一前端输出的是 caption 片段，**不能直接当作 v10 的小事件事务输入**；两者之间的 E 创建/续写适配尚未整理为当前版本的统一入口。`retained_frontend.py` 另提供对已保留帧的哈希检查及片段批次生成。

## v10 代码位置

| 文件 | 用途 |
| --- | --- |
| `src/event_memory/buffered_phase.py` | 当前管理提示词、5/30 调度和 P/PH 归组 |
| `src/event_memory/phase_end_collapse.py` | 阶段结束后整体合并 |
| `src/event_memory/direct_frame_memory.py` | 直接帧存储与事务校验的共享底层 |
| `src/event_memory/phase_lifecycle.py` | 阶段/大事件闭合及确认时间 |
| `src/event_memory/backend_qwen38.py` | 冻结模型推理与调用记录 |
| `src/event_memory/text_retrieval.py` | 整段／逐句余弦相似度取最大值，返回 E 和 PH/P 上下文 |
| `scripts/search_caption_memory.py` | 使用本地文本 embedding 模型检索已有 v10 快照 |
| `scripts/build_buffered_phase.py` | 当前 v10 因果小事件重放入口 |
| `scripts/export_memory_browser.py` | 当前态可折叠浏览页导出 |

部分底层模块保留历史命名及通用函数，当前入口使用 `BufferedPhaseStore` / `BufferedPhaseManager` / `PhaseEndCollapseManager`。不要调用旧的两两合并语义代替当前阶段整体合并。

VAM 图像过滤参考与许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。第三方数据、模型分别受其原有许可约束；此快照未为原创代码另行指定开放源代码许可证。
