# v12 无损步骤记忆（lossless_step_memory_v12）

v12 保留 v11 的前端（1 FPS 采样、VAM 图像过滤、只看过去的 CLIP 边界）和问答 agent loop，只改写入器与管理器。

```text
P 大事件      标题（开新 P 时写）+ 摘要（PH 关闭或建快照时写）
└ PH 阶段     标题 + 摘要
   └ Step     可选的步骤短语（"add ham"），覆盖同一 PH 内连续的一个或多个 E
      └ E     写入器原始输出，永不合并、改写或删除
```

## 写入器

每个候选片段调用一次，输出英文 JSON：

| 字段 | 内容 |
| --- | --- |
| `caption` | 逐行观察；时间标题由程序按帧时间添加 |
| `summary` | 一句话（≤40 词）：地点、主要人物/物体、具体活动和结果，供管理器归组 |
| `changes` | 可见状态变化 `{object, change, result}`，最多 8 条，静态片段为 `[]` |
| `keywords` | 3–12 个英文名词短语，供检索 |

形状破碎、模糊或像重建伪影的物体只描述可见特征，不猜测名称。

## 管理器

- 工作缓存最多 5 个待归组 E，满 5 个或距上次复核 30 秒触发；不允许等待，每次复核放置全部待归组事件。
- 输入只有每个 E 的 summary、changes 和至多 8 张锚点帧，加上当前 PH 的标题、最近步骤和最后一个事件的 summary；不读 caption。
- 输出只做放置：`phase` 为 current/new，`episode` 为 current/new，`step` 为 current/new/none；只有新建节点时写标题。
- PH/P 摘要由单独的纯文字调用生成：PH 关闭时写一次，开放中的 PH 在快照时补写。

## 与 v11 问答的兼容

`build/rolling/<video-id>/memory.json` 是 v12 状态的 v11 投影：每个 PH 一个 E，caption 按时间拼接成员块并保留各自的观察窗口；Step、Keywords、Change 以普通行附在所属块内，可被逐行检索命中。现有 `run_agent_epm_pilot.py` 直接读取该文件，不需要修改。`build/v12/<video-id>/memory_v12.json` 是权威状态。

## 运行

每个提问时刻构建一个前缀（与 v11 相同），目录对齐 `<source-root>/jobs/<job-id>/build/rolling/<video-id>/memory.json`：

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/build_lossless_memory.py \
  --video data/videos/OpenEQA/017-hm3d-Dd4bFSTQ8gi.mp4 --frontier 59 --video-id ovo-256 \
  --writer-model models/writer --vision-encoder models/clip \
  --output outputs/sources-v12/jobs/ovo-256
```

测试使用脚本化模型，只验证结构与因果约束，不是语义质量评测：

```bash
PYTHONPATH=src:scripts python -m unittest tests.test_lossless_hierarchy tests.test_build_lossless_memory -v
```
