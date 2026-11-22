# OVO EPM 的记忆构建与问答

以下用 OVO-Bench 的第 256 题展示完整调用。先按官方目录准备视频及标注，清单生成器会核对输入标注覆盖的视频。每个步骤在仓库根目录运行。[OVO-Bench 数据说明](https://github.com/joeleelyf/OVO-Bench)

## 准备环境与模型

安装匹配 GPU 的 Torch 及模型依赖。使用仓库的下载脚本读取以下固定版本清单，校验下载内容并生成模型加载器使用的 `verified_model.json`。

```bash
python -m pip install -e .
python -m pip install -r requirements-image-filter.txt -r requirements-gpu.txt
python scripts/download_qwen38.py --lock configs/qwen38_model.lock.json --destination models/writer
python scripts/download_qwen38.py --lock configs/clip.lock.json --destination models/clip
python scripts/download_qwen38.py --lock configs/qwen3_embedding.lock.json --destination models/embedding
python scripts/download_qwen38.py --lock configs/qwen3vl8.lock.json --destination models/reader
```

## 冻结输入并构建记忆

`reader.json` 包含问题及选项。`jobs.json` 给出每个视频前缀的标识、提问时间及关联题号。写入器只接收视频和截止时间。下例中 ovo-256 对应 59 秒；其他题的标识和时间应读生成的清单。

```bash
python scripts/prepare_ovo_non_fpd_inputs.py --annotations data/ovo_bench_new.json \
  --video-root data/videos --output outputs/inputs --include-fpd

CUDA_VISIBLE_DEVICES=0,1 python scripts/build_streaming_memory.py \
  --video data/videos/OpenEQA/017-hm3d-Dd4bFSTQ8gi.mp4 --frontier 59 --video-id ovo-256 \
  --writer-model models/writer --vision-encoder models/clip \
  --output outputs/sources/jobs/ovo-256/build
```

## 回答与计分

8B 使用一张 GPU。源记忆应位于 `<source-root>/jobs/<job_id>/build/rolling/<video_id>/memory.json`。请保留配套的帧和拼图。默认正常路由允许短窗直接作答。历史分支最多调用 20 次工具，包含相似度搜索、阶段展开、时间窗文字查询和历史图像检查。检索依据是管理器合并后的 caption。评分程序在预测冻结后读取官方标签，目前该脚本用于 EPM。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_agent_epm_pilot.py \
  --configs outputs/inputs --source-root outputs/sources --output outputs/qa-8b \
  --answer-model models/reader --embedding-model models/embedding \
  --id 256 --policies search_agent --evaluations normal --max-history-tools 20

python scripts/score_agent_epm_pilot.py --pilot outputs/qa-8b --official data/ovo_bench_new.json
```

选择支持图像的 Chat Completions 服务时，在本机环境中设置 `CHAT_API_KEY` 与 `CHAT_API_BASE_URL`。后者为服务的 API 根地址，适配器在其后追加 `/chat/completions`。将上例的 `--answer-model models/reader` 改为 `--api-model <服务提供的模型名>`，并使用独立的 `--output outputs/qa-api`。图像与提问会送往所选服务，文本向量在本机 CPU 计算。API 使用模型返回的选项字母，本地 8B 在强制作答时比较允许选项的 logits。

各次调用写入 `reader.jsonl`。正常路由的完整轨迹位于 `normal/search_agent/answers/<id>.json`，包含实际使用的工具、证据引用、是否强制作答与最终选项。修改模型、记忆或输入后，使用新的输出目录。
