# v10 小事件输入

`build_buffered_phase.py` 从下列 JSON 与对应 JPG 帧开始。`demo_buffered_phase.py`
可以生成完整的可执行样例。

```json
{
  "kind": "causal_frozen_event_input",
  "videos": [{
    "video_id": "example",
    "frames": {
      "f1": {"id": "f1", "time": 1.0, "sha256": "替换为帧文件的64位SHA256"}
    },
    "steps": [{
      "segment_id": "S1",
      "now": 1.0,
      "frame_ids": ["f1"],
      "events": [{
        "source_event_id": "input-event-1",
        "op": "new",
        "caption": "只描述当前已观察到的画面。",
        "frame_ids": ["f1"],
        "start": 1.0,
        "end": 1.0
      }]
    }]
  }]
}
```

文件位置为 `--frames-run/example/frames/f1.jpg`。每个视频至少一个事务。
视频/帧 ID 应为简单文件名，不包含路径分隔符；仅使用可信的本地清单。

- `now` 是本次可以使用证据的观察前沿，按时间递增；所有引用帧时间必须不晚于它。
- `frame_ids`（事务层）是这一步首次引入的帧。事件的 `frame_ids` 是该事件当前完整、有序、无重复的帧列表。
- `start/end` 必须与事件帧的最早/最晚时间一致。输入帧内容按 SHA256 验证。
- `op=new` 创建小事件；`op=continue` 更新之前的小事件完整 caption 和帧列表。不能续写已经关闭或被合并删除的事件。
- `source_event_id` 仅供离线输入适配器映射当前 E ID，不进入管理器输入或活动记忆，不是 source 事件库。
- 不向写入器或管理器输入未来画面、未来 caption、问题、选项、任务真值或答案。

运行器在本地准备整个已声明前缀的帧文件，但逐步仅将已到达的帧加入 store；模型只获得当前状态及已验证的可见证据。它检查格式与时间合法性，无法自动证明上游 caption 没有使用未来信息，上游需自行保证因果性。

当前 `memory.json` 是最新状态，`manager_state` 保存调度计数。结束后合并会改变事件 ID，不能用最终记忆树回答更早时刻的问题；未来做流式 QA 时应在对应前沿读取/保存记忆快照。

同一 E 的完整帧集合可以大于单次管理调用展示的帧数（默认最多 8 张锚点帧）。`support_frame_ids` 必须来自实际展示的帧；合并后保留的是所有成员帧的并集，不只是 support 帧。
