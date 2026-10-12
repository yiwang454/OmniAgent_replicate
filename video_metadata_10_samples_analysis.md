# DailyOmni O 行：10 个 `video_metadata` 样本与作用分析

## 范围与来源

- 实验：DailyOmni 表格 O 行，`Full OmniAgent (o3)`，reasoning effort = `high`。
- 结果文件：`/mnt/ceph_rbd/workspace/avqa_project/general_scripts/react-agent-avqa-optimize/artifacts/results/daily_omni/omniagent_replication/output_test.jsonl.gz`
- 原始实验结果：`/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_repeat1/output_test.jsonl`
- 总样本数：1,197。
- 调用过 `video_metadata` 的样本数：766（64.0%）。
- `video_metadata` 总调用数：769；报错数：0。
- 下列 10 个样本是从这 766 个样本中人工挑选的代表性例子，既包含 metadata 明确帮助选择 clip 时间窗的轨迹，也包含 metadata 没有被实质利用的轨迹。

## 先说结论：谁在使用 metadata？

`video_metadata` 自己不看视频内容，也不回答问题。它只用 OpenCV 返回：

```json
{
  "duration_seconds": 30,
  "frame_count": 900,
  "fps": 30.0,
  "width": 640,
  "height": 360
}
```

真正“使用”这份 observation 的是中央 planner，即 o3。返回值会被追加到 agent scratchpad / 对话轨迹，o3 在下一轮看到它后决定：

1. 是否调用 `video_clip_qa`；
2. clip 的 `time_range=[start, end]` 应取什么值；
3. 时间窗是否在 `0 <= start < end <= duration_seconds` 内；
4. 或者根本不使用这些信息，转去调用 global/audio 工具或直接回答。

工具之间没有直接的数据流连接：`video_metadata` 不会自动裁剪视频，也不会自动把 duration 传给 `video_clip_qa`。中间唯一的消费者/桥梁是 o3 planner。

## 10 个样本总览

| # | Question ID | metadata 所在位置 | 后续关键动作 | 判断 | GT / Pred |
|---:|---|---|---|---|---|
| 1 | `rieb8cBb2z8-1` | EventLocation 后 | `video_clip_qa [4,8]` 两次 | 明确用于局部视觉核验 | D / D |
| 2 | `Me4W36_lUcI-2` | EventLocation 后 | `video_clip_qa [23,27]` | 明确用于尾段时间窗核验 | C / C |
| 3 | `d6b4OmUFt7I-1` | audio QA 后 | `video_clip_qa [0,4]` | 明确用于开头时间窗核验 | B / B |
| 4 | `XUWxQYmiBQY-1` | EventLocation 后 | clips `[0,12]`, `[0,3]`, `[4,8]` | 支持多次细化搜索 | A / A |
| 5 | `Ec_lQgZ9wlg-1` | 多次 audio 后 | global QA，再连续 6 个 clips | 支持在完整 30 秒内逐段搜索 | B / B |
| 6 | `S8D4VF8Yl1g-2` | 第一调用 | audio 定位后 clips `[6,12]`, `[7,11]` | duration 可作全局边界，但不是直接定位来源 | B / B |
| 7 | `WkBkvlUoZpU-2` | 第一调用 | audio caption 后 clip `[0,10]` | 有使用可能，但问题已给出 0–10 秒 | D / D |
| 8 | `0U_X5BIkYoA-1` | 第一调用 | 只调用 `video_global_qa` | 未用于 clip；基本冗余 | C / C |
| 9 | `7yY1OWjT4zA-1` | 第一调用 | 只调用 `video_global_qa` | 未用于 clip；基本冗余 | D / D |
| 10 | `G8ywbM85GVE-1` | 最后一个工具调用 | 随后直接 final | 没有下游消费者，明显冗余 | A / A |

## 样本详情

### 1. `rieb8cBb2z8-1`：典型的“音频定位 → metadata → clip”

**问题**：音频提到在森林里发现 blue tag 时，画面中有什么视觉细节？

**轨迹**：

```text
Audio_EventLocation("blue tag") -> 00:06
video_metadata -> duration=30s, fps=30, 270x480
video_clip_qa([4,8], "What is the man doing and wearing?")
video_clip_qa([4,8], "What color is the man's outer clothing?")
final D
```

**关键 observation**：第一个 clip 看到男子面对镜头讲话、穿深色外衣；第二个 clip 进一步确认是 blue jacket。

**解读**：这是 metadata 的目标用法。音频工具先给锚点 6 秒，planner 读取总时长后构造覆盖锚点的 `[4,8]` 合法窗口。真正提供答案证据的是 `video_clip_qa`，metadata 只负责边界/参数规划。

### 2. `Me4W36_lUcI-2`：检查接近视频末尾的时间窗

**问题**：speaker 说完 Loop Bravo 尚未 vented 时，画面采用什么视角？

**轨迹**：

```text
Audio_EventLocation("Loop Bravo") -> 00:07-00:08 and 00:25-00:26
video_metadata -> duration=30s, fps=29.97, 480x272
video_clip_qa([23,27], visual perspective?)
final C
```

**关键 observation**：clip 返回 elevated wide shot of mission control workers。

**解读**：25–26 秒靠近 30 秒结尾。metadata 给 planner 一个明确上界，使围绕事件构造的 `[23,27]` 不越界。这比中段时间窗更能体现 duration 的实际意义。

### 3. `d6b4OmUFt7I-1`：检查视频开头

**问题**：视频最开始出现的是哪组视听组合？

**轨迹**：

```text
audio_qa(first 10s) -> "We're in West Bank, BC ..."
video_metadata -> duration=30s, fps=23.976, 640x360
video_clip_qa([0,4], first visuals?)
final B
```

**关键 observation**：0–4 秒显示 Heavy Equipment Operator Training Centre 的详细标牌。

**解读**：planner 用 `[0,4]` 检查开头并与音频的 geographical location mention 对齐。这里 metadata 确认文件可读、总时长与合法边界，但由于 `[0,4]` 显然不会越界，其增量价值较小。

### 4. `XUWxQYmiBQY-1`：metadata 后多轮缩小搜索范围

**问题**：Obsidian Fury 静态画面首次出现时，同步的音频事件是什么？

**轨迹**：

```text
Audio_EventLocation("Obsidian Fury") -> 00:08-00:09, 00:26-00:29
video_metadata -> duration=30s, fps=29.97, 640x360
audio_ASR -> introduction at 00:05-00:11
video_clip_qa([0,12]) -> static shot around 6s
video_clip_qa([0,3]) -> opening animation/SFX
video_clip_qa([4,8]) -> static figure begins around 4s
audio_qa(first 4s) -> metallic/robotic SFX
final A
```

**解读**：metadata 提供完整时间轴边界，planner 随后从 `[0,12]` 粗查，细化到 `[0,3]` 和 `[4,8]`。不过“缩小范围”的智能来自 o3；metadata 本身没有给出 Obsidian Fury 出现的时间。

### 5. `Ec_lQgZ9wlg-1`：在整个 30 秒内做分段扫描

**问题**：Dr. Rajani 的 inset video 结束后紧接着出现什么视觉元素？

**轨迹（压缩）**：

```text
audio_global_caption / ASR / EventLocation / ASR
video_metadata -> duration=30s, fps=25, 640x360
video_global_qa -> 得到整体场景顺序
video_clip_qa([5,12])
video_clip_qa([12,16])
video_clip_qa([16,20]) -> inset 大约在窗口内 2 秒处消失
video_clip_qa([18,24]) -> presenter 手持产品继续讲解
video_clip_qa([20,28])
video_clip_qa([24,30])
final B
```

**解读**：这是 metadata 最明显的“搜索空间边界”案例。planner 知道总长 30 秒后，覆盖了从 5 秒到末尾的多个合法窗口，并把最后一个窗口精确截到 `[24,30]`。答案内容仍来自 global/clip QA，而不是 metadata。

### 6. `S8D4VF8Yl1g-2`：metadata 先给边界，音频工具再定位

**问题**：手机显示 DVD 的 eBay 价格后，画面如何转场？

**轨迹**：

```text
video_metadata -> duration=30s, fps=30, 640x360
Audio_EventLocation("eBay") -> 00:00-00:01 and 00:08-00:09
audio_ASR -> 更完整的语音时间线
video_clip_qa([6,12])
video_clip_qa([7,11])
final B
```

**关键 observation**：价格页面后切到一个男子直接对镜头讲话。

**解读**：metadata 提供全局上界，但真正决定 `[6,12]` / `[7,11]` 的主要依据是后面的 EventLocation 与 ASR。因此它有辅助作用，但不是定位来源。

### 7. `WkBkvlUoZpU-2`：调用合理，但问题本身已给时间窗

**问题**：0–10 秒内 zoom in 到 TV screen 的主要目的是什么？

**轨迹**：

```text
video_metadata -> duration=30s, fps=29.97, 640x360
audio_global_caption
video_clip_qa([0,10])
audio_qa(first 10s)
final D
```

**解读**：metadata 可验证 `[0,10]` 合法，但题目已经明确写出 0–10 秒，所以 planner 即使不调用 metadata 也能安全构造该窗口。属于“形式上被后续 clip 使用、信息上接近冗余”。

### 8. `0U_X5BIkYoA-1`：没有 clip consumer

**问题**：哪个社交媒体 logo 最先出现？

**轨迹**：

```text
video_metadata -> duration=30s, fps=30, 640x360
video_global_qa(which logo appears first?) -> Instagram at 00:08.500
final C
```

**解读**：后续是 whole-video 的 `video_global_qa`，不需要 planner 提供 `time_range`。duration/FPS/分辨率均未明显参与下一步参数，因此这次 metadata 调用基本冗余。

### 9. `7yY1OWjT4zA-1`：metadata 后仅做 global chronology

**问题**：视频时间线上哪个事件最先发生？

**轨迹**：

```text
video_metadata -> duration=30s, fps=30, 640x360
video_global_qa(describe main scenes chronologically)
  -> 00:00-00:04 blurred hands + "Conservation" title
final D
```

**解读**：与样本 8 相同，答案由 global QA 直接提供。metadata 只告诉 planner 视频有 30 秒，但没有被用于 clip 参数或最终推理中的任何可见事实。

### 10. `G8ywbM85GVE-1`：metadata 是最后一步，完全没有下游使用

**问题**：根据视听上下文可推断 Phil 的职业角色是什么？

**轨迹**：

```text
audio_global_caption
audio_qa
video_global_qa -> Phil 戴 THIESS 安全帽，身处大型工业设施
video_metadata -> duration=30s, fps=23.976, 640x360
final A
```

**解读**：调用 metadata 之前，audio/global video evidence 已足以回答；调用之后没有任何 tool call，planner 直接 final。这里没有下游组件消费 metadata，属于最明确的无效调用。

## 全量调用模式

对 766 个调用过 metadata 的样本，以第一次 `video_metadata` 为基准：

- 157 个样本把 metadata 作为第一个工具调用。
- 426 个样本下一步立刻调用 `video_clip_qa`。
- 106 个下一步调用 `video_global_qa`。
- 216 个下一步转向某种 audio 工具。
- 2 个下一步再次调用 `video_metadata`。
- 16 个调用 metadata 后直接 final。
- 675 个在 metadata 之后的某个时刻至少调用过一次 `video_clip_qa`。
- 91 个在 metadata 之后完全没有调用 `video_clip_qa`。

因此可以把作用概括为：

```text
video_metadata
  -> 提供容器级事实（尤其 duration）
  -> o3 planner 读取 observation
  -> planner 自己设计合法的 time_range
  -> video_clip_qa 才真正读取短片并产出语义证据
```

最有用的字段是 `duration_seconds`。`frame_count`、`fps`、`width`、`height` 在这 10 条轨迹里没有看到被显式用于推理；它们更像诊断信息。由于 DailyOmni 的这些 cut 大多是 30 秒，且很多题目或 audio 工具已经给出时间戳，metadata 的边际价值有限，planner 存在过度调用。

## 对这个 tool 的最终判断

1. **它不能独立解决任何语义问题**：不知道人物、动作、物体、台词或事件。
2. **它是参数规划/校验工具**：主要防止 `video_clip_qa` 的时间窗越界。
3. **消费者是 o3 planner，不是另一个 tool**：planner 通过轨迹 observation 间接把 duration 转化为下一次 tool call 的参数。
4. **在短且长度高度统一的 DailyOmni cut 上经常冗余**：特别是问题已经给出时间段、只需 global QA、或 metadata 后直接 final 的情况。
5. **在长视频或尾部事件上更有价值**：当总时长未知且需要围绕靠近结尾的时间点构造 clip 时，它能避免非法区间。

若要优化，可以让 runtime 在加载视频时自动把 duration 放入初始 sample context，或由 `video_clip_qa` 自己 clamp/校验时间窗。这样可保留安全性，同时省掉大量单独的 metadata tool calls。
