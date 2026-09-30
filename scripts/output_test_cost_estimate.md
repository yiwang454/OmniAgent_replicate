# `output_test.jsonl` 每题模型费用估算

- 数据：`/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_repeat1/output_test.jsonl`，1,197 个 question sample。逐题 turn/token 数见 [usage 报告](output_test_question_usage.md)。
- 价格口径：公开 API Standard / Paid Tier，单位 USD / 1M tokens；仅估算模型 token 费用。实际经 Gemini-compatible gateway 的结算单价未核实。

## 价格

| 模型 | Input text | Input image/video | Input audio | Cached input | Output text（含 thinking） |
|---|---:|---:|---:|---:|---:|
| Gemini 2.5 Flash | $0.30 | $0.30 | $1.00 | 本估算未用缓存 | $2.50 |
| o3 | $2.00 | — | — | $0.50 | $8.00 |

价格来源：[Google Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing)、[OpenAI o3 model pricing](https://developers.openai.com/api/docs/models/o3)。o3 planner 只接收文本工具结果。

## 平均费用

| 项目 | 全部题目合计 | 每题平均 |
|---|---:|---:|
| o3：按已记录 token 与 cached input 计费 | $61.0916 | $0.0510 |
| Gemini：可见文本输出 + 估算媒体输入；暂不计 hidden thinking | $19.6209 | $0.0164 |
| 两模型合计：暂不计 Gemini hidden thinking | $80.7125 | $0.0674 |
| 两模型合计：假设 Gemini 每次额外 500 thinking tokens | $90.6262 | $0.0757 |
| 两模型合计：假设 Gemini 每次额外 1,000 thinking tokens | $100.5400 | $0.0840 |

o3 费用的计算式：`(uncached_input × 2 + cached_input × 0.5 + output × 8) / 1,000,000`。
记录中 o3 input 为 30,053,445 tokens，其中 cached 为 26,419,456；output 为 5,076,734。
Gemini 共 7,931 次成功调用，重试记录 3 次；重试失败请求可能产生的费用未计。

Gemini 基础估算拆分：视频输入约 3,749 万 tokens（$11.2476）、音频输入约 593 万 tokens（$5.9311）、文本输入约 157 万 tokens（$0.4703）、可见文本输出约 79 万 tokens（$1.9719）。这些 token 数均为估算值。

## Gemini 估算方法与限制

- 按工具决定主要模态：audio 工具输入按音频，video 工具输入按视频；`video_global_qa` 使用原始视频，若其文件包含音轨，则另计音频；`video_clip_qa` 的代码导出无音轨片段。文本 prompt 仍单独计价，没有将整个混合输入粗略归给一个模态。
- 用本地视频实际时长和 `video_clip_qa.time_range` 计算送入的片段时长；`video_global_qa` 为 2 fps，`video_clip_qa` 为 5 fps。视频输入按 `263 × fps × 秒数` tokens，音频按 `32 × 秒数` tokens。依据 [Google token counting 文档](https://ai.google.dev/gemini-api/docs/tokens)；具体服务端帧采样和视频 tokenization 可能不同。
- 有 19 次旧 trace 的片段范围起止相同且调用成功；按 5 fps 的一帧（0.2 秒）估算。输入/输出文本按约 4 字符/token 估算。`perception_calls.response` 只有可见文本，不含 Gemini hidden thinking；上表额外列出每次 500 和 1,000 thinking tokens 的敏感性场景，均非实测。
- Gemini gateway 原始响应的 `usageMetadata` 没有写入 JSONL；无法从现有 trace 还原精确的 prompt modality breakdown、output thinking tokens、缓存或实际账单。未来应记录 `usageMetadata.promptTokensDetails`、`promptTokenCount`、`candidatesTokenCount`、`thoughtsTokenCount`、`cachedContentTokenCount` 及每次请求的 model/错误/重试状态。
- `QtKT3q7xB4M-2` 只有一个 o3 error turn，未记录 usage；表中给该题的已知 token 费用 0，但真实账单不能据此断言为 0。其余题目的 o3 费用可按 trace 与公开单价精确计算。

## `dspy_free_react_o3_gemini25flash_omni_clip_caption_maxturns6` 实验

数据来源：`/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_free_react_o3_gemini25flash_omni_clip_caption_maxturns6/output_test.jsonl`。共有 1,197 个 question sample。价格沿用本报告前文的 Gemini 2.5 Flash 和 o3 Standard / Paid Tier 公开 API 单价。

### 题均 token 与 tool calls

| 指标 | 合计 | 每题平均 |
|---|---:|---:|
| o3 planner 调用数（含解析重试／fallback） | 3,370 | 2.8154 |
| o3 input tokens | 4,844,553 | 4,047.25 |
| o3 cached input tokens | 49,536 | 41.38 |
| o3 output tokens（含 reasoning） | 1,037,765 | 866.97 |
| Gemini 实时调用数 | 732 | 0.6115 |
| Gemini 实时 input tokens | 6,115,166 | 5,108.74 |
| Gemini 实时 output tokens（可见输出 + thinking） | 315,569 | 263.63 |
| 实际 tool calls | 1,929 | 1.6115 |
| Turns（含 final decision） | 3,126 | 2.6115 |
| Turns excluding final decision | 1,929 | 1.6115 |

Gemini 实时输入的题均细分为：video 3,777.44 tokens、audio 367.37 tokens、text 127.15 tokens；题均 output 中可见文本为 79.85 tokens，thinking 为 183.79 tokens。实际调用包括 683 次 `ask_perception`、40 次 `omni_clip_caption` 和 9 次未命中/额外的 `ask_caption`。

### 题均费用（USD）

| 费用口径 | 全部题目合计 | 每题平均 |
|---|---:|---:|
| o3（区分普通与 cached input） | $17.9169 | $0.01497 |
| Gemini：本次运行的实时调用 | $2.6308 | $0.00220 |
| 本次运行的边际模型费用合计 | $20.5477 | $0.01717 |
| Gemini caption cache 原始生成成本 | $2.5339 | $0.00212 |
| 含 caption cache 原始生成成本的合计 | $23.0816 | $0.01928 |

“本次运行的边际模型费用”将 1,197 个首轮 `ask_caption` 视为 cache hit，因此不重复计费；这是运行这个实验时新增的 API 用量。最后一行将这些 cache 内容首次生成时的 Gemini 用量也归入本实验，适合比较“无预热 cache”的端到端成本。

费用公式：o3 为 `(普通 input × $2 + cached input × $0.50 + output × $8) / 1M`。Gemini 为 `(text/video input × $0.30 + audio input × $1.00 + (visible output + thinking) × $2.50) / 1M`。Gemini 费用使用 trace 内逐模态 `promptTokensDetails` 和 `thoughtsTokenCount`，无需用时长或字符数近似。

### Latency

题均 latency：**27.31 s/question**。该值来自实验运行日志，结果 JSONL 和每题 trace 本身仍未记录请求开始／结束时间、单调用 elapsed time 或整个样本 wall-clock duration，因此不能据此拆分 planner、captioner 与工具调用各自的耗时。
