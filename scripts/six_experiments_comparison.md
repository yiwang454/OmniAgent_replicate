# 六个实验的题均资源与结果汇总

所有费用均为 Gemini + GPT/o3 的合计，并使用各实验记录中的公开 API Standard / Paid Tier 价格口径。`OmniAgent_repeat1` 的 Gemini token 没有完整 usage metadata，input/output 和费用为此前报告的媒体时长、帧率及文本长度估算，且不含未知的 Gemini hidden thinking。`maxturns6` 采用无预热 cache 口径，已计入 1,197 个 caption cache 的原始 Gemini token 与费用。Accuracy 按要求留空。

| Experiment | input tokens | output tokens | latency | cost (USD) | Accuracy |
|---|---:|---:|---:|---:|---|
| GPT-4.1 · None (no GEPA) · None | 13,303.92 | 853.03 | 16.77 s | $0.01358 | 71.68 |
| GPT-4.1 · GEPA G0 · Planner | 28,602.56 | 1,989.31 | | $0.03167 | 75.61 |
| GPT-5.4 · None (no GEPA) · None | 16,586.79 | 1,320.83 | 18.36 s | $0.01950 | 76.94 |
| GPT-5.4 · GEPA G0 · Planner | 17,228.18 | 807.49 | | $0.02742 | 74.44 |
| o3 + Gemini 2.5 Flash · OmniAgent_repeat1 | 62,693.55 | 4,900.16 | 71 s | $0.06743 | 77.53* |
| o3 + Gemini 2.5 Flash · ReAct + omni_clip_caption (max turns 6) | 14,356.54 | 1,738.53 | 27.31 s | $0.01928 | 78.36 |


四个 GPT 实验的 token、latency 与 cost 取自 [`g0_gpt_token_usage_and_latency.md`](/mnt/ceph_rbd/workspace/avqa_project/general_scripts/react-agent-avqa-optimize/scripts/experiment_records/g0_gpt_token_usage_and_latency.md)。两个 OmniAgent 实验的 token 和成本取自本目录的 [`output_test_question_usage.md`](output_test_question_usage.md) 与 [`output_test_cost_estimate.md`](output_test_cost_estimate.md)。前三个没有完整全轮 latency 记录的实验按要求留空。
