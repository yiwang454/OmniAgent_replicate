# OmniAgent: Active Perception Agent for Omnimodal Audio-Video Understanding

[Keda Tao](https://kd-tao.github.io/), [Wenjie Du](https://kurt232.github.io/), [Bohan Yu](), [Weiqiang Wang](), [Jian liu](), [Huan Wang](https://huanwang.tech/), "OmniAgent: Active Perception Agent for Omnimodal Audio-Video Understanding"

[[Paper](https://arxiv.org/abs/2512.23646)]

#### 🔥🔥🔥 News

- **2026-03-11**: **The code is released!**
- **2025-12-30**: The paper is released.
- **2025-12-28:** This repo is released.


![overview](figures/teaser.png)


> **Abstract:** Omnimodal large language models have made significant strides in unifying audio and visual modalities; however, they often lack the fine-grained cross-modal understanding and have difficulty with multimodal alignment. To address these limitations, we introduce OmniAgent, a fully audio-guided active perception agent that dynamically orchestrates specialized tools to achieve more fine-grained audio-visual reasoning. Unlike previous works that rely on rigid, static workflows and dense frame-captioning, this paper demonstrates a paradigm shift from passive response generation to active multimodal inquiry. OmniAgent employs dynamic planning to autonomously orchestrate tool invocation on demand, strategically concentrating perceptual attention on task-relevant cues. Central to our approach is a novel coarse-to-fine audio-guided perception paradigm, which leverages audio cues to localize temporal events and guide subsequent reasoning. Extensive empirical evaluations on three audio-video understanding benchmarks demonstrate that OmniAgent achieves state-of-the-art performance, surpassing leading open-source and proprietary models by substantial margins of 10% - 20% accuracy.

## ⚒️ TODO

* [x] Release code 
* [x] Release paper 
* [ ] Build a Gradio demo
* [ ] Support more models or API

## Install

Install our codebase:
```bash
git https://github.com/KD-TAO/OmniAgent.git
cd OmniAgent
```
Create the project-local uv environment and install the pinned dependencies:

```bash
python -m pip install --user uv  # only needed if uv is not installed yet
python -m uv venv .omniagent_venv --python 3.11
python -m uv pip install --python .omniagent_venv/bin/python -r requirements.txt
```

The virtual environment is deliberately stored at `.omniagent_venv/` and is
ignored by Git. Run commands through it directly (or activate it with
`source .omniagent_venv/bin/activate`).

## Quick Start

Create a local environment file (it is ignored by Git):

```bash
cp .env.example .env
```

Then set the ELM and Gemini credentials in `.env`:

```bash
ELM_API_KEY=...
GEMINI_BASE_URL=https://api.apiplus.org
GEMINI_API_KEY=...
```

The central reasoner keeps the repository's original model name (`o3` by
default, configurable with `BRAIN_MODEL`) and uses the ELM key through the
OpenAI SDK default endpoint. No custom OpenAI base URL is injected. Every
audio/video MLLM tool is fixed to `gemini-2.5-flash` and uses the same Bearer
token `generateContent` endpoint as `react-agent-avqa-optimize`.

OmniAgent loads credentials from its own `.env` only; it never reads the `.env`
file of a sibling repository by default. For an explicit one-off credential
source, set `OMNIAGENT_ENV_FILE=/absolute/path/to/.env`; values from that file
override inherited credential variables for that process.

The repository also requires the `ffmpeg` and `ffprobe` command-line programs.
They are used for local media extraction/inspection and do not require an API
key or GPU.

`main.py` runs the complete DailyOmni benchmark by default. It reads
`/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_v3.jsonl`, writes
one `<question-id>.json` rollout per sample, and continuously updates the
required aggregate artifact `<output-dir>/output_test.jsonl`. Each rollout
contains the planner prompts/responses, tool inputs/observations, and Gemini
perception prompts/responses.

```bash
.omniagent_venv/bin/python main.py --output-dir output
.omniagent_venv/bin/python main.py --output-dir output --no-print-steps
.omniagent_venv/bin/python main.py --example
.omniagent_venv/bin/python main.py --video_path YOUR_VIDEO --question "YOUR_Q"
```

Step-by-step terminal output is enabled by default. Use `--no-print-steps` to
disable LangChain's verbose rollout while retaining the complete JSON trace.
Benchmark samples run with four concurrent workers by default; use
`--workers N` to change this (for example, `--workers 1` for serial execution).
For a smoke test, add `--limit 1`; to run selected cuts, repeat
`--sample-id QUESTION_ID`.

Benchmark runs resume by default. Existing per-question JSON trajectories are
used to rebuild a missing or interrupted aggregate; completed non-error samples
are skipped, while explicit `[ERROR]` and empty responses are rerun. The
original `Agent stopped due to max iterations.` outcome is preserved rather
than silently rerun. Use `--no-resume` only when intentionally starting the
entire output directory again.

AVUT and WorldSense launchers run four question workers, log to `run.log`, and
write an aggregate latency/token report after all expected rows complete:

```bash
scripts/run_omniagent_avut.sh --no-print-steps
scripts/run_omniagent_worldsense.sh --no-print-steps
```

Every new per-question result contains `question_data.latency` with wall-clock,
retry-adjusted, and successful API-call durations, plus successful planner and
perception call counts. Retry-adjusted duration follows the sibling
`react-agent-avqa-optimize` convention: failed API attempts and the backoff
before their retry are removed; it is `null` for failed questions. The result
also contains `question_data.token_usage`, aggregating exact brain/planner usage
and Gemini `usageMetadata`. Each Gemini call retains its raw `usage_metadata`
inside `turn_trace[*].perception_calls`.

To validate and summarize either a completed or partial output manually:

```bash
.omniagent_venv/bin/python scripts/aggregate_inference_metrics.py \
  /path/to/output_test.jsonl \
  --output-json /path/to/inference_metrics_summary.json
```

## Tool-selection ablations (DailyOmni selectedVal125)

Completed 125-question results and metric definitions are summarized in
[`TOOL_ABLATION_SELECTEDVAL125_RESULTS.md`](TOOL_ABLATION_SELECTEDVAL125_RESULTS.md).
The same 125 questions extracted from both completed Full OmniAgent replication
runs are summarized in
[`FULL_OMNIAGENT_SELECTEDVAL125_RESULTS.md`](FULL_OMNIAGENT_SELECTEDVAL125_RESULTS.md).

The ablation registry lives in `configs/tool_ablation/`. Each experiment JSON
declares its enabled groups/tools and points to a prompt template derived from
the verbatim base template in `configs/tool_ablation/prompts/base.txt`.
Unavailable-tool guidance is removed while the rest of the prompt is preserved.

The seven resumable launchers use questions/labels from
`daily_omni_cuts_selectedVal125.jsonl` (125 rows) and resolve the corresponding
silent-video and WAV sources by question ID from
`daily_omni_cuts_v3_sepAV.jsonl`. They retain the default planner, high reasoning
effort, agent loop, tool implementations, and `max_iterations=30`.

```bash
scripts/run_ablation_minus_a.sh --no-print-steps
scripts/run_ablation_minus_e.sh --no-print-steps
scripts/run_ablation_minus_a_e.sh --no-print-steps
scripts/run_ablation_minus_c.sh --no-print-steps
scripts/run_ablation_minus_q.sh --no-print-steps
scripts/run_ablation_minus_m.sh --no-print-steps
scripts/run_ablation_minimal.sh --no-print-steps
```

To run repeat 2 and repeat 3, with all seven ablations parallel within each
repeat and the two repeats serialized:

```bash
scripts/run_all_tool_ablation_selectedval125_repeats2_3.sh
```

This defaults to four question workers per experiment (28 maximum concurrent
question workers). To reduce load while retaining seven-way experiment
parallelism, set, for example, `WORKERS_PER_EXPERIMENT=1`. The launcher is
resumable and writes to `OmniAgent_tool_ablation_selectedVal125_repeat2` and
`OmniAgent_tool_ablation_selectedVal125_repeat3` under the DailyOmni data root.

Runs resume by default. Override `OUTPUT_ROOT`, `WORKERS`, or
`CAPTION_CACHE_DIR` through environment variables when needed. Configurations
containing group C use the existing question-scoped `audio_global_caption`
cache; `minus_c` cannot accept a caption cache. Original cache-generation token
usage and latency are retained when present. Older trace-derived cache entries
that predate exact usage/latency logging are reported as unknown rather than
estimated.

For a five-question smoke test across all seven configurations:

```bash
OUTPUT_ROOT=/path/to/smoke-output scripts/run_all_tool_ablation_smoke.sh
```

Each experiment directory contains resumable per-question JSON files,
`output_test.jsonl`, `run_config.json`, `metrics_summary.json`,
`accuracy_summary.json`, and `run.log`. Per-question metrics include correctness,
tool-call counts, exact planner/perception input/output/reasoning tokens,
estimated model cost, warm-cache lookup/runtime, available cold-cache source
generation cost/latency, and structured failure details.

To inspect or recoverably remove stale explicit-error histories before a
resume, run the cleanup command without `--apply` first. Applied removals are
moved under `.removed_error_history/` rather than permanently deleted.

```bash
.omniagent_venv/bin/python scripts/remove_error_question_history.py \
  /path/to/output-dir

.omniagent_venv/bin/python scripts/remove_error_question_history.py \
  /path/to/output-dir --apply
```

The Gemini-compatible gateway is retried for transient 429/5xx responses and
its intermittent `Part.data` oneof 400. The latter retry uses the equivalent
text-first/media-second Gemini payload. Defaults are six attempts with capped
exponential backoff; tune them with `GEMINI_MAX_RETRIES`,
`GEMINI_RETRY_DELAY_S`, and `GEMINI_RETRY_MAX_DELAY_S`.

Evaluate a completed or partial output while excluding explicit errors and
empty responses:

```bash
.omniagent_venv/bin/python scripts/evaluate_dailyomni_output.py \
  /path/to/output-dir \
  --json-output /path/to/output-dir/evaluation_success_only.json
```

The report separates accuracy among parseable successful answers from the
strict accuracy that excludes only errors/empty responses and therefore counts
non-empty responses without a valid answer as incorrect.

## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{tao2025omniagent,
  title={OmniAgent: Audio-Guided Active Perception Agent for Omnimodal Audio-Video Understanding},
  author={Tao, Keda and Du, Wenjie and Yu, Bohan and Wang, Weiqiang and Liu, Jian and Wang, Huan},
  journal={arXiv preprint arXiv:2512.23646},
  year={2025}
}
```
