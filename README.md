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
file of a sibling repository.

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
For a smoke test, add `--limit 1`; to run selected cuts, repeat
`--sample-id QUESTION_ID`.

Benchmark runs resume by default. Existing per-question JSON trajectories are
used to rebuild a missing or interrupted aggregate; completed non-error samples
are skipped, while explicit `[ERROR]` and empty responses are rerun. The
original `Agent stopped due to max iterations.` outcome is preserved rather
than silently rerun. Use `--no-resume` only when intentionally starting the
entire output directory again.

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
