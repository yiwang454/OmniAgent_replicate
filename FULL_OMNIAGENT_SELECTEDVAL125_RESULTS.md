# Full OmniAgent control: selectedVal125 extraction

The Full OmniAgent (`C+Q+E+A+M`, all eight tools) control was extracted from
the two completed 1,197-question replication runs by exact question ID and
emitted in `daily_omni_cuts_selectedVal125.jsonl` order. Both extracted outputs
contain exactly 125 unique questions.

## Main results

All token, tool-call, and cost columns are means per question. Accuracy uses
all 125 selected questions as the denominator, so a terminal failure is wrong.
The two-run mean is descriptive only (`n=2`).

| Run | Extracted output | Accuracy | Input tok.† | Output thinking tok.† | Output non-thinking tok.† | Tool calls | Cost† | Latency | Failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full OmniAgent repeat 1 | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_repeat1_selectedVal125/output_test.jsonl) | **80.00% (100/125)** | 62.473k | 3.707k | 1.421k | 7.440 | $0.06971 | — | 0 |
| Full OmniAgent repeat 2 | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_repeat2_selectedVal125/output_test.jsonl) | **76.80% (96/125)** | 58.152k | 3.627k | 1.290k | 6.872 | $0.06640 | — | 1 |
| Two-run mean | — | **78.40%** | 60.312k | 3.667k | 1.355k | 7.156 | $0.06806 | — | 0.5 |

† The legacy trace has exact o3 planner usage but does not retain Gemini
`usageMetadata`. As in the sibling README's Full OmniAgent row O, Gemini input
is estimated from media duration/frame rate and prompt text; visible Gemini
output is estimated by text length. Hidden Gemini thinking is unavailable and
excluded, so “thinking” is the recorded o3 reasoning lower bound. Cost combines
exact recorded o3 usage with estimated Gemini usage. These original control
runs invoked `audio_global_caption` live; their caption cost is included.
Repeat 2's failed terminal turn has no recorded planner usage, so its reported
token/cost total excludes any unrecorded charge for that failed turn.

Per-question latency was not persisted in either legacy output/trace and is not
recoverable, so it is reported as unavailable rather than inferred from file
timestamps.

## Exact recorded planner usage and estimated perception usage

| Run | Planner input exact | Planner cached input exact | Planner output exact | Planner reasoning exact | Gemini calls | Gemini input estimated | Gemini visible output estimated | Total estimated cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Repeat 1 | 3,415,305 | 3,016,448 | 551,288 | 463,424 | 860 | 4,393,797 | 89,703 | $8.714255 |
| Repeat 2 | 3,084,960 | 2,715,008 | 535,395 | 453,376 | 798 | 4,184,001 | 79,254 | $8.300271 |

“Gemini input estimated” is text + video + audio input. The machine-readable
summaries retain the modality breakdown and separate planner/Gemini cost:

- [Repeat 1 summary](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_repeat1_selectedVal125/metrics_summary.json)
- [Repeat 2 summary](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_repeat2_selectedVal125/metrics_summary.json)

## Tool calls

| Run | Caption | Audio QA | Video global | Video clip | EventList | EventLocation | ASR | Metadata | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Repeat 1 | 77 | 150 | 61 | 402 | 44 | 75 | 51 | 70 | 930 |
| Repeat 2 | 74 | 146 | 66 | 328 | 37 | 83 | 64 | 61 | 859 |

## Run-to-run comparison and failure

The accuracy spread is 3.2 percentage points. Of the 125 questions, 88 were
correct in both runs, 12 only in repeat 1, 8 only in repeat 2, and 17 wrong in
both. The selected predictions agree on 102/125 questions.

Repeat 2 has one terminal failure, `TF9I1GxNdJQ-2`: `Audio_EventList` returned
no answer text with `finishReason=MAX_TOKENS`. Repeat 1 has no selected-subset
failure.

## Artifacts and reproduction

Each extraction directory contains the unchanged selected result rows,
`per_question_metrics.jsonl`, and `metrics_summary.json`. The summary records
the source/subset SHA-256 hashes and output hash.

```bash
python3 scripts/extract_full_omniagent_selectedval.py \
  --source /mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_repeat1/output_test.jsonl \
  --subset /mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_selectedVal125.jsonl \
  --manifest /mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_v3_sepAV.jsonl \
  --output-dir /mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_repeat1_selectedVal125
```

Change `repeat1` to `repeat2` in both paths for the second extraction.
