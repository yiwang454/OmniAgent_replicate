# OmniAgent tool-selection ablation: selectedVal125 results

Generated from the completed runs under
`/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125`
on 2026-10-09. Every experiment contains 125 rows and 125 unique question IDs.

## Main results

All token, tool-call, and cost columns are averages per question. Accuracy is
full-set accuracy (`correct / 125`), so terminal failures count as incorrect.
`Latency` is the mean persisted per-question `retry_adjusted_seconds`; `(n/125)`
shows latency coverage. Configurations containing C use the existing warm
`audio_global_caption` cache, while `−C` has no caption-cache access.

| Experiment | Output | Enabled groups | Accuracy | Input tok. | Output thinking tok. | Output non-thinking tok. | Tool calls | Live cost | Latency | Failures |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| −A | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125/minus_a/output_test.jsonl) | C+Q+E+M | 76.00% (95/125) | 29.171k | 3.143k | 0.914k | 7.224 | $0.05434 | 69.23 s (124/125) | 1 |
| −E | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125/minus_e/output_test.jsonl) | C+Q+A+M | 79.20% (99/125) | 33.011k | 4.163k | 1.188k | 8.432 | $0.06720 | 80.62 s (124/125) | 1 |
| **−A−E** | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125/minus_a_e/output_test.jsonl) | C+Q+M | **80.00% (100/125)** | 29.217k | 3.359k | 0.958k | 7.848 | $0.05665 | 72.51 s (125/125) | 0 |
| −C | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125/minus_c/output_test.jsonl) | Q+E+A+M | 73.60% (92/125) | 25.935k | 3.074k | 1.008k | 6.480 | $0.05206 | 65.56 s (124/125) | 1 |
| −Q | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125/minus_q/output_test.jsonl) | C+E+A+M | 63.20% (79/125) | 15.064k | 3.376k | 0.679k | 4.960 | $0.04458 | 54.14 s (125/125) | 0 |
| −M | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125/minus_m/output_test.jsonl) | C+Q+E+A | 76.80% (96/125) | 28.893k | 3.786k | 1.056k | 7.144 | $0.05964 | 76.68 s (124/125) | 1 |
| Minimal | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125/minimal/output_test.jsonl) | C+Q | 72.00% (90/125) | 21.391k | 2.717k | 0.841k | 6.200 | $0.04466 | 61.71 s (124/125) | 1 |

Group definitions: C = `audio_global_caption`; Q = `audio_qa`,
`video_global_qa`, `video_clip_qa`; E = `Audio_EventList`,
`Audio_EventLocation`; A = `audio_ASR`; M = `video_metadata`.

## Full-tool control extracted from the replication runs

For the `C+Q+E+A+M` control on exactly the same 125 questions, repeat 1 scores
**80.00% (100/125)** and repeat 2 scores **76.80% (96/125)**. The corresponding
token, cost, tool-call, failure, and provenance details are in
[`FULL_OMNIAGENT_SELECTEDVAL125_RESULTS.md`](FULL_OMNIAGENT_SELECTEDVAL125_RESULTS.md).

The control is a legacy trace: o3 usage is exact, while Gemini token/cost fields
use the same media-duration approximation as README row O. It also generated
captions live, unlike the ablation runs' warm caption cache. For those reasons,
accuracy and tool-call counts are directly comparable, while legacy control
token/cost figures are marked as estimates in the control report.

## Exact aggregate usage and cost

`Live input` is planner input plus live Gemini prompt input. `Thinking output`
is o3 reasoning plus Gemini thinking. `Non-thinking output` is planner output
excluding o3 reasoning, plus Gemini candidate output. The cache read itself
does not add live model tokens.

| Experiment | Live input tok. | Thinking output tok. | Non-thinking output tok. | Planner cached input tok. | Tool calls | Live cost total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| −A | 3,646,354 | 392,832 | 114,246 | 1,907,328 | 903 | $6.792574 |
| −E | 4,126,413 | 520,384 | 148,559 | 2,263,552 | 1,054 | $8.400207 |
| −A−E | 3,652,138 | 419,904 | 119,792 | 1,893,376 | 981 | $7.081210 |
| −C | 3,241,930 | 384,192 | 125,997 | 1,600,000 | 810 | $6.507037 |
| −Q | 1,882,968 | 422,016 | 84,919 | 775,296 | 620 | $5.572368 |
| −M | 3,611,607 | 473,280 | 132,025 | 1,835,648 | 893 | $7.454385 |
| Minimal | 2,673,922 | 339,584 | 105,139 | 1,167,232 | 775 | $5.582035 |

## Tool-call totals

The final planner-only answer round is excluded. A dash means the tool was not
available in that configuration.

| Experiment | C caption | audio QA | video global | video clip | EventList | EventLocation | ASR | Metadata | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| −A | 82 | 230 | 70 | 312 | 46 | 97 | — | 66 | 903 |
| −E | 98 | 249 | 68 | 475 | — | — | 93 | 71 | 1,054 |
| −A−E | 101 | 390 | 72 | 345 | — | — | — | 73 | 981 |
| −C | — | 143 | 64 | 349 | 41 | 62 | 77 | 74 | 810 |
| −Q | 104 | — | — | — | 64 | 131 | 180 | 141 | 620 |
| −M | 75 | 175 | 74 | 390 | 38 | 72 | 69 | — | 893 |
| Minimal | 97 | 324 | 81 | 273 | — | — | — | — | 775 |

## Caption-cache accounting

The main table reports marginal live-run cost, matching the sibling README's
main-table convention. The cache retains exact original-generation usage only
for entries generated during cache completion; older entries extracted from
the original OmniAgent trace do not contain source token/latency metadata.
Consequently, only the known source cost is shown and no incomplete “total
cold-cache cost” is presented as if it were complete.

| Experiment | Cache calls | Unique entries read | Entries with source usage/latency | Entries with unknown provenance | Known original generation cost | Warm lookup total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| −A | 82 | 82 | 15 | 67 | $0.020000 | 0.066 s |
| −E | 98 | 96 | 23 | 73 | $0.032122 | 0.063 s |
| −A−E | 101 | 101 | 27 | 74 | $0.036339 | 0.075 s |
| −C | 0 | 0 | 0 | 0 | $0 | 0 s |
| −Q | 104 | 104 | 33 | 71 | $0.044460 | 0.148 s |
| −M | 75 | 73 | 10 | 63 | $0.013735 | 0.036 s |
| Minimal | 97 | 95 | 20 | 75 | $0.029347 | 0.078 s |

## Failures and accuracy denominator

Five experiment outputs contain one terminal failure each. These rows remain
in the 125-question denominator and are counted as incorrect in the main table.

| Experiment | Failed question | Failure | Answered accuracy (diagnostic only) |
| --- | --- | --- | ---: |
| −A | `6_Foh42T5XM-1` | Gemini 400 `PROHIBITED_CONTENT` in `video_clip_qa` | 76.61% (95/124) |
| −E | `HProiNnmGwI-1` | Gemini 400 invalid argument in `video_clip_qa` | 79.84% (99/124) |
| −C | `6_Foh42T5XM-1` | Gemini 400 `PROHIBITED_CONTENT` in `video_clip_qa` | 74.19% (92/124) |
| −M | `6_Foh42T5XM-1` | Gemini 400 `PROHIBITED_CONTENT` in `video_clip_qa` | 77.42% (96/124) |
| Minimal | `6_Foh42T5XM-1` | Gemini 400 `PROHIBITED_CONTENT` in `video_clip_qa` | 72.58% (90/124) |
| −A−E | — | none | 80.00% (100/125) |
| −Q | — | none | 63.20% (79/125) |

## Metric definitions and interpretation

- **Input tok.**: planner input plus live Gemini tool input, including cached
  planner input in the token count. Billing discounts for planner cached input
  are applied in the cost calculation.
- **Output thinking tok.**: o3 reasoning tokens plus Gemini thinking tokens.
  Gemini reported zero thinking tokens in these runs.
- **Output non-thinking tok.**: planner output excluding o3 reasoning, plus
  Gemini visible candidate output.
- **Live cost**: o3 at $2.00/M uncached input, $0.50/M cached input and $8.00/M
  output; Gemini 2.5 Flash at $0.30/M text/image/video input, $1.00/M audio input
  and $2.50/M output. Historical caption-cache generation is excluded.
- **Latency**: mean non-null retry-adjusted question latency. It removes failed
  request duration and retry backoff and is independent of worker concurrency.
- **Accuracy**: exact multiple-choice full-set accuracy over all 125 questions.

On this subset, `−A−E` has the highest full-set accuracy (80.0%) and also has
complete results. `−Q` has the lowest live cost and latency among the
cache-enabled runs, but its accuracy falls to 63.2%, the largest degradation in
the ablation set. `−E` is the most expensive and tool-intensive configuration.
These comparisons describe this fixed 125-question subset and should not be
treated as confidence-tested estimates of full-dataset effects.

## Repeat 2 results

All seven outputs contain the same 125 unique `selectedVal125` question IDs as
repeat 1. Columns use the same definitions as the main table; failures remain
in the 125-question accuracy denominator.

| Experiment | Output | Accuracy | Input tok. | Output thinking tok. | Output non-thinking tok. | Tool calls | Live cost | Latency | Failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| −A | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125_repeat2/minus_a/output_test.jsonl) | 80.00% (100/125) | 30.158k | 3.033k | 0.960k | 7.368 | $0.05570 | 77.56 s (124/125) | 1 |
| −E | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125_repeat2/minus_e/output_test.jsonl) | 77.60% (97/125) | 28.534k | 3.391k | 1.072k | 7.264 | $0.05797 | 78.04 s (124/125) | 1 |
| −A−E | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125_repeat2/minus_a_e/output_test.jsonl) | 80.00% (100/125) | 30.765k | 3.530k | 1.054k | 7.952 | $0.06011 | 84.75 s (124/125) | 1 |
| −C | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125_repeat2/minus_c/output_test.jsonl) | 76.00% (95/125) | 26.367k | 3.099k | 1.031k | 6.440 | $0.05354 | 74.57 s (124/125) | 1 |
| −Q | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125_repeat2/minus_q/output_test.jsonl) | 64.00% (80/125) | 17.347k | 3.762k | 0.714k | 5.360 | $0.05030 | 67.92 s (125/125) | 0 |
| −M | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125_repeat2/minus_m/output_test.jsonl) | 76.80% (96/125) | 27.067k | 3.365k | 0.989k | 6.648 | $0.05659 | 78.85 s (122/125) | 3 |
| Minimal | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125_repeat2/minimal/output_test.jsonl) | 76.00% (95/125) | 24.641k | 3.068k | 0.951k | 7.008 | $0.05150 | 75.13 s (125/125) | 0 |

## Repeat 3 results

| Experiment | Output | Accuracy | Input tok. | Output thinking tok. | Output non-thinking tok. | Tool calls | Live cost | Latency | Failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| −A | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125_repeat3/minus_a/output_test.jsonl) | 76.80% (96/125) | 34.072k | 3.728k | 1.076k | 8.408 | $0.06182 | 87.85 s (124/125) | 1 |
| −E | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125_repeat3/minus_e/output_test.jsonl) | 79.20% (99/125) | 31.128k | 3.788k | 1.129k | 7.864 | $0.06160 | 84.63 s (123/125) | 2 |
| −A−E | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125_repeat3/minus_a_e/output_test.jsonl) | 76.80% (96/125) | 31.176k | 3.453k | 1.052k | 8.120 | $0.05925 | 81.91 s (125/125) | 0 |
| −C | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125_repeat3/minus_c/output_test.jsonl) | 76.80% (96/125) | 26.488k | 3.069k | 1.020k | 6.416 | $0.05216 | 73.72 s (124/125) | 1 |
| −Q | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125_repeat3/minus_q/output_test.jsonl) | 59.20% (74/125) | 14.507k | 3.143k | 0.636k | 4.696 | $0.04230 | 58.80 s (125/125) | 0 |
| −M | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125_repeat3/minus_m/output_test.jsonl) | 80.80% (101/125) | 23.234k | 2.939k | 0.942k | 5.912 | $0.04842 | 70.71 s (123/125) | 2 |
| Minimal | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125_repeat3/minimal/output_test.jsonl) | 74.40% (93/125) | 25.340k | 3.234k | 0.931k | 6.840 | $0.05191 | 77.27 s (124/125) | 1 |

## Three-repeat mean and standard deviation

Each cell is the arithmetic mean across repeats 1–3 plus/minus the **sample
standard deviation** (`ddof=1`, `n=3`) of the per-repeat value. Latency uses
each run's persisted retry-adjusted mean; the coverage shown in the individual
tables varies when a run terminates before producing latency for a question.

| Experiment | Accuracy | Input tok. | Output thinking tok. | Output non-thinking tok. | Tool calls | Live cost | Latency | Failures/run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| −A | 77.60% ± 2.12 pp | 31.133k ± 2.592k | 3.301k ± 0.374k | 0.983k ± 0.083k | 7.667 ± 0.646 | $0.05729 ± $0.00399 | 78.21 ± 9.33 s | 1.00 ± 0.00 |
| −E | 78.67% ± 0.92 pp | 30.891k ± 2.248k | 3.781k ± 0.386k | 1.130k ± 0.058k | 7.853 ± 0.584 | $0.06226 ± $0.00465 | 81.10 ± 3.32 s | 1.33 ± 0.58 |
| **−A−E** | **78.93% ± 1.85 pp** | 30.386k ± 1.033k | 3.448k ± 0.086k | 1.022k ± 0.055k | 7.973 ± 0.137 | $0.05867 ± $0.00180 | 79.72 ± 6.41 s | 0.33 ± 0.58 |
| −C | 75.47% ± 1.67 pp | 26.264k ± 0.291k | 3.081k ± 0.016k | 1.020k ± 0.012k | 6.445 ± 0.032 | $0.05258 ± $0.00083 | 71.28 ± 4.97 s | 1.00 ± 0.00 |
| −Q | 62.13% ± 2.57 pp | 15.639k ± 1.505k | 3.427k ± 0.312k | 0.676k ± 0.039k | 5.005 ± 0.334 | $0.04573 ± $0.00412 | 60.29 ± 7.01 s | 0.00 ± 0.00 |
| −M | 78.13% ± 2.31 pp | 26.398k ± 2.888k | 3.364k ± 0.423k | 0.996k ± 0.057k | 6.568 ± 0.620 | $0.05488 ± $0.00580 | 75.41 ± 4.21 s | 2.00 ± 1.00 |
| Minimal | 74.13% ± 2.01 pp | 23.791k ± 2.107k | 3.006k ± 0.264k | 0.907k ± 0.058k | 6.683 ± 0.426 | $0.04936 ± $0.00407 | 71.37 ± 8.43 s | 0.67 ± 0.58 |

Repeat 2/3 failure details remain in each configuration's
`metrics_summary.json`. Most terminal errors are Gemini 400 responses for
specific media/tool requests; they are included in the reported full-set
accuracy rather than excluded.
