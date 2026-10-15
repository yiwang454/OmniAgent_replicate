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
Input, output, and cost below include one original caption-generation call per
question that invoked C. These are **estimated** for C-enabled runs because
some legacy cache entries lack source usage metadata; see the accounting method
below. They are not just the marginal warm-cache run totals.

| Experiment | Output | Enabled groups | Accuracy | Input tok. | Output thinking tok. | Output non-thinking tok. | Tool calls | Cost incl. caption | Latency | Failures |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full (live C) | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_full_selectedVal125_measured_sepav/output_test.jsonl) | A+C+E+Q+M | 75.20% (94/125) | 49.661k | 6.140k | 1.322k | 7.776 | $0.07142 | 118.32 s (122/125) | 3 |
| −A | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125/minus_a/output_test.jsonl) | C+Q+E+M | 76.00% (95/125) | 30.055k | 3.143k | 0.965k | 7.224 | ~$0.05534 | 69.23 s (124/125) | 1 |
| −E | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125/minus_e/output_test.jsonl) | C+Q+A+M | 79.20% (99/125) | 34.065k | 4.163k | 1.246k | 8.432 | ~$0.06838 | 80.62 s (124/125) | 1 |
| **−A−E** | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125/minus_a_e/output_test.jsonl) | C+Q+M | **80.00% (100/125)** | 30.318k | 3.359k | 1.018k | 7.848 | ~$0.05788 | 72.51 s (125/125) | 0 |
| −C | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125/minus_c/output_test.jsonl) | Q+E+A+M | 73.60% (92/125) | 25.935k | 3.074k | 1.008k | 6.480 | $0.05206 | 65.56 s (124/125) | 1 |
| −Q | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125/minus_q/output_test.jsonl) | C+E+A+M | 63.20% (79/125) | 16.192k | 3.376k | 0.738k | 4.960 | ~$0.04583 | 54.14 s (125/125) | 0 |
| −M | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125/minus_m/output_test.jsonl) | C+Q+E+A | 76.80% (96/125) | 29.692k | 3.786k | 1.103k | 7.144 | ~$0.06054 | 76.68 s (124/125) | 1 |
| Minimal | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125/minimal/output_test.jsonl) | C+Q | 72.00% (90/125) | 22.449k | 2.717k | 0.901k | 6.200 | ~$0.04585 | 61.71 s (124/125) | 1 |

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

## Exact live-run aggregate usage and cost (excluding cached caption generation)

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

The old main/repeat tables omitted the **generation** round behind each warm
`audio_global_caption` lookup, both from tokens and from cost. The exact
live-run-only totals remain in the aggregate table above for auditing. In the
corrected results, a question incurs one source caption call if it used C;
repeated C lookups within that same question are counted once. The charge is
not amortized over unique videos or shared across questions/benchmarks.

For cache entries with `source_token_usage`, prompt, candidate, and thinking
tokens come directly from Gemini metadata. The older `omniagent_trace` entries
have no such metadata, so their source-call tokens **cannot be recovered
exactly**. For those entries, prompt audio/text tokens use a measured entry
from the same video when available; otherwise they use the modal measured
split for the clip duration (30 s: 961 audio + 69 text; 60 s: 1,780 audio +
0 text). Visible output is estimated as `round(0.19 × caption response
characters)`. On 502 measured cache entries this estimator has mean absolute
error 2.51 output tokens; their measured mean is 0.192 tokens/character.
All 502 reported zero thinking tokens, so missing-source thinking is estimated
as zero, **not known to be zero**. Source cost uses the Gemini 2.5 Flash rates
listed below, including the audio-input rate. Thus C-enabled adjusted totals
are estimates; no model calls were rerun to obtain them. Original caption
generation latency is not included in the reported warm-run latency.

| Repeat | Experiment | C calls | Charged questions | Exact / estimated source entries | Caption input tok. | Caption visible output tok. | Caption source cost |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | −A | 82 | 82 | 15 / 67 | 110,567 | 6,429 | ~$0.124341 |
| 1 | −E | 98 | 96 | 23 / 73 | 131,737 | 7,140 | ~$0.147047 |
| 1 | −A−E | 101 | 101 | 27 / 74 | 137,637 | 7,397 | ~$0.153397 |
| 1 | −C | 0 | 0 | 0 / 0 | 0 | 0 | $0 |
| 1 | −Q | 104 | 104 | 33 / 71 | 140,987 | 7,291 | ~$0.156369 |
| 1 | −M | 75 | 73 | 10 / 63 | 99,940 | 5,856 | ~$0.112648 |
| 1 | Minimal | 97 | 95 | 20 / 75 | 132,207 | 7,528 | ~$0.148632 |
| 2 | −A | 78 | 78 | 14 / 64 | 110,340 | 6,197 | ~$0.123997 |
| 2 | −E | 82 | 82 | 14 / 68 | 115,067 | 6,508 | ~$0.129329 |
| 2 | −A−E | 101 | 100 | 26 / 74 | 138,510 | 7,422 | ~$0.154537 |
| 2 | −C | 0 | 0 | 0 / 0 | 0 | 0 | $0 |
| 2 | −Q | 108 | 106 | 35 / 71 | 143,797 | 7,375 | ~$0.159341 |
| 2 | −M | 68 | 68 | 6 / 62 | 93,290 | 5,846 | ~$0.106118 |
| 2 | Minimal | 101 | 100 | 27 / 73 | 140,357 | 7,362 | ~$0.156319 |
| 3 | −A | 87 | 86 | 17 / 69 | 117,687 | 6,843 | ~$0.132496 |
| 3 | −E | 87 | 86 | 16 / 70 | 118,919 | 6,746 | ~$0.133610 |
| 3 | −A−E | 98 | 93 | 22 / 71 | 130,290 | 7,164 | ~$0.145930 |
| 3 | −C | 0 | 0 | 0 / 0 | 0 | 0 | $0 |
| 3 | −Q | 108 | 107 | 35 / 72 | 147,077 | 7,382 | ~$0.162735 |
| 3 | −M | 68 | 68 | 9 / 59 | 91,040 | 5,467 | ~$0.102776 |
| 3 | Minimal | 99 | 97 | 24 / 73 | 134,606 | 7,386 | ~$0.150628 |

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

- **Input tok.**: planner input plus live Gemini tool input plus the attributed
  caption-source prompt input, including cached planner input in the token
  count. Billing discounts for planner cached input are applied in cost.
- **Output thinking tok.**: o3 reasoning tokens plus live Gemini thinking and
  caption-source thinking tokens. Live Gemini and measured caption-source
  calls reported zero; missing caption-source thinking is estimated as zero.
- **Output non-thinking tok.**: planner output excluding o3 reasoning, plus
  live Gemini and attributed caption-source visible candidate output.
- **Cost incl. caption**: live-run cost plus the attributed caption-generation
  cost. o3 at $2.00/M uncached input, $0.50/M cached input and $8.00/M
  output; Gemini 2.5 Flash at $0.30/M text/image/video input, $1.00/M audio input
  and $2.50/M output. Values prefixed `~` contain estimated legacy usage.
- **Latency**: mean non-null retry-adjusted question latency. It removes failed
  request duration and retry backoff and is independent of worker concurrency.
- **Accuracy**: exact multiple-choice full-set accuracy over all 125 questions.

On this subset, `−A−E` has the highest full-set accuracy (80.0%) and also has
complete results. `−Q` has the lowest caption-inclusive estimated cost and
latency among the cache-enabled runs, but its accuracy falls to 63.2%, the
largest degradation in the ablation set. `−E` is the most expensive and
tool-intensive configuration.
These comparisons describe this fixed 125-question subset and should not be
treated as confidence-tested estimates of full-dataset effects.

## Repeat 2 results

All seven outputs contain the same 125 unique `selectedVal125` question IDs as
repeat 1. Columns use the same definitions as the main table; failures remain
in the 125-question accuracy denominator.

| Experiment | Output | Accuracy | Input tok. | Output thinking tok. | Output non-thinking tok. | Tool calls | Cost incl. caption | Latency | Failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| −A | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125_repeat2/minus_a/output_test.jsonl) | 80.00% (100/125) | 31.040k | 3.033k | 1.010k | 7.368 | ~$0.05669 | 77.56 s (124/125) | 1 |
| −E | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125_repeat2/minus_e/output_test.jsonl) | 77.60% (97/125) | 29.455k | 3.391k | 1.124k | 7.264 | ~$0.05900 | 78.04 s (124/125) | 1 |
| −A−E | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125_repeat2/minus_a_e/output_test.jsonl) | 80.00% (100/125) | 31.873k | 3.530k | 1.114k | 7.952 | ~$0.06135 | 84.75 s (124/125) | 1 |
| −C | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125_repeat2/minus_c/output_test.jsonl) | 76.00% (95/125) | 26.367k | 3.099k | 1.031k | 6.440 | $0.05354 | 74.57 s (124/125) | 1 |
| −Q | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125_repeat2/minus_q/output_test.jsonl) | 64.00% (80/125) | 18.498k | 3.762k | 0.773k | 5.360 | ~$0.05157 | 67.92 s (125/125) | 0 |
| −M | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125_repeat2/minus_m/output_test.jsonl) | 76.80% (96/125) | 27.813k | 3.365k | 1.036k | 6.648 | ~$0.05744 | 78.85 s (122/125) | 3 |
| Minimal | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125_repeat2/minimal/output_test.jsonl) | 76.00% (95/125) | 25.763k | 3.068k | 1.010k | 7.008 | ~$0.05275 | 75.13 s (125/125) | 0 |

## Repeat 3 results

| Experiment | Output | Accuracy | Input tok. | Output thinking tok. | Output non-thinking tok. | Tool calls | Cost incl. caption | Latency | Failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| −A | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125_repeat3/minus_a/output_test.jsonl) | 76.80% (96/125) | 35.013k | 3.728k | 1.130k | 8.408 | ~$0.06288 | 87.85 s (124/125) | 1 |
| −E | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125_repeat3/minus_e/output_test.jsonl) | 79.20% (99/125) | 32.079k | 3.788k | 1.183k | 7.864 | ~$0.06267 | 84.63 s (123/125) | 2 |
| −A−E | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125_repeat3/minus_a_e/output_test.jsonl) | 76.80% (96/125) | 32.218k | 3.453k | 1.109k | 8.120 | ~$0.06042 | 81.91 s (125/125) | 0 |
| −C | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125_repeat3/minus_c/output_test.jsonl) | 76.80% (96/125) | 26.488k | 3.069k | 1.020k | 6.416 | $0.05216 | 73.72 s (124/125) | 1 |
| −Q | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125_repeat3/minus_q/output_test.jsonl) | 59.20% (74/125) | 15.684k | 3.143k | 0.695k | 4.696 | ~$0.04360 | 58.80 s (125/125) | 0 |
| −M | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125_repeat3/minus_m/output_test.jsonl) | 80.80% (101/125) | 23.962k | 2.939k | 0.986k | 5.912 | ~$0.04925 | 70.71 s (123/125) | 2 |
| Minimal | [JSONL](/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125_repeat3/minimal/output_test.jsonl) | 74.40% (93/125) | 26.416k | 3.234k | 0.990k | 6.840 | ~$0.05312 | 77.27 s (124/125) | 1 |

## Three-repeat mean and standard deviation

Each cell is the arithmetic mean across repeats 1–3 plus/minus the **sample
standard deviation** (`ddof=1`, `n=3`) of the per-repeat value. Latency uses
each run's persisted retry-adjusted mean; the coverage shown in the individual
tables varies when a run terminates before producing latency for a question.

| Experiment | Accuracy | Input tok. | Output thinking tok. | Output non-thinking tok. | Tool calls | Cost incl. caption | Latency | Failures/run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| −A | 77.60% ± 2.12 pp | 32.036k ± 2.625k | 3.301k ± 0.374k | 1.035k ± 0.085k | 7.667 ± 0.646 | ~$0.05830 ± $0.00403 | 78.21 ± 9.33 s | 1.00 ± 0.00 |
| −E | 78.67% ± 0.92 pp | 31.866k ± 2.313k | 3.781k ± 0.386k | 1.184k ± 0.061k | 7.853 ± 0.584 | ~$0.06335 ± $0.00473 | 81.10 ± 3.32 s | 1.33 ± 0.58 |
| **−A−E** | **78.93% ± 1.85 pp** | 31.470k ± 1.012k | 3.448k ± 0.086k | 1.080k ± 0.054k | 7.973 ± 0.137 | ~$0.05988 ± $0.00180 | 79.72 ± 6.41 s | 0.33 ± 0.58 |
| −C | 75.47% ± 1.67 pp | 26.264k ± 0.291k | 3.081k ± 0.016k | 1.020k ± 0.012k | 6.445 ± 0.032 | $0.05258 ± $0.00083 | 71.28 ± 4.97 s | 1.00 ± 0.00 |
| −Q | 62.13% ± 2.57 pp | 16.791k ± 1.500k | 3.427k ± 0.312k | 0.735k ± 0.039k | 5.005 ± 0.334 | ~$0.04700 ± $0.00411 | 60.29 ± 7.01 s | 0.00 ± 0.00 |
| −M | 78.13% ± 2.31 pp | 27.156k ± 2.921k | 3.364k ± 0.423k | 1.042k ± 0.059k | 6.568 ± 0.620 | ~$0.05574 ± $0.00583 | 75.41 ± 4.21 s | 2.00 ± 1.00 |
| Minimal | 74.13% ± 2.01 pp | 24.876k ± 2.127k | 3.006k ± 0.264k | 0.967k ± 0.058k | 6.683 ± 0.426 | ~$0.05057 ± $0.00410 | 71.37 ± 8.43 s | 0.67 ± 0.58 |

Repeat 2/3 failure details remain in each configuration's
`metrics_summary.json`. Most terminal errors are Gemini 400 responses for
specific media/tool requests; they are included in the reported full-set
accuracy rather than excluded.
