# OmniAgent tool usage and original-question reuse

Source: `/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_repeat1/output_test.jsonl`

## Scope

- Samples: 1197
- Total tool calls: 8700
- Question-bearing tool calls: 6294
- Samples with a question-bearing tool: 1172

The copied ReAct analyzer was not directly valid for this run: it only selected `ask_perception.perceptual_question`. This report recognizes OmniAgent's `question` and `query` arguments using an explicit per-tool schema. Calls without a semantic question argument remain in tool-usage totals but are excluded from the question-match denominator.

## Tool usage

| Tool | Calls | Share of all calls | Samples using tool | Sample coverage |
| --- | ---: | ---: | ---: | ---: |
| `video_clip_qa` | 3592 | 41.29% | 756 | 63.16% |
| `audio_qa` | 1299 | 14.93% | 659 | 55.05% |
| `Audio_EventLocation` | 778 | 8.94% | 544 | 45.45% |
| `video_metadata` | 769 | 8.84% | 766 | 63.99% |
| `audio_global_caption` | 696 | 8.00% | 695 | 58.06% |
| `video_global_qa` | 625 | 7.18% | 535 | 44.70% |
| `audio_ASR` | 561 | 6.45% | 447 | 37.34% |
| `Audio_EventList` | 380 | 4.37% | 374 | 31.24% |

## Original-question reuse

| Match definition | Question-bearing calls | All samples with >=1 match | Samples with question tool |
| --- | ---: | ---: | ---: |
| Verbatim (case-sensitive) | 0.16% (10/6294) | 0.84% (10/1197) | 0.85% (10/1172) |
| Verbatim (case-insensitive) | 0.16% (10/6294) | 0.84% (10/1197) | 0.85% (10/1172) |
| Full normalized question | 0.17% (11/6294) | 0.92% (11/1197) | 0.94% (11/1172) |
| First 3 normalized words | 1.87% (118/6294) | 9.19% (110/1197) | 9.39% (110/1172) |
| Last 3 normalized words | 9.71% (611/6294) | 33.58% (402/1197) | 34.30% (402/1172) |
| Rough: first or last 3 words | 11.09% (698/6294) | 39.01% (467/1197) | 39.85% (467/1172) |

`Rough` follows the copied analyzer's edge heuristic: after lowercasing and removing punctuation, either the first or last 3 words of the original question must occur contiguously in the tool question. It includes full normalized matches and is a lexical heuristic, not a semantic-similarity judgment.

## Match frequency by tool

| Tool | Question calls | Samples | Verbatim | Full normalized | Rough edge |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Audio_EventLocation` | 778 | 544 | 0.00% (0/778) | 0.00% (0/778) | 21.08% (164/778) |
| `audio_qa` | 1299 | 659 | 0.46% (6/1299) | 0.54% (7/1299) | 17.63% (229/1299) |
| `video_clip_qa` | 3592 | 756 | 0.03% (1/3592) | 0.03% (1/3592) | 4.96% (178/3592) |
| `video_global_qa` | 625 | 535 | 0.48% (3/625) | 0.48% (3/625) | 20.32% (127/625) |

Detailed artifacts: `summary.json`, `tool_question_calls.csv`, and `rough_only_review_sample.json`.
