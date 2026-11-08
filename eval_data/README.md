# Evaluation data

Only the dataset files consumed by evaluation are included here.

| Benchmark | File |
|---|---|
| Math500 | `math500.jsonl` |
| GSM8K | `gsm8k.parquet` |
| HumanEval | `humaneval.parquet` |
| MBPP | `mbpp.parquet` |
| GPQA Diamond | `gpqa.csv` |
| MMLU validation | `mmlu.parquet` |
| WildJailbreak | `wildjailbreak.parquet` |
| StrongREJECT | `strongreject.parquet` |
| Strata SWORD English level 3 | `strata_sword_en_level_3.jsonl` |

`bash evaluate.sh --preflight-only` checks the files, columns and row counts.
`manifest.json` records source filenames, sizes and SHA-256 checksums.
