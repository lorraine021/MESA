#!/usr/bin/env python3
import json
import os
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("DATA_ROOT", ROOT / "eval_data"))
SPECS = {
    "math500": (DATA/"math500.jsonl", {"problem", "answer"}),
    "gsm8k": (DATA/"gsm8k.parquet", {"question", "answer"}),
    "humaneval": (DATA/"humaneval.parquet", {"task_id", "prompt", "test", "entry_point"}),
    "mbpp": (DATA/"mbpp.parquet", {"task_id", "text", "test_list", "test_setup_code"}),
    "gpqa": (DATA/"gpqa.csv", {"Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"}),
    "mmlu": (DATA/"mmlu.parquet", {"question", "subject", "choices", "answer"}),
    "wildjailbreak": (DATA/"wildjailbreak.parquet", {"prompt", "label"}),
    "strongreject": (DATA/"strongreject.parquet", {"prompt"}),
    "strata_sword_en_level_3": (DATA/"strata_sword_en_level_3.jsonl", {"\u653b\u51fb\u6307\u4ee4", "attack method", "\u98ce\u9669\u5206\u7c7b"}),
}
for name, (path, required) in SPECS.items():
    if not path.is_file():
        raise FileNotFoundError(f"{name}: {path}")
    if path.suffix == ".jsonl":
        rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        columns=set(rows[0]) if rows else set(); count=len(rows)
    elif path.suffix == ".csv":
        df=pd.read_csv(path); columns=set(df.columns); count=len(df)
    else:
        df=pd.read_parquet(path); columns=set(df.columns); count=len(df)
    missing=required-columns
    if missing:
        raise ValueError(f"{name}: missing={sorted(missing)}, columns={sorted(columns)}")
    print(f"{name}: OK rows={count} path={path}")
