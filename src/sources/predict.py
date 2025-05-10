#!/usr/bin/env python3
# scripts/predict.py

import argparse
from pathlib import Path
import sys
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

HERE       = Path(__file__).resolve().parent
REPO_ROOT  = HERE.parent / "gec_t5" 
SRC_FOLDER = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_FOLDER))

from generate import generate
from tqdm.auto import tqdm

def main():
    p = argparse.ArgumentParser(
        description="Fill a CSV of source sentences with GEC-T5 predictions"
    )
    p.add_argument(
        "--input_csv", "-i",
        type=Path,
        required=True,
        help="Path to CSV with a 'source' column and empty 'prediction' column"
    )
    p.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help=(
            "Either a local folder (e.g. models/checkpoint-9600) "
            "or a Hugging Face repo ID "
            "(e.g. TranscendentalX/finetuned_gec_t5_chkpt3600)"
        )
    )
    p.add_argument(
        "--output_csv", "-o",
        type=Path,
        required=True,
        help="Where to write the filled CSV"
    )
    p.add_argument(
        "--batch_size", "-b",
        type=int,
        default=64,
        help="Batch size for generation"
    )
    args = p.parse_args()

    # 1. Sanity check CSV
    assert args.input_csv.exists(), f"❌ Input CSV not found: {args.input_csv}"
    df = pd.read_csv(args.input_csv)
    assert "source" in df.columns, "CSV must have a 'source' column"
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    # 2. Determine model source
    model_id = args.checkpoint
    if Path(model_id).is_dir():
        # local checkpoint folder
        repo = model_id
        local_files = True
    else:
        # treat as Hugging Face repo ID
        repo = model_id
        local_files = False

    # 3. Load tokenizer & model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from `{repo}` (local_files={local_files}) on {device}")
    tok = AutoTokenizer.from_pretrained(repo, local_files_only=local_files)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        repo,
        local_files_only=local_files,
        torch_dtype=(torch.bfloat16 if device=="cuda" else torch.float32),
        device_map="auto",
    )
    model.eval()

    # 4. Generate predictions
    sources = df["source"].fillna("").tolist()
    preds = []
    for i in tqdm(range(0, len(sources), args.batch_size), desc="Generating"):
        batch = sources[i : i + args.batch_size]
        out = generate(
            model=model,
            tokenizer=tok,
            sources=batch,
            batch_size=args.batch_size
        )
        preds.extend(out)

    assert len(preds) == len(df), "Row count mismatch after generation"
    df["prediction"] = preds

    # 5. Save output
    df.to_csv(args.output_csv, index=False)
    print(f"✓ Saved filled CSV to {args.output_csv}")

if __name__ == "__main__":
    main()
