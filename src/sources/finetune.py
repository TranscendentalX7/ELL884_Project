#!/usr/bin/env python3
# scripts/finetune.py
import argparse, json, random, numpy as np, os, torch
from pathlib import Path
from datasets import Dataset, DatasetDict
from typing import List, Dict
import math
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)


def load_split(path: Path, prefix: str, dev_ratio: float = 0.05) -> Dataset:
    """
    Load {prefix}.src/.trg if they exist; else parse train.m2, split 95/5,
    write out train/valid .src/.trg, and return the requested split.
    """
    src_fp = path / f"{prefix}.src"
    trg_fp = path / f"{prefix}.trg"

    # 1. If .src & .trg already exist, load them directly
    if src_fp.exists() and trg_fp.exists():
        with src_fp.open(encoding="utf-8") as f:
            sources = [l.rstrip("\n") for l in f]
        with trg_fp.open(encoding="utf-8") as f:
            targets = [l.rstrip("\n") for l in f]
        assert len(sources)==len(targets), "Mismatch in src/trg lengths"
        return Dataset.from_dict({"source": sources, "target": targets})

    # 2. Otherwise, fallback to parsing train.m2
    m2_fp = path / "train.m2"
    assert m2_fp.exists(), f"Need either {src_fp.name} or train.m2 in {path}"

    # parse & filter identity pairs
    records = M2Parser.parse_m2_file(str(m2_fp))
    pairs = []
    for rec in records:
        src = rec["source"]
        trg = M2Parser.apply_corrections(src, rec["corrections"])
        if src != trg:
            pairs.append((src, trg))

    # shuffle & split 95/5
    random.seed(42)
    random.shuffle(pairs)
    n_val = math.ceil(dev_ratio * len(pairs))
    valid_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    # write out files for inspection / reuse
    for name, subset in (("train", train_pairs), ("valid", valid_pairs)):
        with open(path / f"{name}.src", "w", encoding="utf-8") as S, \
             open(path / f"{name}.trg", "w", encoding="utf-8") as T:
            for s, t in subset:
                S.write(s + "\n")
                T.write(t + "\n")

    # now load the requested split
    sel = train_pairs if prefix=="train" else valid_pairs
    sources = [s for (s,_) in sel]
    targets = [t for (_,t) in sel]
    return Dataset.from_dict({"source": sources, "target": targets})

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",           type=Path, required=True,  help="path to data/")
    p.add_argument("--model_name",         type=str,   default="google-t5/t5-large")
    p.add_argument("--output_dir",         type=Path,  default=Path("models"))
    p.add_argument("--num_train_epochs",   type=int,   default=8)
    p.add_argument("--per_device_train_batch_size",  type=int, default=4)
    p.add_argument("--per_device_eval_batch_size",   type=int, default=4)
    p.add_argument("--gradient_accumulation_steps",  type=int, default=8)
    p.add_argument("--learning_rate",      type=float, default=1e-5)
    p.add_argument("--warmup_steps",       type=int,   default=1000)
    p.add_argument("--max_length",         type=int,   default=128)
    p.add_argument("--save_steps",         type=int,   default=3600)
    p.add_argument("--save_total_limit",   type=int,   default=3)
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--deepspeed",          type=str,   default="ds_config.json")
    p.add_argument("--push_to_hub",        action="store_true")
    args = p.parse_args()

    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load data
    raw = DatasetDict({
        "train":      load_split(args.data_dir, "train"),
        "validation": load_split(args.data_dir, "valid"),
    })

    # tokenizer & preprocess
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    def preprocess(batch):
        enc = tokenizer(batch["source"],
                        truncation=True,
                        padding="longest",
                        max_length=args.max_length)
        with tokenizer.as_target_tokenizer():
            lbl = tokenizer(batch["target"],
                            truncation=True,
                            padding="longest",
                            max_length=args.max_length)
        enc["labels"] = lbl["input_ids"]
        return enc

    tokenized = raw.map(
        preprocess,
        batched=True,
        remove_columns=raw["train"].column_names,
    )

    # model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    use_ds = args.deepspeed and torch.cuda.is_available()
    if use_ds:
        ds_config = args.deepspeed
        print(f"[INFO] CUDA detected. Using DeepSpeed config: {ds_config}")
    else:
        ds_config = None
        print("[INFO] No CUDA or no deepspeed config provided. Running without DeepSpeed.")

     # Build training arguments dict dynamically
    train_args_kwargs = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type":      "linear",
        "warmup_steps": args.warmup_steps,
        "eval_strategy": "epoch",
        "eval_steps": args.save_steps,
        "save_strategy": "epoch",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "logging_steps": 100,
        "report_to": ["wandb"],
        "run_name": "gec_t5_finetune",
        "fp16": False,
        "bf16": True,
        "push_to_hub": args.push_to_hub,
        "hub_model_id": None,
        "seed": args.seed,
    }
    if use_ds:
        train_args_kwargs["deepspeed"] = ds_config

    training_args = Seq2SeqTrainingArguments(**train_args_kwargs)

    trainer = Seq2SeqTrainer(
        model            = model,
        args             = training_args,
        train_dataset    = tokenized["train"],
        eval_dataset     = tokenized["validation"],
        data_collator    = data_collator,
        tokenizer        = tokenizer,
    )

    trainer.train()
    trainer.save_model()
    if args.push_to_hub:
        trainer.push_to_hub()

if __name__=="__main__":
    main()
