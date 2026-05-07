#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MolT5 / 3D-MolT5 inference that outputs JSON lines in the exact format:
{"prompt": "<system/user/assistant-formatted prompt>", "predict": "...", "label": "..."}

Input: a JSON array with fields: instruction (str), input (str), output (optional str as label)
"""

import os
import json
import argparse
from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


QWEN_SYSTEM = ""
# 拼接成你需要的 prompt 形态：
# system\n{系统提示}\nuser\n{instruction}\n{input}\nassistant\n
def build_qwen_prompt(instruction: str, molecule: str) -> str:
    ins = "" if instruction is None else str(instruction)
    mol = "" if molecule is None else str(molecule)
    return f"{QWEN_SYSTEM}\nuser\n{ins}\n{mol}\nassistant\n"


def str2dtype(name: str):
    if not name:
        return torch.float16 if torch.cuda.is_available() else None
    n = name.lower()
    if n in ("bf16", "bfloat16"):
        return torch.bfloat16
    if n in ("fp16", "float16", "half"):
        return torch.float16
    if n in ("fp32", "float32"):
        return torch.float32
    if n in ("auto",):
        return torch.float16 if torch.cuda.is_available() else None
    return None


def load_json_array(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), "Input JSON must be a list."
    return data


def main():
    ap = argparse.ArgumentParser("MolT5 inference (Qwen-style prompt JSON output)")
    ap.add_argument("--model_name_or_path", type=str, required=True,
                    help="HF repo id or local dir, e.g. laituan245/molt5-large-smiles2caption")
    ap.add_argument("--input_json", type=str, required=True,
                    help="Path to JSON array like tox_test.json")
    ap.add_argument("--output_jsonl", type=str, required=True,
                    help="Where to write JSONL lines with prompt/predict/label")
    ap.add_argument("--instruction_field", type=str, default="instruction")
    ap.add_argument("--input_field", type=str, default="input")
    ap.add_argument("--label_field", type=str, default="output")

    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--max_source_length", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=128)

    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--length_penalty", type=float, default=1.0)

    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32", "auto"])
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--attn_impl", type=str, default=None, choices=[None, "flash_attention_2", "sdpa"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    # 小优化：碎片管理、TF32（A100 友好）
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

    # 读取数据
    rows = load_json_array(args.input_json)
    if args.limit:
        rows = rows[: args.limit]
    assert len(rows) > 0, "No samples."

    # 构造 Qwen 风格 prompt（精确匹配你给的结构）
    prompts: List[str] = [
        build_qwen_prompt(item.get(args.instruction_field), item.get(args.input_field))
        for item in rows
    ]

    # MolT5 tokenizer（Seq2Seq：一般右填充）
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # MolT5 模型（Seq2Seq）
    dtype = str2dtype(args.dtype)
    model_kwargs = dict(
        device_map=args.device_map,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if args.attn_impl:
        model_kwargs["attn_implementation"] = args.attn_impl
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    model.eval()

    # DataLoader
    def collate(batch_prompts: List[str]):
        enc = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=args.max_source_length,
            return_tensors="pt",
        )
        return enc, batch_prompts

    loader = DataLoader(prompts, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # 生成参数
    do_sample = (args.num_beams == 1 and args.temperature > 0.0)
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=do_sample,
        temperature=max(0.0, float(args.temperature)),
        top_p=float(args.top_p),
        repetition_penalty=float(args.repetition_penalty),
        length_penalty=float(args.length_penalty),
        num_beams=max(1, int(args.num_beams)),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # 推理 & 写出（逐行 JSON）
    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    written = 0
    with open(args.output_jsonl, "w", encoding="utf-8") as fout, torch.no_grad():
        for enc, batch_prompts in tqdm(loader, desc="MolT5 inference"):
            # 分片时不手搬到 cuda:0；仅在非分片时 .to(model.device)
            if getattr(model, "hf_device_map", None) is None:
                for k in enc:
                    enc[k] = enc[k].to(model.device)

            out = model.generate(**enc, **gen_kwargs)
            preds = tokenizer.batch_decode(out, skip_special_tokens=True)

            for i, pred in enumerate(preds):
                j = written + i
                item = rows[j]
                record = {
                    "prompt": batch_prompts[i],
                    "predict": pred.strip(),
                }
                if args.label_field in item:
                    record["label"] = item[args.label_field]
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += len(preds)

    print(f"Saved {written} lines to {args.output_jsonl}")


if __name__ == "__main__":
    main()
