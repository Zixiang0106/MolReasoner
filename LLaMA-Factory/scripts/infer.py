# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Run batched inference **without vLLM**, using Hugging Face Transformers only.
- Supports full merged models (recommended) and optional LoRA adapters (PEFT).
- Uses LLaMA-Factory's dataset/template pipeline so your data path/flags remain the same.
- Multimodal fields (images/videos/audios) are **skipped** in this fallback (Transformers doesn't accept vLLM's
  multi-modal input dict). Text-only samples are generated normally.

Usage example:
    python hf_infer_no_vllm.py \
      --model_name_or_path /path/to/merged_model_dir \
      --dataset your_dataset_name --dataset_dir data \
      --template chatml --cutoff_len 4096 \
      --save_name preds.jsonl --max_new_tokens 1024 --temperature 0.7 --top_p 0.9

If you *must* use a base + LoRA (not merged), also pass:
      --adapter_name_or_path /path/to/lora_adapter_dir
"""

import gc
import json
from typing import Optional
import os
import random

import fire
from tqdm import tqdm

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, Seq2SeqTrainingArguments

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.misc import get_device_count
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer

# ------------------------------- Helpers ----------------------------------

def _str_to_dtype(name: Optional[str]):
    if not name:
        return torch.float16 if torch.cuda.is_available() else None
    n = str(name).lower()
    if n in ("auto",):
        return torch.float16 if torch.cuda.is_available() else None
    if n in ("bfloat16", "bf16"): return torch.bfloat16
    if n in ("float16", "fp16", "half"): return torch.float16
    if n in ("float32", "fp32"): return torch.float32
    return torch.float16


def _maybe_load_peft(model, adapter_name_or_path):
    if adapter_name_or_path is None:
        return model
    try:
        from peft import PeftModel
    except Exception as e:
        raise ImportError(
            "PEFT is required to load LoRA adapters. Install with `pip install peft`."
        ) from e

    path = adapter_name_or_path
    if isinstance(path, (list, tuple)):
        path = path[0]
    return PeftModel.from_pretrained(model, path)


# ------------------------------- Main -------------------------------------

def hf_infer(
    model_name_or_path: str,
    adapter_name_or_path: str = None,
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 2048,
    max_samples: Optional[int] = None,
    save_name: str = "generated_predictions.jsonl",
    temperature: float = 0.95,
    top_p: float = 0.7,
    top_k: int = 50,
    max_new_tokens: int = 1024,
    repetition_penalty: float = 1.0,
    skip_special_tokens: bool = True,
    default_system: Optional[str] = None,
    enable_thinking: bool = True,
    seed: Optional[int] = None,
    batch_size: int = 32,
    preprocessing_num_workers: int = 8,
):
    """Batched inference using Transformers (no vLLM).

    Notes:
    - Multimodal entries (images/videos/audios) are skipped in this fallback.
    - If you pass a LoRA adapter, base model must match the one used during finetuning.
    """

    # ---- Parse args via LLaMA-Factory to keep behavior consistent ----
    model_args, data_args, finetuning_args, generating_args = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=preprocessing_num_workers,
            default_system=default_system,
            enable_thinking=enable_thinking,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
    )

    # Seeding
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ---- Tokenizer & template ----
    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    # ensure pad token exists
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    template_obj.mm_plugin.expand_mm_tokens = False

    # ---- Load model ----
    dtype = _str_to_dtype(getattr(model_args, "infer_dtype", None))
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model = _maybe_load_peft(model, model_args.adapter_name_or_path)
    model.eval()

    # ---- Dataset ----
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, stage='selfies', **tokenizer_module)
    train_dataset = dataset_module["train_dataset"]

    # ---- Generation hyper-params ----
    # guard values per HF generate contracts
    do_sample = True
    gen_kwargs = {
        "max_new_tokens": generating_args.max_new_tokens,
        "do_sample": do_sample,
        "temperature": max(0.0, float(generating_args.temperature)),
        "top_p": float(generating_args.top_p) if generating_args.top_p else 1.0,
        "repetition_penalty": float(generating_args.repetition_penalty or 1.0),
    }
    if generating_args.top_k and int(generating_args.top_k) > 0:
        gen_kwargs["top_k"] = int(generating_args.top_k)

    stop_ids = template_obj.get_stop_token_ids(tokenizer)
    if stop_ids and len(stop_ids) > 0:
        gen_kwargs["eos_token_id"] = stop_ids

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # ---- Loop ----
    all_prompts, all_preds, all_labels = [], [], []
    mm_skipped = 0

    for i in tqdm(range(0, len(train_dataset), batch_size), desc="Transformers batched inference"):
        batch = train_dataset[i: min(i + batch_size, len(train_dataset))]

        # collect valid (text-only) samples
        seq_list, keep_idx = [], []
        for j in range(len(batch["input_ids"])):
            has_mm = False
            try:
                has_mm = (batch.get("images") and batch["images"][j] is not None) or \
                         (batch.get("videos") and batch["videos"][j] is not None) or \
                         (batch.get("audios") and batch["audios"][j] is not None)
            except Exception:
                has_mm = False

            if has_mm:
                mm_skipped += 1
                continue

            ids = torch.tensor(batch["input_ids"][j], dtype=torch.long)
            seq_list.append(ids)
            keep_idx.append(j)

        if not seq_list:
            continue

        input_ids = pad_sequence(seq_list, batch_first=True, padding_value=pad_id)
        attention_mask = (input_ids != pad_id).to(torch.bool)

        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        # decode newly generated tokens only
        start = input_ids.size(1)
        preds_text = []
        for k in range(outputs.size(0)):
            gen_part = outputs[k, start:]
            preds_text.append(tokenizer.decode(gen_part, skip_special_tokens=skip_special_tokens))

        # collect prompts/preds/labels for kept indices
        for local_idx, pred in zip(keep_idx, preds_text):
            prompt_txt = tokenizer.decode(batch["input_ids"][local_idx], skip_special_tokens=skip_special_tokens)
            label = batch.get("groundtruth", [""] * len(batch["input_ids"]))[local_idx]
            all_prompts.append(prompt_txt)
            all_preds.append(pred)
            all_labels.append(label)

        gc.collect()

    # ---- Save ----
    with open(save_name, "w", encoding="utf-8") as f:
        for text, pred, label in zip(all_prompts, all_preds, all_labels):
            f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")

    print("*" * 70)
    print(f"{len(all_prompts)} results saved to {save_name}.")
    if mm_skipped:
        print(f"[Note] Skipped {mm_skipped} multimodal samples (images/videos/audios) in Transformers fallback.")
    print("*" * 70)


if __name__ == "__main__":
    fire.Fire(hf_infer)
