import json
import os
from pathlib import Path

import selfies as sf
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==========================
#  配置区
# ==========================

# Base LLaMA2 chat 模型 + Mol-Instructions LoRA
BASE_MODEL = "meta-llama/Llama-2-7b-chat-hf"
ADAPTER_MODEL = "zjunlp/llama2-molinst-molecule-7b"

# 数据 & 输出路径
INPUT_PATH = "/mnt/home/zlu10/ceph/llm/MolReasoner/LLaMA-Factory/scripts/molreasoner/data/IUPAC.txt"
OUTPUT_PATH = "/mnt/home/zlu10/ceph/llm/MolReasoner/LLaMA-Factory/scripts/molreasoner/molinst_IUPAC.jsonl"

# 推理超参
BATCH_SIZE = 4
MAX_NEW_TOKENS = 4096

# prompt：贴近 Mol-Inst 原始任务，然后加一个 <answer> 壳方便后处理
BASE_PROMPT = (
    "You are a professional chemist. Your task is to generate the IUPAC name of the given molecule."
    "Molecule SELFIES: "
)


# ==========================
#  工具函数
# ==========================

def smiles_to_selfies(smiles: str) -> str:
    """SMILES -> SELFIES."""
    try:
        s = sf.encoder(smiles)
        return s if s is not None else ""
    except Exception:
        return ""


def load_data(path: str):
    """
    读取 test.txt，格式为:
    CID<TAB>SMILES<TAB>description

    返回: 列表，每个元素是 dict:
    {
        "cid": ...,
        "smiles": ...,
        "selfies": ...,
        "description": ...
    }
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()  # 跳过表头
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                print("Skip malformed line:", line)
                continue
            cid, smiles, desc = parts[0], parts[1], parts[2]
            if not desc.strip():
                print("Skip empty description, CID:", cid)
                continue
            selfies = smiles_to_selfies(smiles)
            data.append(
                {
                    "cid": cid,
                    "smiles": smiles,
                    "selfies": selfies,
                    "description": desc,
                }
            )
    return data


def build_prompt_from_selfies(selfies: str) -> str:
    """把 SELFIES 塞进 prompt."""
    return BASE_PROMPT.format(selfies=selfies)


# ==========================
#  主流程
# ==========================

def main():
    print("Loading data from:", INPUT_PATH)
    samples = load_data(INPUT_PATH)
    print(f"Loaded {len(samples)} valid samples.")

    # ---- 加载 tokenizer & base 模型 ----
    print("Loading base model:", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)

    # ⭐ LLaMA2 默认没有 pad_token，这里手动指定为 eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    # ---- 加载 Mol-Instructions LoRA ----
    print("Loading Mol-Instructions adapter:", ADAPTER_MODEL)
    model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL)

    # 推理阶段可以尝试 merge LoRA 提升速度 / 减少显存占用
    try:
        model = model.merge_and_unload()
    except Exception as e:
        print("merge_and_unload failed, using PeftModel directly:", e)

    model.eval()

    # ---- 准备输出文件 ----
    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.unlink(missing_ok=True)  # 删旧文件
    fout = out_path.open("w", encoding="utf-8")

    total = len(samples)
    print("Start inference... total =", total)

    # ---- 批量推理 ----
    with torch.inference_mode():
        for start in range(0, total, BATCH_SIZE):
            batch = samples[start: start + BATCH_SIZE]
            prompts = [build_prompt_from_selfies(s["selfies"]) for s in batch]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,      # 现在有 pad_token 了，可以安全 padding
                truncation=True,
            ).to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,   # 评测先用 greedy，稳定一点
                temperature=1.0,
            )

            attn = inputs["attention_mask"]

            for i, s in enumerate(batch):
                input_len = int(attn[i].sum().item())
                gen_ids = outputs[i, input_len:]
                pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

                record = {
                    "SELFIES": s["selfies"],
                    "predict": pred_text,
                    "label": s["description"],
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            # 每个 batch 强制刷盘，防止中途挂了全丢
            fout.flush()
            os.fsync(fout.fileno())

            print(f"Processed {min(start + BATCH_SIZE, total)}/{total}")

    fout.close()
    print("Done. Saved to", OUTPUT_PATH)


if __name__ == "__main__":
    main()
