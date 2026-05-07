import json
import os
from pathlib import Path

import selfies as sf
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== 路径配置 =====
MODEL_DIR = "/mnt/home/zlu10/ceph/llm/weight/molreasoner/grpo/grpo_molecule_captioning"
INPUT_PATH = "/mnt/home/zlu10/ceph/llm/MolReasoner/LLaMA-Factory/scripts/molreasoner/data/IUPAC.txt"
OUTPUT_PATH = "/mnt/home/zlu10/ceph/llm/MolReasoner/LLaMA-Factory/scripts/molreasoner/molreasoner_IUPAC.jsonl"

BATCH_SIZE = 4
MAX_NEW_TOKENS = 4096

# ===== 你的 mol2desc prompt 模板（注意最后是 Molecule SELFIES:） =====
BASE_PROMPT = (
    "You are a professional chemist. Your task is to generate the IUPAC name of the given molecule."
    "Molecule SELFIES: "
)


def smiles_to_selfies(smiles: str) -> str:
    """SMILES -> SELFIES，如果失败就返回空字符串."""
    try:
        s = sf.encoder(smiles)
        return s if s is not None else ""
    except Exception:
        return ""


def load_data(path: str):
    """读取 test.txt，返回包含 cid, smiles, selfies, desc 的列表."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()  # 跳过第一行 CID SMILES description
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                # 例如最后一行只有 CID，直接跳过
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
    """把 SELFIES 拼到你的模板后面."""
    return BASE_PROMPT + selfies


def main():
    print("Loading data from:", INPUT_PATH)
    samples = load_data(INPUT_PATH)
    print(f"Loaded {len(samples)} valid samples.")

    print("Loading model from:", MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 先删旧文件，避免重复追加
    out_path.unlink(missing_ok=True)
    fout = out_path.open("w", encoding="utf-8")

    total = len(samples)
    print("Start inference...")

    with torch.inference_mode():
        for start in range(0, total, BATCH_SIZE):
            batch = samples[start : start + BATCH_SIZE]
            prompts = [build_prompt_from_selfies(s["selfies"]) for s in batch]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,   # 先用 greedy，评测更稳定
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

            # 👉 每个 batch 结束强制 flush 一次，防止中途挂了全丢
            fout.flush()
            os.fsync(fout.fileno())

            print(f"Processed {min(start + BATCH_SIZE, total)}/{total}")

    fout.close()
    print("Done. Saved to", OUTPUT_PATH)


if __name__ == "__main__":
    main()
