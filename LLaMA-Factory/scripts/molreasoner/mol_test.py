import sys

import selfies as sf
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ===== 1. 模型 & 数据路径 =====
BASE_MODEL = "meta-llama/Llama-2-7b-chat-hf"          # LLaMA2-7B-chat
ADAPTER_MODEL = "zjunlp/llama2-molinst-molecule-7b"   # Mol-Inst LoRA 适配器

TEST_TXT = "/mnt/home/zlu10/ceph/llm/MolReasoner/LLaMA-Factory/scripts/molreasoner/data/test.txt"

# ===== 2. 你的 mol2desc prompt =====
# BASE_PROMPT = (
#     "You are a professional chemist. Your task is to generate a natural, concise, "
#     "and chemically accurate description of a given molecule.\n\n"
#     "Please provide a step-by-step analysis explaining how you interpret the "
#     "molecular structure, identify key features and functional groups, and "
#     "summarize it into a clear and informative description.\n\n"
#     "Think step by step, and your final answer must be returned in the format: "
#     "<answer> ... </answer>. For example:\n"
#     "<answer>The molecule is an epoxy(hydroxy)icosatrienoate that is the "
#     "conjugate base of 11-hydroxy-(14R,15S)-epoxy-(5Z,8Z,12E)-icosatrienoic acid, "
#     "obtained by deprotonation of the carboxy group; major species at pH 7.3. "
#     "It is a conjugate base of an 11-hydroxy-(14R,15S)-epoxy-(5Z,8Z,12E)-"
#     "icosatrienoic acid.</answer>\n\n"
#     "Molecule SELFIES: "
# )
BASE_PROMPT = "Please give me some details about this molecule: "


def smiles_to_selfies(smiles: str) -> str:
    """SMILES -> SELFIES."""
    try:
        s = sf.encoder(smiles)
        return s if s is not None else ""
    except Exception:
        return ""


def load_first_sample(path: str):
    """从 test.txt 读第一条有效样本（跳过表头），返回一个 dict."""
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()  # 跳过 "CID\tSMILES\tdescription"
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            cid, smiles, desc = parts[0], parts[1], parts[2]
            if not desc.strip():
                continue
            selfies = smiles_to_selfies(smiles)
            return {
                "cid": cid,
                "smiles": smiles,
                "selfies": selfies,
                "description": desc,
            }
    raise RuntimeError("No valid sample found in test.txt")


def build_prompt_from_selfies(selfies: str) -> str:
    return BASE_PROMPT + selfies


if __name__ == "__main__":
    # ===== 3. 加载 base LLaMA2 + Mol-Inst LoRA =====
    print("Loading base model:", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    print("Loading Mol-Instructions adapter:", ADAPTER_MODEL)
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_MODEL,
    )
    # 纯推理可以（可选）merge 一下，提高速度，节省显存
    # 如果显存不够，可以先注释掉这一行
    try:
        model = model.merge_and_unload()
    except Exception as e:
        print("merge_and_unload failed, use PeftModel directly:", e)

    model.eval()

    # ===== 4. 取一条分子 SELFIES 作为输入 =====
    if len(sys.argv) > 1:
        selfies = sys.argv[1]
        sample_info = {"cid": "CLI", "smiles": "", "selfies": selfies, "description": ""}
    else:
        print("Loading first sample from:", TEST_TXT)
        sample_info = load_first_sample(TEST_TXT)
        selfies = sample_info["selfies"]

    print("=" * 80)
    print("CID:", sample_info.get("cid", ""))
    print("SMILES:", sample_info.get("smiles", ""))
    print("SELFIES:", selfies)
    print("=" * 80)

    prompt = build_prompt_from_selfies(selfies)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # ===== 5. 生成描述 =====
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,   # 先 greedy 看风格
            temperature=1.0,
        )

    input_len = int(inputs["attention_mask"][0].sum().item())
    gen_ids = outputs[0, input_len:]
    pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    print("=== MODEL OUTPUT ===")
    print(pred)
