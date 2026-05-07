import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# === 1. 模型路径 ===
MODEL_DIR = "/mnt/home/zlu10/ceph/llm/weight/molreasoner/grpo/grpo_molecule_captioning"

print("Loading tokenizer & model from:", MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)

# === 2. 构造 prompt（用 chat_template） ===
def build_prompt_from_smiles(smiles: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a molecular expert. Given a molecule in SMILES, "
                "describe its key functional groups, core scaffold, and basic "
                "physicochemical and reactivity properties in a concise paragraph."
            ),
        },
        {
            "role": "user",
            "content": (
                f"SMILES: {smiles}\n\n"
                "Please provide a detailed natural language description of this molecule."
            ),
        },
    ]
    # 用 config 里的 chat_template 来包 prompt
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return text

# === 3. 单条 SMILES caption 函数 ===
@torch.inference_mode()
def caption_one(
    smiles: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.8,
) -> str:
    prompt = build_prompt_from_smiles(smiles)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.05,  # 和 RL 时常用设置类似，防止太啰嗦
    )

    # 只取新生成部分
    gen_ids = outputs[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()

if __name__ == "__main__":
    # 用法：
    #   python molreasoner_caption_infer.py "CC(=O)Oc1ccccc1C(=O)O" "C1=CC=CC=C1"
    smiles_list = sys.argv[1:] or ["CC(=O)Oc1ccccc1C(=O)O"]  # 默认阿司匹林

    for s in smiles_list:
        desc = caption_one(s)
        print("=" * 80)
        print("SMILES:", s)
        print("Caption:\n", desc)
