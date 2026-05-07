import sys
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# === 1. 模型路径 ===
MODEL_DIR = "/mnt/home/zlu10/ceph/llm/weight/molreasoner/grpo/grpo_molecule_captioning"

print("Loading tokenizer from:", MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

def build_prompt(smiles: str) -> str:
    """用 chat_template 构建 SMILES -> caption 的对话 prompt."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a molecular expert. Given a molecule in SMILES, "
                "describe its key functional groups, core scaffold, and "
                "basic physicochemical and reactivity properties."
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
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

print("Initializing vLLM engine...")
llm = LLM(
    model=MODEL_DIR,
    trust_remote_code=True,
    dtype="bfloat16",          # Qwen2 默认 bf16，集群支持就用它
    tensor_parallel_size=1,    # 多卡就改成 GPU 数
    max_model_len=4096,        # 够用就行
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    max_tokens=4096,
    repetition_penalty=1.05,
)

if __name__ == "__main__":
    # 用法：
    #   python mol_infer_vllm.py "CC(=O)Oc1ccccc1C(=O)O" "CCN(CC)CCOC(=O)c1ccccc1"
    smiles_list = sys.argv[1:] or [
        "CC(=O)Oc1ccccc1C(=O)O",          # aspirin
        "CCN(CC)CCOC(=O)c1ccccc1",       # lidocaine-ish
    ]

    prompts = [build_prompt(s) for s in smiles_list]
    print(f"Running vLLM on {len(prompts)} molecules...")
    outputs = llm.generate(prompts, sampling_params)

    for s, out in zip(smiles_list, outputs):
        text = out.outputs[0].text.strip()
        print("=" * 80)
        print("SMILES:", s)
        print("Caption:\n", text)
