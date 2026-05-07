# infer_transformers.py
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/mnt/ceph/users/zlu10/llm/MolReasoner/LLaMA-Factory/output"

tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,   # A100 推荐 bf16
    device_map="auto"
)

# --- your prompt (UNCHANGED) ---
instruction = (
"You are a professional chemist. Your task is to generate a natural, concise, and chemically accurate description of the given molecule.\n\nPlease provide a step-by-step analysis explaining how you interpret the molecular structure, identify key features and functional groups, and summarize it into a clear and informative description.\n\nThink step by step, and your final answer **must** be returned in the format: <answer> ... </answer>. For example:\n<answer>This molecule is a polycyclic aromatic hydrocarbon (PAH) with a fused aromatic ring system that binds to the aryl hydrocarbon receptor, inducing cytochrome P450 enzymes (CYP1A1, CYP1A2, CYP1B1) and facilitating systemic transport via albumin; it has high toxicity and is carcinogenic, increasing the risk of cancers (skin, respiratory tract, bladder, stomach, kidney), reproductive harm, immune suppression, and acute irritation of skin and lung tissue.</answer>"
)
smiles = "ClC(Cl)Cl"
prompt = f"{instruction}\n\nSMILES: {smiles}"

inputs = tok(prompt, return_tensors="pt").to(model.device)
with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=2056,
        temperature=0.7,
        top_p=0.8,
        repetition_penalty=1.2,
        do_sample=True
    )

text = tok.decode(out[0], skip_special_tokens=True)

def extract_answer(s: str):
    m = re.search(r"<answer>\s*(.*)\s*$", s, flags=re.S|re.I)
    if m: return m.group(1).strip()
    m2 = re.search(r"<answer>\s*(.*?)\s*</answer>", s, flags=re.S|re.I)
    return m2.group(1).strip() if m2 else s.strip()

print("RAW:", text)
print("DESC:", extract_answer(text))
