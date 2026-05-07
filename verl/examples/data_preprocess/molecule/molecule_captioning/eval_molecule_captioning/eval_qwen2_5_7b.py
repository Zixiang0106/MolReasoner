from eval_metrics import evaluate_from_json
import json
import pandas as pd
import os

def evaluate(json_path: str, output_txt_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f if line.strip()]

    #data = pd.DataFrame(records)
    # count_prompt_without_assistant = sum(
    #     1 for item in records if "assistant" not in item["prompt"]
    # )
    # print(f"Number of prompts without 'assistant': {count_prompt_without_assistant}")
    metrics = evaluate_from_json(json_path=json_path)
    with open(output_txt_path, "w") as f:
        f.write("🔬 Evaluation Metrics (Text-Based)\n")
        f.write("=====================\n")
        f.write(f"📘 BLEU-2 (↑): {metrics['bleu2']:.4f}\n")
        f.write(f"📘 BLEU-4 (↑): {metrics['bleu4']:.4f}\n")
        f.write(f"💡 METEOR (↑): {metrics['meteor']:.4f}\n")
        f.write(f"📕 ROUGE-1 (↑): {metrics['rouge1']:.4f}\n")
        f.write(f"📕 ROUGE-2 (↑): {metrics['rouge2']:.4f}\n")
        f.write(f"📕 ROUGE-L (↑): {metrics['rougeL']:.4f}\n")


if __name__ == "__main__":
    base_dir = "/mnt/home/zlu10/ceph/llm/ToxAgent"

    files_to_eval = [
        "sft_infer.jsonl",

    ]

    for fname in files_to_eval:
        json_path = os.path.join(base_dir, fname)
        out_path = os.path.join(base_dir, fname.replace(".jsonl", "_metrics.txt"))
        print(f"▶ Evaluating {fname}")
        evaluate(json_path, out_path)
        print(f"✔ Saved metrics to {out_path}\n")


# BLEU-2 score: 0.07924136173282727
# BLEU-4 score: 0.02580415191855082
# Average Meteor score: 0.2131933659537381
# ROUGE score:
# rouge1: 0.20913937310529748
# rouge2: 0.06012665049242252
# rougeL: 0.1483177575557626