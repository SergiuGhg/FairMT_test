import os
import subprocess

BASE = "FairMT_10K"
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
PROMPT = "code/prompt_standard.txt"
SAVE = "../outputs_gender_minus_1"

for root, dirs, files in os.walk(BASE):
    for file in files:
        if file.endswith(".json"):
            dataset_path = os.path.join(root, file)

            print(f"\n=== Running on {dataset_path} ===\n")

            cmd = [
                "python", "code/generate_answer_vectors.py",
                "--model", MODEL,
                "--prompt", PROMPT,
                "--dataset", dataset_path,
                "--save_path", SAVE,
                "--num_samples", "-1"
            ]

            subprocess.run(cmd)
