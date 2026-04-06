import os
import subprocess

BASE = "../outputs"
MODEL = "meta-llama/Llama-Guard-3-8B"
PROMPT = "code/prompt_standard.txt"
SAVE = "../evals"

for root, dirs, files in os.walk(BASE):
    for file in files:
        if file.endswith(".json"):
            dataset_path = os.path.join(root, file)

            print(f"\n=== Running on {dataset_path} ===\n")

            cmd = [
                "python", "code/evaluation_coreference.py",
                "--model", MODEL,
                "--prompt", PROMPT,
                "--dataset", dataset_path,
                "--save_path", SAVE,
                "--num_samples", "-1"
            ]

            subprocess.run(cmd)
