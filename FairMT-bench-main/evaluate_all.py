import os
import subprocess

folders = [
    "../outputs_religion_minus_1",
    "../outputs_religion_plus_1",
    "../outputs_race_minus_1",
    "../outputs_race_plus_1",
    "../outputs_gender_minus_1",
    "../outputs_gender_plus_1"
]
MODEL = "meta-llama/Llama-Guard-3-8B"
PROMPT = "code/prompt_standard.txt"
for folder in folders:
    folder_name = os.path.basename(folder)
    SAVE = f"../evals_{folder_name}"
    for root, dirs, files in os.walk(folder):
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
