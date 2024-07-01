from datasets import load_dataset
from transformers import pipeline, AutoTokenizer
from typing import Literal
import torch
import typer
from tqdm import tqdm
import re
import os
import json
import diff_utils
import subprocess
from enum import Enum

class OutputEnum(str, Enum):
    line = "line"
    ir = "ir"
    whole = "whole"
    udiff = "udiff"

def main(hf_model_id: str, model_type: OutputEnum, output_folder: str):
    dataset = load_dataset("nuprl/CanItEdit", split="test")
    pipe = pipeline(model=hf_model_id, torch_dtype=torch.bfloat16, device_map="auto")

    dataset_inputs = []
    for row in tqdm(dataset):
        file_path = os.path.join(output_folder, f"{row['id']}_direct.txt")
        if os.path.exists(file_path):
            continue

        old_contents = f"<TOP/>\n{row['before']}"
        formatted_input = f"# File:\n{old_contents}\n# Instruction:{row['instruction_descriptive']}"
        if model_type == "whole":
            formatted_input += "\nPlease completely rewrite the file, adding  the changes from instruction. Output using a Python code block (start with ```python and end with ```)."
        
        dataset_inputs.append(formatted_input)

    outputs = pipe(dataset_inputs)
    outputs = [output[0]["generated_text"] for output in outputs]
    dataset["outputs"] = outputs
    
    model_type = model_type.value

    if not(os.path.exists(output_folder)):
        os.makedirs(output_folder)

    for row in tqdm(dataset):
        file_path = os.path.join(output_folder, f"{row['id']}_direct.txt")
        if os.path.exists(file_path):
            continue

        old_contents = f"<TOP/>\n{row['before']}"
        formatted_input = f"# File:\n{old_contents}\n# Instruction:{row['instruction_descriptive']}"
        if model_type == "whole":
            formatted_input += "\nPlease completely rewrite the file, adding  the changes from instruction. Output using a Python code block (start with ```python and end with ```)."
        # output = pipe(formatted_input, do_sample=True, max_new_tokens=500, top_p=0.95, **{"use_cache": True})
        output = row['outputs']
        output = output.replace(formatted_input, "")
        out_file = open(file_path, "w+")
        out_file.write(output)
        out_file.close()

if __name__ == "__main__":
    typer.run(main)
