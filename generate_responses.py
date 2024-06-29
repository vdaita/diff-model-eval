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

    model_type = model_type.value

    for row in tqdm(dataset):
        file_path = os.path.join(output_folder, f"{row['id']}_direct.txt")
        if os.exists(file_path, "w+"):
            continue

        old_contents = f"<TOP/>\n{row['before']}"
        formatted_input = f"# File:\n{old_contents}\n# Instruction:{row['instruction_descriptive']}"
        if model_type == "whole":
            formatted_input += "Please completely the file given the instruction in the form of a Python code block (start with ```python and end with ```)."
        output = pipe(formatted_input, do_sample=True, max_new_tokens=500, top_p=0.95, **{"use_cache": True})
        output = output[0]["generated_text"]
        out_file = open(os.path.join(output_folder, f"{row['id']}_direct.txt"), "w+")
        out_file.write(output)
        out_file.close()

if __name__ == "__main__":
    typer.run(main)
