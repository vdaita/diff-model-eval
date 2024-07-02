from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
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

shot_ir_format = """## File:
<TOP/>
def multiply(a, b):
    return a * b

def add(a, b):
    sum = a + b
    return sum

## Instruction:
1. Remove the multiply function
2. Make the add function more concise by replacing it with only a return statement
3. Add a subtract function

### Response:
<Delete>
def multiply(a, b):
    return a * b
</Delete>
<Replace>
    sum = a + b
    return sum
<With>
    return a + b
</Replace>
<Insert>
def subtract(a, b):
    return a - b
<After>
    sum = a + b
    return sum

</Insert>
    """

shot_direct_edit = """## File:
```python
def add(a, b):
    return a + b
```
## Changes:
Add a "sub" function that subtracts two numbers. Also write docstrings for both functions and change a,b to x,y.
### Response:
```python
def add(x, y):
    \"\"\"Adds two numbers.\"\"\"
    return x + y

def sub(x, y):
    \"\"\"Subtracts two numbers.\"\"\"
    return x - y
```"""

class OutputEnum(str, Enum):
    line = "line"
    ir = "ir"
    whole = "whole"
    udiff = "udiff"

def main(hf_model_id: str, model_type: OutputEnum, output_folder: str):
    dataset = load_dataset("nuprl/CanItEdit", split="test")
    # pipe = pipeline(model=hf_model_id, torch_dtype=torch.bfloat16, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(model=hf_model_id, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model=hf_model_id, device_map="auto", torch_dtype=torch.bfloat16)
    model_type = model_type.value

    if not(os.path.exists(output_folder)):
        os.makedirs(output_folder)

    for row in tqdm(dataset):
        file_path = os.path.join(output_folder, f"{row['id']}_direct.txt")
        if os.path.exists(file_path):
            continue

        formatted_input = ""
        if model_type == "whole":
            formatted_input += f"""Rewrite the file per the user's desired changes. Here's an example: \n{shot_direct_edit}\n## File:
{row['before']}

## Changes:
{row['instruction_descriptive']}

"""
        elif model_type == "ir":
            formatted_input = f"""Generate insert-after, delete, and replace blocks to edit the given file according to the user's instruction. Here's an example:
{shot_ir_format}

## File:
<TOP/>
{row['before']}

## Changes: 
{row['instruction_descriptive']}

"""
        
        formatted_input = tokenizer.apply_chat_template([{
            "role": "user",
            "content": formatted_input
        }], add_generation_prompt=True, tokenize=True, return_tensors="pt")
        output = model(**formatted_input)
        output = tokenizer.batch_decode(output)[0]

        # output = pipe(formatted_input, do_sample=True, max_new_tokens=500, top_p=0.95, **{"use_cache": True})
        # output = row['outputs']
        # output = output.replace(formatted_input, "")
        out_file = open(file_path, "w+")
        out_file.write(output)
        out_file.close()

if __name__ == "__main__":
    typer.run(main)
