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


def add_line_modifications_to_code(original_code, modification_xml):
    new_code = []
    delete_lines = []
    add_lines = {}

    insert_pattern = re.compile(r'<Insert>(.*?)</Insert>')
    insert_matches = insert_pattern.findall(modification_xml)

    for match in insert_matches:
        try:
            code = match.split("<AfterLine/>")[0]
            line_number = int(match.split("<AfterLine/>")[1].strip())
            add_lines[line_number] = code
        except:
            print("Error parsing insert: " , match)


    delete_pattern = re.compile(r'<Delete>(.*?)</Delete>')
    start_line_pattern = re.compile(r'<StartLine>(.*?)</StartLine>')
    end_line_pattern = re.compile(r'<EndLine>(.*?)</EndLine>')

    delete_matches = delete_pattern.findall(modification_xml)
    for match in delete_matches:
        try:
            start_line_matches =  start_line_pattern.findall(match)
            end_line_matches = end_line_pattern.findall(match)
            start_line_match = int(start_line_matches[0].strip())
            end_line_match = int(end_line_matches[0].strip())
            for i in range(start_line_match, end_line_match + 1):
                delete_lines.append(i)
        except:
            print("Error parsing delete:", match)
            continue
    for index, line in enumerate(original_code.splitlines()):
        if index + 1 in delete_lines:
            continue
        new_code.append(line)
        if index + 1 in add_lines:
            new_code.append(add_lines[index])

    return "\n".join(new_code)
   
def add_ir_modifications_to_code(original_code, modification_xml):
    # Do the similar thing as search-replace blocks
    new_code = "<TOP/>\n" + original_code # Since insertions are happening below, use this to make things easier.

    delete_pattern = re.compile(r'<Delete>(.*?)</Delete>')
    insert_pattern = re.compile(r'<Insert>(.*?)</Insert>')

    for match in delete_pattern.findall(modification_xml):
        try:
            delete_lines_match = diff_utils.find_best_match(match, original_code)
            new_code = new_code.replace(delete_lines_match.block, "")
        except:
            print("Error parsing code deletion: ", match)

    for match in insert_pattern.findall(modification_xml):
        try: 
            new_lines = match.split("<After>")[0]
            existing_lines = match.split("<After>")[1]

            existing_lines_match = diff_utils.find_best_match(existing_lines, original_code)
            new_code = new_code.replace(
                existing_lines_match.block,
                existing_lines_match.block + "\n" + new_lines
            )
        except:
            print("Error parsing code modification: ", match)

    return new_code

def extract_code_block_for_direct_modifications(response):
    try:
        return response.split("```python")[2].split("```")[0].strip()
    except:
        # print("Failed to parse code block: ", response)
        return response.split("with ```).")[1].strip()

def run_python_file_with_timeout(file_path, timeout):
    try:
        # Run the python file with a timeout
        result = subprocess.run(
            ["python", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True  # To get the output as a string
        )
        # Return the standard output and error
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return "The process timed out.", ""
    except Exception as e:
        return "", str(e)

def main(hf_model_id: str, model_type: OutputEnum, output_folder: str):
    dataset = load_dataset("nuprl/CanItEdit", split="test")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)

    model_type = model_type.value

    output_data_json = {}
    token_count = 0
    accurate_count = 0
    
    for row in tqdm(dataset):
        old_contents = f"<TOP/>\n{row['before']}"

        file_path = os.path.join(output_folder, f"{row['id']}_direct.txt")
        output_file = open(file_path, "r")
        output = output_file.read()
        output_file.close()
        # print(output)
        # Take the output, save it, run it, and then check that it worked
        if not(os.path.exists(output_folder)):
            os.makedirs(output_folder)

        new_code = ""
        if model_type == "line":
            new_code = add_line_modifications_to_code(old_contents, output)
        elif model_type == "ir":
            new_code = add_ir_modifications_to_code(old_contents, output)
        elif model_type == "whole":
            new_code = extract_code_block_for_direct_modifications(output)

        new_code = new_code.replace("<TOP/>", "")

        test_file = open("test.py", "w+")
        test_file.write(new_code + f"\n{row['tests']}\n" + "print('SUCCESS')")
        execution_output, error = run_python_file_with_timeout("test.py", 7)

        out_file = open(os.path.join(output_folder, f"{row['id']}_processed.txt"), "w+")
        out_file.write(new_code)
        out_file.close()

        # Evaluate the response length in number of tokens
        tokens = tokenizer(output)
        # print(len(tokens))

        output_data_json[row["id"]] = {}
        output_data_json[row["id"]]["length"] = len(tokens[0])
        output_data_json[row["id"]]["correct"] = ("SUCCESS" in execution_output)
        
        token_count += len(tokens[0])
        
        if "SUCCESS" in execution_output:
            accurate_count += 1
    
    out_file = open(os.path.join(output_folder, "data.json"), "w+")
    out_file.write(json.dumps(output_data_json, indent=4))
    out_file.close()

    print("Average token count in output: ", token_count / len(output_data_json))
    print("Number correct: ", accurate_count, "/", len(output_data_json))

if __name__ == "__main__":
    typer.run(main)
