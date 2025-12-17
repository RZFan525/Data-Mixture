import os
import json
from glob import glob
from tqdm import tqdm

import re

def extract_last_score_alternative(text):
    # 找出所有的分数
    pattern = r"(?:Score|评分|得分)\**[:：][\s\n*]*(10|[0-9])"
    matches = re.findall(pattern, text)
    
    if matches:
        return int(matches[-1])  # 返回最后一个匹配的分数
    
    pattern2 = r"\\boxed{(10|[0-9])}"
    matches2 = re.findall(pattern2, text)
    
    if matches2:
        return int(matches2[-1])  # 返回最后一个\boxed{}中的数字
    
    return None  # 如果两种模式都没找到，返回None


def process_jsonl_files(root_dir):
    # List to store all data
    all_data = []
    
    # Find all jsonl files in the root directory and its subdirectories
    jsonl_files = glob(os.path.join(root_dir, "**", "*.jsonl"), recursive=True)
    
    print(f"Found {len(jsonl_files)} JSONL files to process")
    
    # Process each file
    for file_path in jsonl_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        # Parse the JSON object
                        item = json.loads(line.strip())
                        
                        # Add to our list of all data
                        all_data.append(item)
                    except json.JSONDecodeError:
                        print(f"Error parsing JSON in file {file_path}")
                        continue
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
    
    print(f"Loaded {len(all_data)} total items from all files")
    
    # Filter data based on "Result: Yes"
    new_data = []
    unresolve_number = 0
    for item in tqdm(all_data):
        # Extract the refineqa_output content
        eval_results_list = item["Qwen2.5_32B_Instruct_Eval"]
        
        # Initialize empty strings for question and answer
        model_eval_score = []

        for eval_result in eval_results_list:
            score = extract_last_score_alternative(eval_result)
            if score != None:
                model_eval_score.append(score)
            else:
                model_eval_score.append(-1)
                unresolve_number += 1
        item["model_eval_score_list"] = model_eval_score
        valid_score = [sc for sc in model_eval_score if sc != -1]
        item["model_eval_average_score"] = sum(valid_score) / len(valid_score) if len(valid_score) > 0 else 0
        
        # Add the new item to new_data
        new_data.append(item)
    print(f"unresolve_number: {unresolve_number}")
    return new_data

def save_filtered_data(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(data)} items to {output_file}")

if __name__ == "__main__":
    # Directory containing the economics data
    # subject = "mathpile"
    # root_directory = f"/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/model_eval_Qwen2.5_32B_Instruct/original_output_36w"
    root_directory = "/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/model_eval_Qwen2.5_32B_Instruct/textbook_reasoning_v3_distill_cot_filtering_qa_decontamination/original_data"
    
    # Process all the files
    new_data = process_jsonl_files(root_directory)
    
    # Save the filtered data
    target_path = "/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/model_eval_Qwen2.5_32B_Instruct/textbook_reasoning_v3_distill_cot_filtering_qa_decontamination/final_data"
    save_filtered_data(new_data, target_path + "/final_data.jsonl")