import os
import json
from glob import glob
import re

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
    
    for item in all_data:
        
        new_data.append(item)
    
    return new_data

def save_filtered_data(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(data)} items to {output_file}")

if __name__ == "__main__":
    # Directory containing the economics data
    subject = "physics"
    root_directory = f"/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/sampling_answer_Qwen2.5_7B_instruct/naturalreasoning/original_data"
    
    # Process all the files
    new_data = process_jsonl_files(root_directory)
    
    # Save the filtered data
    save_filtered_data(new_data, f"/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/sampling_answer_Qwen2.5_7B_instruct/naturalreasoning/final_data/naturalreasoning_sampling_qwen_7b.jsonl")