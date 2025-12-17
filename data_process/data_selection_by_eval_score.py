import json

def data_selection(file_path, output_path):
    
    output_data = []
    subject_number = {}
    # Open and read the JSONL file
    with open(file_path, 'r', encoding='utf-8') as file:
        # Process each line as a separate JSON object
        for line in file:
            # Parse the JSON object
            data = json.loads(line)
            if data["subject"] not in subject_number.keys():
                subject_number[data["subject"]] = 0
            # Extract the question_quality_score
            score = data.get("model_eval_average_score")
            if score <= 5:
                output_data.append(data)
                subject_number[data["subject"]] += 1
    
    print(f"select {len(output_data)}")
    print(subject_number)
    
    with open(output_path, 'w', encoding='utf-8') as file:
        for d in output_data:
            file.write(json.dumps(d, ensure_ascii=False) + "\n")
    
    print("finish!")
    return

# Example usage
if __name__ == "__main__":
    # Replace with your actual file path
    subjects = [
        "biology",
        "chemistry",
        "cs",
        "economics",
        "math",
        "mathpile",
        "medicine",
        "physics"
    ]
    # for subject in subjects:
    jsonl_file_path = f"/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/model_eval_Qwen2.5_32B_Instruct/final_data_36w/model_eval_Qwen2.5_32B_Instruct_v3_distill_cot_filtering_wrong_question_without_sample.jsonl"
    output_path = f"/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/model_eval_Qwen2.5_32B_Instruct/final_data_36w/v3_distill_cot_filtering_wrong_question_without_sample_difficulty_selection_5.jsonl"
        # print(subject)
    data_selection(jsonl_file_path, output_path)