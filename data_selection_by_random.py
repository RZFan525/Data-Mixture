import json
import random
from tqdm import tqdm

def data_selection(file_path, output_path, number):
    
    all_data = []
    # Open and read the JSONL file
    with open(file_path, 'r', encoding='utf-8') as file:
        # Process each line as a separate JSON object
        for line in tqdm(file):
            # Parse the JSON object
            data = json.loads(line)
            # Extract the question_quality_score
            all_data.append(data)
    print(f"all {len(all_data)}")
    
    random.shuffle(all_data)
    output_data = all_data[:number]
    
    del all_data
    print(f"select {len(output_data)}")
    
    with open(output_path, 'w', encoding='utf-8') as file:
        for d in output_data:
            file.write(json.dumps(d, ensure_ascii=False) + "\n")
    
    print("finish!")
    return

# Example usage
if __name__ == "__main__":

    # natural_reasoning
    jsonl_file_path = f"/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/model_eval_Qwen2.5_32B_Instruct/natural_reasoning/final_data/natural_reasoning_model_eval_Qwen2.5_32B.jsonl"
    output_path = f"/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/model_eval_Qwen2.5_32B_Instruct/natural_reasoning/final_data/natural_reasoning_model_eval_Qwen2.5_32B_random_selection_436386.jsonl"
        # print(subject)
    number = 436386
    
    # textbook reasoning
    # jsonl_file_path = f"/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/model_eval_Qwen2.5_32B_Instruct/textbook_reasoning_v3_distill_cot_filtering_qa_decontamination/final_data/final_data.jsonl"
    # output_path = f"/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/model_eval_Qwen2.5_32B_Instruct/textbook_reasoning_v3_distill_cot_filtering_qa_decontamination/final_data/textbook_reasoning_random_selection_297634.jsonl"
    #     # print(subject)
    # number = 297634
    
    # Nemotron
    # jsonl_file_path = f"/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/model_eval_Qwen2.5_32B_Instruct/nemotro_science/final_data/nemotro_science_model_eval_Qwen2.5_32B.jsonl"
    # output_path = f"/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/model_eval_Qwen2.5_32B_Instruct/nemotro_science/final_data/nemotro_science_model_eval_Qwen2.5_32B_random_selection_17w.jsonl"
    #     # print(subject)
    # number = 170000
    
    data_selection(jsonl_file_path, output_path, number)