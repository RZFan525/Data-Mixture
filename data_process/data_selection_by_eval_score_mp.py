import json
from tqdm import tqdm


def data_selection(file_path, output_path, threshold=5):
    subject_number = {}
    select_number = 0
    # Open and read the JSONL file
    with open(output_path, 'w', encoding='utf-8') as writer:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Process each line as a separate JSON object
            for line in tqdm(file):
                # Parse the JSON object
                data = json.loads(line)
                if data["subject"] not in subject_number.keys():
                    subject_number[data["subject"]] = 0
                # Extract the question_quality_score
                score = data.get("model_eval_average_score")
                if score <= threshold:
                    writer.write(json.dumps(data, ensure_ascii=False) + "\n")
                    subject_number[data["subject"]] += 1
                    select_number += 1
                del data
    print(f"select {select_number}")
    print(subject_number)
    print("finish!")
    return

def data_selection_two_threshold(file_path, output_path, low_threshold=1, high_threshold=9):
    select_number = 0
    # Open and read the JSONL file
    with open(output_path, 'w', encoding='utf-8') as writer:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Process each line as a separate JSON object
            for line in tqdm(file):
                # Parse the JSON object
                data = json.loads(line)
                # Extract the question_quality_score
                score = data.get("model_eval_average_score")
                # if score < 1:
                #     print(data)
                if score < high_threshold and score > low_threshold:
                    writer.write(json.dumps(data, ensure_ascii=False) + "\n")
                    select_number += 1
                del data
    print(f"select {select_number}")
    print("finish!")
    return


# Example usage
if __name__ == "__main__":
    # Replace with your actual file path

    # for subject in subjects:
    jsonl_file_path = f"data/natural_reasoning_model_eval_Qwen2.5_32B.jsonl"
    output_path = f"data/natural_reasoning_model_eval_Qwen2.5_32B_difficulty_selection_1_9_.jsonl"
    # print(subject)
    data_selection_two_threshold(jsonl_file_path, output_path)