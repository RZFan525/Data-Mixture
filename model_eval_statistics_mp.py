import json
from tqdm import tqdm

def data_statistics(score):

    print(f"number: {len(score)}")

    intervals = [(i, i + 1) for i in range(10)]  # 产生 [(0,1), (1,2), ..., (9,10)]

    # 统计每个区间的数量
    interval_counts = {f"{start}-{end}": 0 for start, end in intervals}

    for s in score:
        for start, end in intervals:
            if start <= s < end or (start == 9 and s == 10):  # 特殊处理10分属于9-10区间
                interval_counts[f"{start}-{end}"] += 1
                break

    # 输出每个区间的数量
    print("每个区间的数量:")
    for interval, count in interval_counts.items():
        print(f"区间 {interval}: {count} 个")

    # 计算平均分
    average_score = sum(score) / len(score)
    print(f"\n平均分: {average_score:.2f}")

    # 计算中位数
    sorted_scores = sorted(score)
    n = len(sorted_scores)
    if n % 2 == 0:  # 如果列表长度为偶数
        median_score = (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
    else:  # 如果列表长度为奇数
        median_score = sorted_scores[n // 2]
    print(f"中位数: {median_score}")


file_path = "/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/model_eval_Qwen2.5_32B_Instruct/textbook_reasoning_v3_distill_cot_filtering_qa_decontamination/final_data/final_data.jsonl"

# data_sub = {}

# with open(file_path, "r", encoding="utf-8") as fp:
#     for line in fp:
#         item = json.loads(line)
#         if item["subject"] not in data_sub.keys():
#             data_sub[item["subject"]] = []
#         data_sub[item["subject"]].append(item["model_eval_average_score"])

#         del item


# for sub, score in data_sub.items():
#     print()
#     print("=" * 50 + sub + "=" * 50)
#     data_statistics(score)
data = []
with open(file_path, "r", encoding="utf-8") as fp:
    for line in tqdm(fp):
        item = json.loads(line)
        data.append(item["model_eval_average_score"])
        del item

data_statistics(data)