import os
import json
import random
from typing import List, Dict


def sample_failed_cases(
    input_path: str,
    output_path: str,
    samples_per_benchmark: int = 4,
    seed: int = 42,
):
    random.seed(seed)

    all_samples: List[Dict] = []

    for benchmark in os.listdir(input_path):
        benchmark_dir = os.path.join(input_path, benchmark)
        if not os.path.isdir(benchmark_dir):
            continue

        pred_file = os.path.join(
            benchmark_dir,
            "infer_logs",
            "sft",
            "samples",
            "predictions.json",
        )

        if not os.path.exists(pred_file):
            print(f"[Skip] {benchmark}: predictions.json not found")
            continue

        try:
            with open(pred_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[Error] Failed to load {pred_file}: {e}")
            continue

        if not isinstance(data, list):
            print(f"[Skip] {benchmark}: predictions.json is not a list")
            continue

        failed_samples = [x for x in data if x.get("accuracy") == 0]

        if not failed_samples:
            print(f"[Info] {benchmark}: no accuracy==0 samples")
            continue

        sampled = random.sample(
            failed_samples,
            min(samples_per_benchmark, len(failed_samples)),
        )

        # # 可选：加上 benchmark 名字，方便回溯
        # for s in sampled:
        #     s["_benchmark"] = benchmark

        all_samples.extend(sampled)
        print(f"[OK] {benchmark}: sampled {len(sampled)} cases")

    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(all_samples)} samples to {output_path}")


if __name__ == "__main__":
    input_path = "/Users/fanrunze/umass/CS685/final_project/megascience_mixture_filtering.lr5e-6.bs512-epoch-3/checkpoint-7344"
    output_path = "error_cases.json"

    sample_failed_cases(
        input_path=input_path,
        output_path=output_path,
        samples_per_benchmark=4,
    )
