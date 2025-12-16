import os
import json
from glob import glob
from tqdm import tqdm
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import queue

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


def process_single_item(item):
    """处理单个item的函数"""
    unresolve_count = 0
    eval_results_list = item["Qwen2.5_32B_Instruct_Eval"]
    
    model_eval_score = []
    for eval_result in eval_results_list:
        score = extract_last_score_alternative(eval_result)
        if score is not None:
            model_eval_score.append(score)
        else:
            model_eval_score.append(-1)
            unresolve_count += 1
    
    item["model_eval_score_list"] = model_eval_score
    valid_score = [sc for sc in model_eval_score if sc != -1]
    item["model_eval_average_score"] = sum(valid_score) / len(valid_score) if len(valid_score) > 0 else 0
    
    return item, unresolve_count


def process_single_file(file_path):
    """处理单个文件的函数"""
    items = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    items.append(item)
                except json.JSONDecodeError:
                    print(f"Error parsing JSON in file {file_path}")
                    continue
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return []
    
    return items


def process_jsonl_files_threaded(root_dir, output_file, max_workers=4, batch_size=1000):
    """
    使用多线程和流式处理的版本
    
    Args:
        root_dir: 根目录
        output_file: 输出文件路径
        max_workers: 最大线程数
        batch_size: 批处理大小，控制内存使用
    """
    # 找到所有jsonl文件
    jsonl_files = glob(os.path.join(root_dir, "**", "*.jsonl"), recursive=True)
    print(f"Found {len(jsonl_files)} JSONL files to process")
    
    total_items = 0
    total_unresolve = 0
    write_lock = Lock()
    
    # 打开输出文件用于写入
    with open(output_file, 'w', encoding='utf-8') as output_f:
        # 使用线程池处理文件读取
        with ThreadPoolExecutor(max_workers=max_workers) as file_executor:
            # 提交所有文件处理任务
            file_futures = {file_executor.submit(process_single_file, file_path): file_path 
                          for file_path in jsonl_files}
            
            # 处理文件读取结果
            for file_future in tqdm(as_completed(file_futures), total=len(jsonl_files), desc="Processing files"):
                file_path = file_futures[file_future]
                try:
                    items = file_future.result()
                    if not items:
                        continue
                    
                    # 分批处理items以控制内存使用
                    for i in range(0, len(items), batch_size):
                        batch = items[i:i + batch_size]
                        
                        # 使用线程池处理当前批次的items
                        with ThreadPoolExecutor(max_workers=max_workers) as item_executor:
                            item_futures = {item_executor.submit(process_single_item, item): item 
                                          for item in batch}
                            
                            # 收集处理结果
                            batch_results = []
                            batch_unresolve = 0
                            
                            for item_future in as_completed(item_futures):
                                try:
                                    processed_item, unresolve_count = item_future.result()
                                    batch_results.append(processed_item)
                                    batch_unresolve += unresolve_count
                                except Exception as e:
                                    print(f"Error processing item: {e}")
                                    continue
                            
                            # 线程安全地写入结果
                            with write_lock:
                                for item in batch_results:
                                    output_f.write(json.dumps(item, ensure_ascii=False) + '\n')
                                total_items += len(batch_results)
                                total_unresolve += batch_unresolve
                        
                        # 清理内存
                        del batch_results
                        del batch
                    
                    # 清理内存
                    del items
                    
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    continue
    
    print(f"Total processed items: {total_items}")
    print(f"Total unresolved items: {total_unresolve}")
    print(f"Results saved to {output_file}")


def process_jsonl_files_memory_efficient(root_dir, output_file, max_workers=4):
    """
    内存效率更高的版本 - 逐文件流式处理
    """
    jsonl_files = glob(os.path.join(root_dir, "**", "*.jsonl"), recursive=True)
    print(f"Found {len(jsonl_files)} JSONL files to process")
    
    total_items = 0
    total_unresolve = 0
    write_lock = Lock()
    
    with open(output_file, 'w', encoding='utf-8') as output_f:
        for file_path in tqdm(jsonl_files, desc="Processing files"):
            try:
                with open(file_path, 'r', encoding='utf-8') as input_f:
                    items_batch = []
                    
                    for line in input_f:
                        try:
                            item = json.loads(line.strip())
                            items_batch.append(item)
                            
                            # 当批次达到一定大小时处理
                            if len(items_batch) >= 10000:  # 调整批次大小
                                processed_count, unresolve_count = process_batch_threaded(
                                    items_batch, output_f, write_lock, max_workers)
                                total_items += processed_count
                                total_unresolve += unresolve_count
                                items_batch = []
                                
                        except json.JSONDecodeError:
                            print(f"Error parsing JSON in file {file_path}")
                            continue
                    
                    # 处理剩余的items
                    if items_batch:
                        processed_count, unresolve_count = process_batch_threaded(
                            items_batch, output_f, write_lock, max_workers)
                        total_items += processed_count
                        total_unresolve += unresolve_count
                        
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue
    
    print(f"Total processed items: {total_items}")
    print(f"Total unresolved items: {total_unresolve}")
    print(f"Results saved to {output_file}")


def process_batch_threaded(items_batch, output_f, write_lock, max_workers):
    """处理一个批次的items"""
    processed_count = 0
    unresolve_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_item, item): item for item in items_batch}
        
        batch_results = []
        for future in as_completed(futures):
            try:
                processed_item, item_unresolve_count = future.result()
                batch_results.append(processed_item)
                unresolve_count += item_unresolve_count
                processed_count += 1
            except Exception as e:
                print(f"Error processing item: {e}")
                continue
        
        # 线程安全地写入结果
        with write_lock:
            for item in batch_results:
                output_f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return processed_count, unresolve_count




if __name__ == "__main__":
    root_directory = "/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/model_eval_Qwen2.5_32B_Instruct/natural_reasoning/original_data"
    
    # Save the filtered data
    target_path = "/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/model_eval_Qwen2.5_32B_Instruct/natural_reasoning/final_data/natural_reasoning_model_eval_Qwen2.5_32B.jsonl"
    process_jsonl_files_memory_efficient(root_directory, target_path, max_workers=24)
    