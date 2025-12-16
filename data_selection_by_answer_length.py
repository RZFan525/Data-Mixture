import json
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm

def count_tokens(text, tokenizer):
    try:
        # 使用tokenizer将文本转换为token并计算数量
        tokens = tokenizer.encode(text)
        token_count = len(tokens)
        
        return token_count
    except Exception as e:
        print(f"发生错误: {e}")
        return -1

def process_jsonl_file(input_path, output_path, tokenizer_path="/inspire/hdd/global_user/liupengfei-24025/rzfan/models/Qwen2.5-7B"):
    """
    处理JSONL文件，计算token长度，排序并保存前17万个最长的数据
    
    Args:
        input_path: 输入JSONL文件路径
        output_path: 输出JSONL文件路径
        tokenizer_path: tokenizer模型路径
    """
    
    # 加载tokenizer
    print("正在加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    print("tokenizer加载完成!")
    
    # 读取JSONL文件并计算token长度
    data_list = []
    print("正在读取文件并计算token长度...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="处理中")):
            try:
                item = json.loads(line.strip())
                
                # 检查是否包含目标字段

                response_text = item["Qwen2.5_72B_Instruct_Response"][0]
                
                # 计算token长度
                length = count_tokens(response_text, tokenizer)
                
                if length != -1:  # 只有成功计算长度的才保留
                    item["length"] = length
                    data_list.append(item)
                else:
                    print(f"第{line_num + 1}行计算token长度失败")
                    
            except json.JSONDecodeError as e:
                print(f"第{line_num + 1}行JSON解析错误: {e}")
            except Exception as e:
                print(f"第{line_num + 1}行处理错误: {e}")
    
    print(f"总共处理了 {len(data_list)} 个有效数据项")
    
    # 按照length从大到小排序
    print("正在排序...")
    data_list.sort(key=lambda x: x["length"], reverse=True)
    normal_idx = 0
    for d in data_list:
        if d["length"] > 10000:
            normal_idx += 1
        else:
            break
    data_list = data_list[normal_idx:]
    print(f"remove extreme long data: {normal_idx}")
    # 取前k个数据
    top_170k = data_list[:436386]
    actual_count = len(top_170k)
    
    print(f"取前 {actual_count} 个数据项 (最多17万个)")
    
    if actual_count > 0:
        print(f"最长token数: {top_170k[0]['length']}")
        print(f"最短token数: {top_170k[-1]['length']}")
    
    # 保存到输出文件
    print("正在保存文件...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in tqdm(top_170k, desc="保存中"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"处理完成! 已保存 {actual_count} 个数据项到 {output_path}")

def main():
    input_path = "/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/distill_answer_Qwen2.5_72B_instruct/natural_reasoning/final_data/natural_reasoning_distill_qwen_72b.jsonl"
    output_path = "/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/distill_answer_Qwen2.5_72B_instruct/natural_reasoning/final_data/natural_reasoning_distill_qwen_72b_selection_long_length_436386.jsonl"
    parser = argparse.ArgumentParser(description="处理JSONL文件，计算token长度并筛选前n个最长数据")
    parser.add_argument("--input_path", help="输入JSONL文件路径", default=input_path)
    parser.add_argument("--output_path", help="输出JSONL文件路径", default=output_path)
    parser.add_argument("--tokenizer_path", 
                       default="/inspire/hdd/global_user/liupengfei-24025/rzfan/models/Qwen2.5-7B",
                       help="tokenizer模型路径")
    
    args = parser.parse_args()
    
    process_jsonl_file(args.input_path, args.output_path, args.tokenizer_path)

if __name__ == "__main__":
    main()