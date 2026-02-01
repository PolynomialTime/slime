#!/usr/bin/env python3
"""
将 hh-rlhf 数据集转换为 slime 训练格式
格式: {"text": "完整对话历史", "label": "最后一个 Assistant 的回复"}
注意：slime 框架默认使用 "text" 作为 prompt 字段，"label" 作为标签字段
"""
import gzip
import json
import re
from pathlib import Path


def parse_conversation(conversation):
    """
    解析对话，提取所有轮次
    返回: (prompt_history, last_answer)
    """
    # 标准化格式：将 \n\nHuman: 和 \n\nAssistant: 分割
    conversation = conversation.strip()
    
    # 分割对话轮次
    turns = []
    current_role = None
    current_text = []
    
    lines = conversation.split('\n\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 检测角色标记
        if line.startswith('Human:') or line.startswith('H:'):
            if current_role and current_text:
                turns.append((current_role, '\n'.join(current_text)))
            current_role = 'Human'
            # 移除角色标记
            text = re.sub(r'^(Human:|H:)\s*', '', line)
            current_text = [text] if text else []
        elif line.startswith('Assistant:') or line.startswith('A:'):
            if current_role and current_text:
                turns.append((current_role, '\n'.join(current_text)))
            current_role = 'Assistant'
            # 移除角色标记
            text = re.sub(r'^(Assistant:|A:)\s*', '', line)
            current_text = [text] if text else []
        else:
            # 继续当前角色的文本
            if current_role:
                current_text.append(line)
    
    # 添加最后一轮
    if current_role and current_text:
        turns.append((current_role, '\n'.join(current_text)))
    
    if not turns:
        return None, None
    
    # 提取最后一个 Assistant 的回复作为 answer
    last_answer = None
    if turns[-1][0] == 'Assistant':
        last_answer = turns[-1][1]
        turns = turns[:-1]  # 移除最后一个 Assistant 回复
    
    # 构建 prompt（包含之前的所有对话历史）
    prompt_parts = []
    for role, text in turns:
        if role == 'Human':
            prompt_parts.append(f"Human: {text}")
        else:
            prompt_parts.append(f"Assistant: {text}")
    
    prompt = '\n\n'.join(prompt_parts)
    
    return prompt, last_answer


def convert_hh_rlhf_to_slime_format(input_file, output_file):
    """
    将 hh-rlhf 格式转换为 slime 格式
    
    hh-rlhf 格式: {"chosen": "...", "rejected": "..."}
    slime 格式: {"text": "对话历史", "label": "最后的回复"}
    """
    output_data = []
    skipped = 0
    
    with gzip.open(input_file, 'rt', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            
            # 解析 chosen 对话
            prompt, answer = parse_conversation(data['chosen'])
            
            if prompt and answer:
                output_data.append({
                    "text": prompt,
                    "label": answer
                })
            else:
                skipped += 1
    
    # 保存为 jsonl 格式
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"转换完成: {len(output_data)} 条数据")
    if skipped > 0:
        print(f"跳过: {skipped} 条数据（解析失败）")
    print(f"输出文件: {output_file}")



def main():
    # 设置路径
    base_dir = Path("hh-rlhf")
    output_dir = Path("hh-rlhf-processed")
    output_dir.mkdir(exist_ok=True)
    
    # 转换不同的数据集
    datasets = [
        ("helpful-base", "train"),
        ("helpful-base", "test"),
        ("harmless-base", "train"),
        ("harmless-base", "test"),
    ]
    
    for dataset_name, split in datasets:
        input_file = base_dir / dataset_name / f"{split}.jsonl.gz"
        if input_file.exists():
            output_file = output_dir / f"{dataset_name}-{split}.jsonl"
            print(f"\n{'='*60}")
            print(f"处理: {input_file}")
            print(f"{'='*60}")
            convert_hh_rlhf_to_slime_format(input_file, output_file)
        else:
            print(f"\n跳过 (文件不存在): {input_file}")
    
    print(f"\n{'='*60}")
    print("所有数据集转换完成！")
    print(f"处理后的数据保存在: {output_dir}")
    print(f"{'='*60}")
    
    # 显示示例数据
    print("\n示例数据预览:")
    example_file = output_dir / "helpful-base-train.jsonl"
    if example_file.exists():
        with open(example_file, 'r', encoding='utf-8') as f:
            example = json.loads(f.readline())
            print(f"\nText (Prompt):\n{example['text'][:200]}...")
            print(f"\nLabel (Answer):\n{example['label'][:200]}...")


if __name__ == "__main__":
    main()
