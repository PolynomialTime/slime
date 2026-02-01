#!/usr/bin/env python3
"""
验证数据格式是否符合 slime 框架标准
"""
import json
from pathlib import Path


def verify_data_format(file_path):
    """验证单个文件的数据格式"""
    print(f"\n{'='*60}")
    print(f"验证文件: {file_path}")
    print(f"{'='*60}")
    
    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return False
    
    required_fields = ["text"]
    optional_fields = ["label", "metadata"]
    
    valid_count = 0
    invalid_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                # 检查必需字段
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    print(f"❌ 行 {line_num}: 缺少必需字段 {missing_fields}")
                    invalid_count += 1
                    continue
                
                # 检查字段类型
                if not isinstance(data.get("text"), str):
                    print(f"❌ 行 {line_num}: 'text' 字段必须是字符串")
                    invalid_count += 1
                    continue
                
                if "label" in data and not isinstance(data["label"], str):
                    print(f"❌ 行 {line_num}: 'label' 字段必须是字符串")
                    invalid_count += 1
                    continue
                
                valid_count += 1
                
                # 显示前几条数据的示例
                if line_num <= 3:
                    print(f"\n✅ 行 {line_num} 格式正确:")
                    print(f"  - text: {data['text'][:100]}...")
                    if "label" in data:
                        print(f"  - label: {data['label'][:100]}...")
                
            except json.JSONDecodeError as e:
                print(f"❌ 行 {line_num}: JSON 解析错误 - {e}")
                invalid_count += 1
    
    print(f"\n{'='*60}")
    print(f"验证结果:")
    print(f"  ✅ 有效数据: {valid_count} 条")
    print(f"  ❌ 无效数据: {invalid_count} 条")
    
    if invalid_count == 0:
        print(f"  🎉 所有数据格式正确，符合 slime 框架标准！")
    else:
        print(f"  ⚠️  发现 {invalid_count} 条格式错误的数据")
    
    print(f"{'='*60}")
    
    return invalid_count == 0


def main():
    """验证所有处理后的数据文件"""
    output_dir = Path("hh-rlhf-processed")
    
    if not output_dir.exists():
        print(f"❌ 目录不存在: {output_dir}")
        print("请先运行 prepare_hh_rlhf.py 生成数据")
        return
    
    print("\n" + "="*60)
    print("验证 HH-RLHF 数据格式是否符合 slime 框架标准")
    print("="*60)
    print("\nSlime 框架标准格式:")
    print('  {"text": "prompt内容", "label": "参考答案（可选）"}')
    print("\n必需字段: text")
    print("可选字段: label, metadata")
    
    # 验证所有数据文件
    files_to_check = [
        "helpful-base-train.jsonl",
        "helpful-base-test.jsonl",
        "harmless-base-train.jsonl",
        "harmless-base-test.jsonl",
    ]
    
    all_valid = True
    for filename in files_to_check:
        file_path = output_dir / filename
        if file_path.exists():
            is_valid = verify_data_format(file_path)
            all_valid = all_valid and is_valid
        else:
            print(f"\n⚠️  文件不存在: {file_path}")
    
    print("\n" + "="*60)
    if all_valid:
        print("🎉 所有数据文件格式验证通过！")
        print("✅ 数据已符合 slime 框架标准，可以开始训练")
    else:
        print("❌ 部分数据文件格式验证失败")
        print("请检查错误信息并重新生成数据")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
