#!/usr/bin/env python
# scripts/precompute_embeddings.py
"""
一次性脚本：用CLIP计算所有任务的文本嵌入并保存
"""
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))


def setup_clip():
    """安装并导入CLIP（只在预计算时使用）"""
    try:
        from transformers import CLIPTextModel, CLIPTokenizer
        return CLIPTextModel, CLIPTokenizer
    except ImportError:
        print("安装transformers...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "tokenizers"])
        from transformers import CLIPTextModel, CLIPTokenizer
        return CLIPTextModel, CLIPTokenizer


def compute_weighted_embedding(descriptions, text_encoder, tokenizer, device="cuda"):
    """计算加权平均的CLIP嵌入"""
    valid_texts = []
    weights = []

    for desc in descriptions:
        # 仅使用已验证的描述
        if not desc.get("is_verified", False):
            continue
        # 提取文本和置信度
        text = desc["description"]
        confidence = desc.get("confidence", 1.0) # 默认置信度为1.0
        # if confidence is None:
        #     confidence = 1.0

        valid_texts.append(text)
        weights.append(float(confidence)) # 获取置信度权重

    if not valid_texts:
        return torch.zeros(512, device=device)

    # 编码所有文本，inputs包含input_ids和attention_mask，attention_mask用于指示填充部分
    inputs = tokenizer(
        valid_texts,
        return_tensors="pt", # 返回PyTorch张量
        padding=True, # 填充到最大长度
        truncation=True, # 截断超过最大长度的文本
        max_length=77 # CLIP的最大长度
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # 获取文本嵌入，得到[n, 512]的张量
    with torch.no_grad(): # 冻结模型参数
        outputs = text_encoder(**inputs) # 前向传播，用CLIP的文本编码器编码
        embeddings = outputs.pooler_output  # [n, 512] 池化输出作为文本嵌入

    # 加权平均
    weights = torch.tensor(weights, device=device) # [n]，获取权重张量
    weights = weights / weights.sum() # 归一化权重，不同文本对应不同权重
    weighted_embed = (embeddings * weights.unsqueeze(1)).sum(dim=0) # [512] 加权求和，weights.unsqueeze(1)变成[n, 1]，sum(dim=0)沿第0维求和，融合所有文本嵌入

    return weighted_embed.cpu()  # 移到CPU保存


def main():
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 1. 加载CLIP模型
    print("加载CLIP文本编码器...")
    CLIPTextModel, CLIPTokenizer = setup_clip()

    model_name = "openai/clip-vit-base-patch32"
    text_encoder = CLIPTextModel.from_pretrained(model_name).to(device) # 加载模型到设备
    tokenizer = CLIPTokenizer.from_pretrained(model_name) # 加载分词器

    text_encoder.eval()
    for param in text_encoder.parameters():
        param.requires_grad = False

    # 2. 加载描述文件
    desc_file = "raw_data/text_description/clip_descriptions.json"
    print(f"加载描述文件: {desc_file}")

    with open(desc_file, 'r') as f:
        all_descriptions = json.load(f)

    # 3. 为每个任务计算嵌入
    task_embeddings = {}
    total_tasks = len(all_descriptions)

    for i, (task_file, desc_list) in enumerate(all_descriptions.items(), 1): # 1表示索引从1开始
        # 提取任务ID（去掉.json后缀）
        task_id = task_file.replace('.json', '')

        print(f"处理任务 {i}/{total_tasks}: {task_id}")
        # 计算每个任务的加权嵌入
        embedding = compute_weighted_embedding(
            desc_list, text_encoder, tokenizer, device
        )

        task_embeddings[task_id] = embedding # 保存嵌入

        if i % 20 == 0:
            print(f"  进度: {i}/{total_tasks}")

    # 4. 保存嵌入文件
    output_file = "saves/task_text_embedings/task_clip_embeddings_v2.pt"
    torch.save(task_embeddings, output_file)

    # 计算文件大小
    file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    print(f"\n完成！保存了 {len(task_embeddings)} 个任务的嵌入")
    print(f"文件: {output_file}")
    print(f"大小: {file_size:.2f} MB")

    # 5. 保存元信息（可选），关于数据的一些统计信息，供后续使用
    meta_file = "saves/task_text_embedings/task_clip_embeddings_meta_v2.json"
    meta_info = {
        "num_tasks": len(task_embeddings), # 任务数量
        "embed_dim": 512, # 嵌入维度
        "model": model_name, # 使用的模型
        "size_mb": file_size # 文件大小
    }
    with open(meta_file, 'w') as f:
        json.dump(meta_info, f, indent=2)

    print(f"元信息保存到: {meta_file}")


if __name__ == "__main__":
    main()