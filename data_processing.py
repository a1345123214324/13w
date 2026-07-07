from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm

"""
使用训练好的 BERT 分类模型对过滤后的评论进行批量预测，
支持情感分类（3 类）和话题分类（6 类）。
"""

# 模型路径
EMOTION_FILE = "../comment_model/sentiment_model"
TOPIC_FILE = "../comment_model/topic_model"

# 输入文件
INPUT_DIR = "../data_filtered"
EMOTION_OUTPUT_DIR = "../data_filtered/emotion"
TOPIC_OUTPUT_DIR = "../data_filtered/topic"

class ModelPredict:
    def __init__(self, comment_path, model_path, output_path, batch_size=32):
        """
        参数:
            comment_path: 待预测的 CSV 路径
            model_path: 模型权重目录路径
            output_path:预测结果保存路径
            batch_size:批量推理的 batch 大小（默认 32）
        """
        self.comment_path = comment_path
        self.output_path = output_path
        self.batch_size = batch_size

        # 自动选择设备（GPU 优先）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"正在加载模型到设备: {self.device} ...")

        # 加载 tokenizer 和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)

        # 切换为评估模式（禁用 dropout 等）
        self.model.eval()

        # 读取待预测数据
        self.df = pd.read_csv(comment_path)

    def predict(self):
        """
        执行批量预测。

        流程:
          1. 将评论按 batch_size 分组
          2. Tokenizer 编码 → 送入 GPU → softmax → argmax 获取标签
          3. 将标签写入原 DataFrame 并保存为 CSV
        """
        print(f"开始批量预测，总数据量: {len(self.df)} 条…")

        texts = self.df['评论内容'].astype(str).tolist()
        all_labels = []

        # 按 batch 切片进行推理
        for i in tqdm(range(0, len(texts), self.batch_size), desc="预测进度"):
            batch_texts = texts[i: i + self.batch_size]

            # Tokenizer 编码：padding + truncation 到 128 长度
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding=True
            ).to(self.device)

            # 推理（不计算梯度）
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                # 取概率最高的类别标签
                labels = torch.argmax(probs, dim=-1).cpu().tolist()

            all_labels.extend(labels)

        # 将预测标签写入 DataFrame 并保存
        self.df['label'] = all_labels
        result_df = self.df[['评论内容', 'label']]

        result_df.to_csv(self.output_path, index=False, encoding='utf-8-sig')
        print(f"预测完成！结果已保存至: {self.output_path}")

def predict_comments():
    # 情感预测
    input_files = [
        Path(INPUT_DIR) / "comments_weibo_filtered.csv",
        Path(INPUT_DIR) / "comments_reddit_filtered.csv",
        Path(INPUT_DIR) / "comments_bilibili_filtered.csv"
    ]

    emotion_output_files = [
        Path(EMOTION_OUTPUT_DIR) / "comments_weibo_sentiment.csv",
        Path(EMOTION_OUTPUT_DIR) / "comments_reddit_sentiment.csv",
        Path(EMOTION_OUTPUT_DIR) / "comments_bilibili_sentiment.csv"
    ]

    topic_output_files = [
        Path(TOPIC_OUTPUT_DIR) / "comments_weibo_topic.csv",
        Path(TOPIC_OUTPUT_DIR) / "comments_reddit_topic.csv",
        Path(TOPIC_OUTPUT_DIR) / "comments_bilibili_topic.csv"
    ]

    for i in range(3):
        predictor = ModelPredict(input_files[i], EMOTION_FILE,emotion_output_files[i], batch_size=32)
        predictor.predict()

    for i in range(3):
        predictor = ModelPredict(input_files[i], TOPIC_FILE,topic_output_files[i], batch_size=32)
        predictor.predict()

if __name__ == "__main__":
    predict_comments()
