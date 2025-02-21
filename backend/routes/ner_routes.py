# backend/routes/ner_routes.py
from flask import Blueprint, request, jsonify
import torch
from transformers import BertTokenizerFast
from models.bert_crf_model import BERT_CRF

# 加载模型和分词器
model_path = "../models/ner_model"
tokenizer_path = "../models/ner_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERT_CRF.from_pretrained(model_path)
model.to(device)
tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

# 定义标签映射
id2label = {0: 'O', 1: 'B-NR', 2: 'I-NR', 3: 'E-NR', 4: 'S-NR', 5: 'B-NS', 6: 'I-NS', 7: 'E-NS', 8: 'S-NS', 9: 'B-NB', 10: 'I-NB', 11: 'E-NB', 12: 'S-NB', 13: 'B-NO', 14: 'I-NO', 15: 'E-NO', 16: 'S-NO', 17: 'B-NG', 18: 'I-NG', 19: 'E-NG', 20: 'S-NG', 21: 'B-T', 22: 'I-T', 23: 'E-T', 24: 'S-T'}

# 创建蓝图
ner_bp = Blueprint("ner", __name__)


@ner_bp.route("/ner", methods=["POST"])
def ner():
    data = request.json
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "输入文本不能为空"}), 400

    # 分词并预测
    inputs = tokenizer(list(text), is_split_into_words=True, max_length=512, truncation=True, padding=True, return_tensors="pt")

    # 移除 token_type_ids（如果存在）
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取 predictions
    predictions = outputs["predictions"][0]  # 假设已经是列表

    # 跳过 [CLS] 和 [SEP]
    labels = []
    for i, pred in enumerate(predictions):
        if i == 0 or i == len(predictions) - 1:  # 跳过第一个和最后一个
            continue
        labels.append(id2label[pred])

    # 修改后端返回的数据格式
    token_label_pairs = [{"char": t, "label": l} for t, l in zip(list(text), labels)]
    return jsonify(token_label_pairs)
