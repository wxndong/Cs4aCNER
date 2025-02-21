好的！根据你的需求，我们将后端框架从FastAPI改为Flask，并选择一个合适的数据库来存储和管理数据。以下是重新设计的Prompt，确保符合你的项目需求。

---

# **第一步：明确项目架构**
## **Prompt示例**：
```
我正在开发一个名为CS4ACNER的项目，采用前后端分离架构，前端使用Vue，后端使用Flask，并通过Docker保证环境一致性。
项目的目录结构如下：
├── data/                   # 原始数据与预处理
│   ├── raw/               # 爬取的100篇原始文本
│   └── kg/                # 知识图谱数据
│       ├── nodes.csv
│       └── relationships.csv
├── models/
│   ├── ner_model/         # 训练好的NER模型
├── backend/               # 服务端代码
│   ├── Dockerfile
│   ├── app.py             # Flask主应用
│   ├── routes/            # 路由模块
│   │   ├── ner_routes.py  # NER模块路由
│   │   ├── kg_routes.py   # KG模块路由
│   │   └── llm_routes.py  # LLM模块路由
│   ├── database/          # 数据库相关代码
│       └── db_conn.py     # 数据库连接配置
├── frontend/              # 交互界面
│   ├── Dockerfile
├── docs/                  # 文档
├── README.md
├── configs/               # 配置文件
├── requirements.txt       # Python依赖
└── docker-compose.yml     # 容器化部署

数据库选择：
- 对于知识图谱（KG）模块，我们使用Neo4j，因为它擅长处理图数据。
- 对于其他模块的简单数据存储，我们使用SQLite（轻量级、易于集成）。

请确认你理解了这个架构，并准备根据这个架构逐步实现功能。
```

## **目标**：确保AI助手对项目的整体架构有清晰的理解，并在此基础上进行后续开发。

## 结果：













---

# **第二步：模块1 - 古汉语NER模块**
## **Prompt示例**：
```
我们先实现模块1：古汉语NER模块。
- 功能需求：
  1. 前端接收用户输入的古汉语文本。
  2. 后端调用训练好的BERT+CRF模型（路径：models/ner_model/）进行实体识别。
  3. 返回JSON格式的数据，包含每个字符的标签（BIOES格式）。
  4. 前端根据返回的JSON数据，按实体类型高亮显示文本。
  5. 支持用户自定义选择高亮哪些实体类型（默认高亮所有类型）。
- 实体类别共25类标签包括：
  - BIOES格式的：NB（书名）、NR（人名）、NO（官职）、NG（地名）、NS（朝代）、T（时间）
  - O（非实体）

- 数据流向：
  用户输入 -> 前端发送请求 -> 后端调用模型 -> 返回JSON -> 前端展示结果。
请为我设计以下内容：
1.务必确保你之前的回答中与现在回答的一致性
2. 前端页面的基本HTML和Vue代码框架，支持文本输入和高亮显示。注意给出完整的前端代码包括package.json等
3. 后端的Flask接口代码（在`routes/ner_routes.py`中），加载微调好的BERT+CRF模型（在`models/ner_model`包括config.json, model.safetensors, tokenizer.json）并返回JSON数据（）。
4. 后端主界面`app.py`的内容，然后docker之后统一部署，先不用给出docker相关
5. 请注意说明你给的文件对应项目的什么位置及文件的名称，注意上下文一致
---
模型原型：
# bert_crf_model.py
from transformers import BertPreTrainedModel, BertModel
import torch
import torch.nn as nn
from torchcrf import CRF


class BERT_CRF(BertPreTrainedModel):
    """BERT+CRF模型"""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        # 网络结构
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)
        # 初始化
        self.init_weights()


    def forward(self, input_ids, attention_mask=None, labels=None):
        # BERT编码 (后续的 BERT 层 *需要* attention_mask 参数)
        outputs = self.bert(input_ids, attention_mask=attention_mask)  # BERT 模型整体 forward  *需要* attention_mask
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)

        # CRF处理
        mask = attention_mask.bool() if attention_mask is not None else None
        tags = self.crf.decode(emissions, mask=mask)

        loss = None
        if labels is not None:
            # 确保labels是LongTensor类型
            labels = labels.long()
            # print(labels)
            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')

        return {
            'loss': loss,
            'predictions': tags,
            'attention_mask': mask,
            'logits': emissions  # 将 emissions 添加到返回字典，键名为 'logits'
        }
---
预测脚本参考：
import torch
import logging
from transformers import BertTokenizerFast
from bert_crf_model import BERT_CRF
from bert_crf_data_processing import generate_label_map

# 配置日志记录和设备
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict(model_path, tokenizer_path, test_path, output_path):
    """
    使用 BERT+CRF 模型进行预测。
    """
    # 加载模型和分词器
    try:
        model = BERT_CRF.from_pretrained(model_path)
        model.to(device)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

    try:
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
        logger.info("Tokenizer loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        raise

    # 生成标签映射
    label_map = generate_label_map()
    id2label = {v: k for k, v in label_map.items()}  # 标签 ID 到标签名称的映射

    # 处理测试文件
    with open(test_path, 'r', encoding='utf-8') as f_in, \
            open(output_path, 'w', encoding='utf-8', newline='\n') as f_out:

        for line_idx, line in enumerate(f_in):
            line = line.strip()

            # 如果是空行，直接写入一个空行到输出文件
            if not line:
                f_out.write("\n")
                continue

            # 字符级分割
            chars = list(line)
            total_length = len(chars)
            max_chunk_length = 510  # 512 - 2 ([CLS] 和 [SEP])
            all_labels = []

            logger.info(f"Processing line {line_idx + 1} with {total_length} characters")

            # 分块处理长文本
            for chunk_idx in range(0, total_length, max_chunk_length):
                chunk = chars[chunk_idx: chunk_idx + max_chunk_length]

                # 对分块进行分词
                inputs = tokenizer(
                    chunk,
                    is_split_into_words=True,
                    truncation=True,
                    max_length=512,
                    padding='max_length',
                    return_tensors='pt',
                    return_offsets_mapping=True
                )

                # 将输入移动到设备
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)

                # 模型预测
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask=attention_mask)

                # 提取预测标签
                batch_tags = outputs['predictions']
                if isinstance(batch_tags, torch.Tensor):
                    batch_tags = batch_tags.cpu().tolist()

                word_ids = inputs.word_ids(batch_index=0)
                valid_tags = []
                for word_id, tag_id in zip(word_ids, batch_tags[0]):
                    if word_id is not None:
                        valid_tags.append((word_id, tag_id))

                # 按 word ID 排序并去重
                valid_tags.sort(key=lambda x: x[0])
                chunk_labels = [id2label.get(tag, 'O') for _, tag in valid_tags]

                # 确保预测标签与字符长度一致
                try:
                    assert len(chunk_labels) == len(chunk)
                except AssertionError:
                    logger.error(f"Alignment error: Predicted tags ({len(chunk_labels)}) != Characters ({len(chunk)})")
                    chunk_labels = ['O'] * len(chunk)  # 如果对齐失败，回退到 'O'

                all_labels.extend(chunk_labels)

            # 最终对齐检查
            try:
                assert len(all_labels) == total_length
            except AssertionError:
                logger.error(f"Final alignment error: Labels ({len(all_labels)}) != Characters ({total_length})")
                all_labels = all_labels[:total_length]  # 截断以匹配字符数

            # 写入结果到输出文件
            for char, label in zip(chars, all_labels):
                f_out.write(f"{char}\t{label}\n")

            # 在每句话后添加一个空行
            f_out.write("\n")

            logger.info(f"Line {line_idx + 1} processed successfully.")


if __name__ == "__main__":
    # 配置路径
    md_dir = "../models/bert_crf_BERTlr5e-5_CRFlr5e-3_cosine_F1_91dot7"
    model_path = md_dir
    tokenizer_path = md_dir
    test_file = "../datasets/test/raw/testset_A.txt"  # 测试文件路径
    output_file = "../results/testset_A_bert_crf_prediction.txt"  # 输出文件路径

    predict(model_path, tokenizer_path, test_file, output_file)
```

## **目标**：生成模块1的前后端代码框架，确保符合项目架构和功能需求。













---

# **第三步：模块2 - 知识图谱KG模块**
## **Prompt示例**：
```
接下来是模块2：知识图谱KG模块。
- 功能需求：
  1. 数据存储在`data/kg/nodes.csv`和`data/kg/relationships.csv`中。
  2. 后端需要支持按关键词筛选实体。
  3. 前端以可视化形式展示筛选后的子图。
  4. 用户可以选中某个实体，作为模块3（大模型API问答模块）的prompt。
- 数据流向：
  用户输入关键词 -> 前端发送请求 -> 后端筛选数据 -> 返回子图信息 -> 前端展示子图。

请为我设计以下内容：
1. 后端代码（在`routes/kg_routes.py`中），读取CSV文件并实现关键词筛选逻辑。
2. 前端代码，使用D3.js或其他可视化库展示子图。
3. 数据库连接代码（在`database/db_conn.py`中），配置Neo4j连接。
```

## **目标**：生成模块2的前后端代码框架，确保符合项目架构和功能需求。

---

# **第四步：模块3 - 大模型API问答模块**
## **Prompt示例**：
```
最后是模块3：大模型API问答模块。
- 功能需求：
  1. 提供类似ChatGPT的聊天界面，左侧是机器人，右侧是用户。
  2. 用户输入问题后，后端调用大模型API生成回答。
  3. 支持查看历史会话、创建新会话、打断生成等功能。
  4. 模块1和模块2可以通过划词跳转到此模块，自动生成prompt。
- 数据流向：
  用户输入问题 -> 前端发送请求 -> 后端调用大模型API -> 返回回答 -> 前端展示。

请为我设计以下内容：
1. 前端聊天界面的HTML和Vue代码框架。
2. 后端代码（在`routes/llm_routes.py`中），调用大模型API并返回回答。
3. 使用SQLite存储历史会话数据（在`database/db_conn.py`中配置）。
```

## **目标**：生成模块3的前后端代码框架，确保符合项目架构和功能需求。

---

# **第五步：模块间的交互设计**
## **Prompt示例**：
```
现在我们需要设计模块间的交互。
- 模块1（NER）：
  1. 用户可以选中识别后的部分文本，作为模块2（KG）的搜索关键词。
  2. 用户可以选中识别后的部分文本，作为模块3（LLM）的prompt。
- 模块2（KG）：
  1. 用户可以选中某个实体，作为模块3（LLM）的prompt。
- 数据流向：
  - 模块1 -> 模块2：传递选中的文本作为关键词。
  - 模块1 -> 模块3：传递选中的文本及上下文作为prompt。
  - 模块2 -> 模块3：传递选中的实体及其关联信息作为prompt。

请为我设计这些交互的具体实现方式。
```

## **目标**：明确模块间的数据流向和交互逻辑。

---

# **第六步：整合项目架构**
## **Prompt示例**：
```
现在我们已经有了各个模块的代码框架和交互设计。
请帮我整合整个项目的架构，包括：
1. 如何组织前后端代码目录（基于之前的目录结构）。
2. 如何使用Docker进行容器化部署。
3. 如何编写`docker-compose.yml`文件，包括Flask、Vue、Neo4j和SQLite的配置。
```

## **目标**：生成完整的项目架构和部署方案。

---

# **第七步：测试和优化**
## **Prompt示例**：
```
项目开发完成后，我们需要进行测试和优化。
请为我设计以下测试方案：
1. 如何对NER模块的模型性能进行测试？
2. 如何验证KG模块的筛选功能是否正确？
3. 如何评估LLM模块的回答质量？
```

## **目标**：生成测试方案，确保项目功能正常。

---

# **第八步：文档撰写**
## **Prompt示例**：
```
最后一步是撰写项目文档。
请为我生成以下内容：
1. 一份README.md文件，包含项目简介、安装指南、使用说明和模块介绍。
2. 一份API文档，详细说明每个接口的功能、参数和返回值。
```

## **目标**：生成完整的项目文档。

---

# **总结**
通过以上更详细的分步骤Prompt设计，你可以确保AI助手始终基于明确的项目架构和功能需求生成内容，同时避免遗漏关键细节。如果某一步的返回内容仍然过长，可以进一步细分问题，直到获得满意的答案为止。