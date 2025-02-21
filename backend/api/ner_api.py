from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.post("/recognize")
async def ner_recognize(req: TextRequest):
    # 实际应替换为您的NER模型调用
    mock_entities = [
        {"name": "黄帝", "type": "人物", "start": 0, "end": 2},
        {"name": "轩辕之丘", "type": "地点", "start": 5, "end": 10}
    ]
    return {"entities": mock_entities}