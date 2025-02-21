from fastapi import FastAPI
import openai

app = FastAPI()
openai.api_key = "sk-your-key"


@app.post("/ask")
async def llm_ask(question: str, context: dict):
    prompt = f"""基于以下知识：
    {context}

    请回答：{question}
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return {"answer": response.choices[0].message.content}