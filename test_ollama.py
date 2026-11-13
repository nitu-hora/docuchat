import time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama")

start = time.time()
reply = client.chat.completions.create(
    model = "qwen2.5:3b",
    messages = [{"role": "user", "content" : " Explain what RAG is in two sentences."}]
)

print(reply.choices[0].message.content)
print("seconds taken :", round(time.time() - start, 1))