import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Nexus Groq Server", version="1.0")

# Lấy Groq API Key từ biến môi trường
api_key = os.environ.get("GROQ_API_KEY")

# Kết nối với endpoint miễn phí của Groq
client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

class AIRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"message": "Nexus Groq Free Server is live!"}

@app.post("/api/nexus-ai")
def process_nexus_ai(request: AIRequest):
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY chưa được cấu hình trên Render.")
    
    try:
        # Dùng model Llama 3.3 70B siêu nhanh và hoàn toàn miễn phí
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are Nexus AI, a professional iOS reverse engineering and tweak development assistant."},
                {"role": "user", "content": request.prompt}
            ]
        )
        answer = response.choices[0].message.content
        return {"status": "success", "response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
