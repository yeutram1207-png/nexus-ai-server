import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai

app = FastAPI(title="Nexus AI Server", version="1.0")

# Khởi tạo client Gemini chính thức từ SDK
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

class AIRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"message": "Nexus Official SDK Server is live!"}

@app.post("/api/nexus-ai")
def process_nexus_ai(request: AIRequest):
    if not client:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY chưa được cấu hình trên Render.")
    
    try:
        # Gọi mô hình gemini-2.5-flash qua SDK chính thức
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=request.prompt,
        )
        return {"status": "success", "response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
