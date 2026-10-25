import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="Nexus AI Server (Gemini)", version="1.0")

# Lấy Gemini API Key từ biến môi trường trên Render
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

class AIRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"message": "Nexus Gemini Server is live!"}

@app.post("/api/nexus-ai")
def process_nexus_ai(request: AIRequest):
    if not api_key:
        raise HTTPException(status_code=500, detail="Gemini API Key not configured on server.")
    
    # Danh sách model dự phòng tự động chuyển đổi khi quá tải hoặc lỗi
    models_to_try = ["gemini-3.8-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
    
    last_error = None
    for m in models_to_try:
        try:
            response = client.models.generate_content(
                model=m,
                contents=request.prompt,
                config=types.GenerateContentConfig(
                    system_instruction="You are Nexus AI, a professional iOS reverse engineering and tweak development assistant.",
                ),
            )
            return {"status": "success", "response": response.text}
        except Exception as e:
            last_error = str(e)
            continue
            
    raise HTTPException(status_code=500, detail=f"All models failed: {last_error}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
