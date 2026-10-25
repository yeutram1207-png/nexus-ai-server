import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Nexus AI Server (Free Direct REST)", version="1.0")

# Lấy Gemini API Key miễn phí từ Google AI Studio
api_key = os.environ.get("GEMINI_API_KEY")

class AIRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"message": "Nexus Gemini Free REST Server is live!"}

@app.post("/api/nexus-ai")
def process_nexus_ai(request: AIRequest):
    if not api_key:
        raise HTTPException(status_code=500, detail="Gemini API Key chưa được cấu hình trên server.")
    
    # Endpoint chuẩn REST API của Google Gemini
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    payload = {
        "contents": [{
            "parts": [{"text": request.prompt}]
        }],
        "systemInstruction": {
            "parts": [{"text": "You are Nexus AI, a professional iOS reverse engineering and tweak development assistant."}]
        }
    }
    
    try:
        res = requests.post(url, json=payload, timeout=20)
        data = res.json()
        
        if res.status_code != 200:
            raise HTTPException(status_code=500, detail=str(data))
            
        answer = data["candidates"][0]["content"]["parts"][0]["text"]
        return {"status": "success", "response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
