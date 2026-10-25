import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Nexus AI Server (Optimized)", version="1.0")

api_key = os.environ.get("GEMINI_API_KEY")

class AIRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"message": "Nexus Optimized Server is live!"}

@app.post("/api/nexus-ai")
def process_nexus_ai(request: AIRequest):
    if not api_key:
        raise HTTPException(status_code=500, detail="API Key chưa được cấu hình.")
    
    # Sử dụng model gemini-1.5-flash với endpoint v1 ổn định nhất
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    payload = {
        "contents": [{
            "parts": [{"text": request.prompt}]
        }]
    }
    
    try:
        res = requests.post(url, json=payload, timeout=12)
        data = res.json()
        
        if res.status_code == 200 and "candidates" in data:
            answer = data["candidates"][0]["content"]["parts"][0]["text"]
            return {"status": "success", "response": answer}
        else:
            # Trả thẳng lỗi chi tiết từ Google về app để thấy rõ nguyên nhân nếu có
            error_msg = data.get("error", {}).get("message", str(data))
            raise HTTPException(status_code=500, detail=error_msg)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
