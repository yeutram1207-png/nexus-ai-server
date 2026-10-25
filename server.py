import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="Nexus AI Server (Gemini)", version="1.0")

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
    
    try:
        # Gọi trực tiếp 1 model ổn định nhất để trả kết quả ngay lập tức, tránh timeout
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=request.prompt,
            config=types.GenerateContentConfig(
                system_instruction="You are Nexus AI, a professional iOS reverse engineering and tweak development assistant.",
            ),
        )
        return {"status": "success", "response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
