import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Nexus AI Server", version="1.0")

api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

class AIRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"message": "Hello, World!"}

@app.post("/api/nexus-ai")
def process_nexus_ai(request: AIRequest):
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API Key not configured on server.")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are Nexus AI, a professional iOS reverse engineering and tweak development assistant."},
                {"role": "user", "content": request.prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        answer = response.choices[0].message.content
        return {"status": "success", "response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
