from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import os

app = FastAPI()

# Lấy OpenAI API Key từ biến môi trường bảo mật trên Render
openai.api_key = os.environ.get("OPENAI_API_KEY")

class AIRequest(BaseModel):
    function_name: str
    vmaddr: str
    file_offset: str
    query: str
    context_summary: str = ""

@app.post("/api/nexus-ai")
async def process_nexus_ai(req: AIRequest):
    try:
        prompt = f"""
        Bạn là trợ lý AI chuyên phân tích mã nhị phân iOS ARM64 và đảo ngược ứng dụng.
        Ngữ cảnh hệ thống hiện tại: {req.context_summary}
        Hàm đang phân tích: {req.function_name} (VMAddr: {req.vmaddr}, FileOffset: {req.file_offset})
        Yêu cầu của người dùng: {req.query}
        
        Hãy đưa ra phân tích logic, hướng dẫn NOP, Bypass hoặc mã Theos Hook / Frida script chuẩn xác nhất bằng tiếng Việt.
        """
        
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        ai_answer = response.choices[0].message.content
        return {"status": "success", "response": ai_answer}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
