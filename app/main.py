import os
from typing import List, Literal, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import load_prompt
from langchain_openai import ChatOpenAI

load_dotenv()

app = FastAPI(title="LLM Chat Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")

# ── YAML 프롬프트 로드 → LLM → 파서 ──
prompt = load_prompt("prompts/prompt.yaml", encoding="utf-8")
llm = ChatOpenAI(model=MODEL, temperature=0.2)
chain = prompt | llm | StrOutputParser()

# ===== 요청 스키마 =====
class ChatReq(BaseModel):
    message: str
    memory: Optional[str] = ""   # 스프링이 DB에서 읽어서 전달

class Turn(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class HistoryReq(BaseModel):
    messages: List[Turn]
    memory: Optional[str] = ""

# ===== 컨텍스트 빌더 =====
def build_context(memory: str = "", messages: Optional[List[Turn]] = None, max_turns: int = 20) -> str:
    parts = []
    if memory:
        parts.append("### Memory\n" + memory.strip())

    if messages:
        # 최근 max_turns만 사용 (토큰 폭주 방지)
        recent = messages[-max_turns:]
        lines = []
        for m in recent:
            role = m.role.lower()
            if role not in ("user", "assistant", "system"):
                role = "user"
            lines.append(f"- {role}: {m.content}")
        parts.append("### Recent Chat\n" + "\n".join(lines))

    return "\n\n".join(parts).strip() or "N/A"

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL}

# ===== 단발 =====
@app.post("/chat")
async def chat(req: ChatReq):
    context = build_context(memory=req.memory)
    out = await chain.ainvoke({"context": context, "question": req.message})
    return {"reply": out}

# ===== 히스토리 =====
@app.post("/chat/history")
async def chat_history(req: HistoryReq):
    context = build_context(memory=req.memory, messages=req.messages)
    # 마지막 유저 메시지를 question으로 쓰는 게 일반적
    last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    out = await chain.ainvoke({"context": context, "question": last_user})
    return {"reply": out}

# ===== 스트리밍(SSE) =====
@app.post("/chat/stream")
async def chat_stream(req: ChatReq):
    context = build_context(memory=req.memory)
    async def gen():
        async for token in chain.astream({"context": context, "question": req.message}):
            if token:
                yield f"data: {token}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")