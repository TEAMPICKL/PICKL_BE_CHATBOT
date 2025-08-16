import os
from typing import List, Literal, Optional, AsyncIterator, Tuple
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.prompts import load_prompt
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from app.db import db_health

load_dotenv()

app = FastAPI(title="LLM Chat Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")

# ===== prompt.yaml: 답변까지 만들어내는 단일 체인 =====
# (이 체인 자체가 최종 답변을 생성한다)
prompt = load_prompt("prompts/prompt.yaml", encoding="utf-8")
llm = ChatOpenAI(model=MODEL, temperature=0.2)

# ===== 요청 스키마 =====
class ChatReq(BaseModel):
    message: str
    memory: Optional[str] = ""

class Turn(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class HistoryReq(BaseModel):
    messages: List[Turn]
    memory: Optional[str] = ""

# ===== 공통 컨텍스트 빌더 =====
def build_context(memory: str = "", messages: Optional[List[Turn]] = None, max_turns: int = 20) -> str:
    parts = []
    if memory:
        parts.append("### Memory\n" + memory.strip())
    if messages:
        recent = messages[-max_turns:]
        lines = []
        for m in recent:
            role = m.role.lower()
            if role not in ("user", "assistant", "system"):
                role = "user"
            lines.append(f"- {role}: {m.content}")
        parts.append("### Recent Chat\n" + "\n".join(lines))
    return "\n\n".join(parts).strip() or "N/A"

def pick_last_user_question(msgs: List[Turn]) -> str:
    for m in reversed(msgs):
        if m.role == "user" and m.content.strip():
            return m.content.strip()
    return msgs[-1].content.strip() if msgs else ""

# ===== prompt.yaml로 '즉시 완성' =====
async def run_prompt_only(context: str, question: str) -> str:
    # prompt.yaml은 일반 프롬프트이므로 메시지 하나(Human)로 전달
    rendered = prompt.format(question=question, context=context)
    resp = await llm.ainvoke([HumanMessage(content=rendered)])
    return (resp.content or "").strip()

# ===== prompt.yaml로 '스트리밍' =====
async def stream_prompt_only(context: str, question: str) -> AsyncIterator[str]:
    rendered = prompt.format(question=question, context=context)
    async for chunk in llm.astream([HumanMessage(content=rendered)]):
        token = getattr(chunk, "content", None) or (chunk if isinstance(chunk, str) else None)
        if token:
            yield token

# =========================
# 1) 단발 채팅
# =========================
@app.post("/chat")
async def chat(req: ChatReq):
    context = build_context(memory=req.memory)
    out = await run_prompt_only(context, req.message)
    return {"reply": out}

# =========================
# 2) 단발 스트리밍
# =========================
@app.post("/chat/stream")
async def chat_stream(req: ChatReq):
    context = build_context(memory=req.memory)

    async def gen():
        async for token in stream_prompt_only(context, req.message):
            yield f"data: {token}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")

# =========================
# 3) 히스토리 기반 채팅
# =========================
@app.post("/chat/history")
async def chat_history(req: HistoryReq):
    context = build_context(memory=req.memory, messages=req.messages)
    question = pick_last_user_question(req.messages)
    out = await run_prompt_only(context, question)
    return {"reply": out}

# =========================
# 4) 히스토리 기반 스트리밍
# =========================
@app.post("/chat/history/stream")
async def chat_history_stream(req: HistoryReq):
    context = build_context(memory=req.memory, messages=req.messages)
    question = pick_last_user_question(req.messages)

    async def gen():
        async for token in stream_prompt_only(context, question):
            yield f"data: {token}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")

# =========================
# 5) 제목 (그대로 유지)
# =========================
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

title_prompt = ChatPromptTemplate.from_template(
    """You are a concise Korean title generator for chat threads.

[Rules]
- Output: ONLY the title text in Korean. No quotes, no punctuation at the end.
- Length: up to {max_len} characters (hard cap).
- Style: short, noun-phrase style (no verbs if possible), include the key intent/keyword when clear.
- If unclear, use a minimal neutral title like "대화".
- Do not mention dates unless the user asked about a specific date.
- Absolutely no surrounding quotes.

[Inputs]
- question: the user's first message (Korean)
- context: optional memory/recent chat (use only if clearly helpful)

[Examples]
- "대한민국 수도는?" -> "대한민국 수도"
- "요즘 감기 기운 있는데 뭐 먹을까?" -> "감기 때 먹을 음식 추천"
- "오늘 감자 시세 어때?" -> "감자 시세 검색"
- "다이어트 식단 추천해줘" -> "다이어트 식단 추천"
- "근처 재래시장 어디 있어?" -> "근처 재래시장 추천"

[Data]
question: {question}
context: {context}
max_len: {max_len}"""
)
title_chain = title_prompt | ChatOpenAI(model=MODEL, temperature=0.2) | StrOutputParser()

class TitleReq(BaseModel):
    message: Optional[str] = None
    messages: Optional[List[Turn]] = None
    memory: Optional[str] = ""
    max_len: int = 20

class TitleRes(BaseModel):
    title: str

@app.post("/title", response_model=TitleRes)
async def make_title(req: TitleReq):
    if req.message:
        first = req.message.strip()
    elif req.messages:
        first = next((m.content.strip() for m in req.messages if m.role == "user" and m.content.strip()), "")
    else:
        first = ""
    if not first:
        return TitleRes(title="대화")
    context = build_context(memory=req.memory)
    try:
        out = await title_chain.ainvoke({"question": first, "context": context, "max_len": req.max_len})
        title = (out or "").strip().replace('"', '').replace("'", "")
        if not title:
            title = "대화"
        if len(title) > req.max_len:
            title = title[:req.max_len]
        title = title.rstrip(".!?~ ")
        return TitleRes(title=title or "대화")
    except Exception:
        s = first
        cut = min(len(s), req.max_len)
        return TitleRes(title=(s[:cut].rstrip(".!?~ ") or "대화"))

# =========================
# 6) 헬스체크
# =========================
@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL}

@app.get("/health/db")
async def health_db():
    return db_health()