import os
import json
import re
from typing import List, Literal, Optional, AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.prompts import load_prompt, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import asyncio, time


# SSE data 인코딩 유틸: 멀티라인도 규격에 맞게 변환
def sse_data(payload: str) -> str:
    s = str(payload).replace("\r\n", "\n").replace("\r", "\n")
    return "data: " + "\ndata: ".join(s.split("\n")) + "\n\n"

# 하트비트 래퍼: 다음 토큰을 ping_interval 내에 못 받으면 댓글 프레임 전송
async def heartbeat_wrap(token_aiter: AsyncIterator[str], ping_interval: int = 15):
    # 스트림을 즉시 시작시키고 클라이언트 재시도 힌트 제공(선택)
    yield "retry: 2000\n\n"     # 브라우저 직접 테스트 시 유용, Spring WebClient에는 영향 없음
    yield ": stream-open\n\n"   # 댓글 프레임: 프록시 버퍼 비우고 연결 활성화

    aiter = token_aiter.__aiter__()
    while True:
        try:
            tok = await asyncio.wait_for(aiter.__anext__(), timeout=ping_interval)
            if tok:  # 빈 토큰 방지
                yield sse_data(tok)
        except asyncio.TimeoutError:
            # 댓글 기반 핑 → 디코더에 데이터로 전달되지 않음(프록시 keep-alive 용도)
            yield f": ping {int(time.time())}\n\n"
        except StopAsyncIteration:
            break

    # 선택: 완료 이벤트(원하면 사용)
    yield "event: done\ndata:\n\n"

# DB 유틸/헬스
from app.db import (
    db_health,
    get_season_items_by_month,
    get_recipes_by_season_item,
)

load_dotenv()

app = FastAPI(title="LLM Chat Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")

# ===== prompt.yaml: 답변까지 만들어내는 단일 체인 =====
# (이 체인 자체가 최종 답변을 생성)
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

# ===== 질문에 따라 MySQL 조회 → CONTEXT에 주입 =====
def fetch_db_results(question: str) -> str:
    """
    질문을 보고 필요한 경우 MySQL에서 값을 조회해 JSON 문자열로 반환.
    반환이 빈 문자열이면 CONTEXT에 주입하지 않음.
    """
    q = (question or "").strip()

    # (A) "n월 제철" 같은 패턴 → 월별 제철 식재료
    m = re.search(r'([1-9]|1[0-2])\s*월.*(제철|식재료)', q)
    if m:
        month = int(m.group(1))
        res = get_season_items_by_month.invoke({"month": month})
        return json.dumps(
            {"kind": "season_items_by_month", "month": month, "result": res},
            ensure_ascii=False
        )

    # (B) "옥수수 레시피", "OOO 요리", "OO 만드는 법" 등 → 식재료 레시피
    m = re.search(r'([가-힣A-Za-z0-9]+)\s*(레시피|요리|만드는\s*법)', q)
    if m:
        item = m.group(1)
        res = get_recipes_by_season_item.invoke({"season_item": item})
        return json.dumps(
            {"kind": "recipes_by_item", "item": item, "result": res},
            ensure_ascii=False
        )

    return ""

# ===== prompt.yaml로 '즉시 완성' =====
async def run_prompt_only(context: str, question: str) -> str:
    db_blob = fetch_db_results(question)
    if db_blob:
        context = (context + "\n\n### DB_RESULTS\n" + db_blob).strip()
    rendered = prompt.format(question=question, context=context)
    resp = await llm.ainvoke([HumanMessage(content=rendered)])
    return (resp.content or "").strip()

# ===== prompt.yaml로 '스트리밍' =====
async def stream_prompt_only(context: str, question: str) -> AsyncIterator[str]:
    db_blob = fetch_db_results(question)
    if db_blob:
        context = (context + "\n\n### DB_RESULTS\n" + db_blob).strip()
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
# 2) 단발 스트리밍 (SSE)
# =========================
@app.post("/chat/stream")
async def chat_stream(req: ChatReq):
    context = build_context(memory=req.memory)

    async def gen():
        # 기존 스트림 생성
        token_stream = stream_prompt_only(context, req.message)
        # 하트비트로 감싸기
        async for frame in heartbeat_wrap(token_stream, ping_interval=15):
            yield frame

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Nginx 사용 시 버퍼링 끄기(브라우저/프록시 즉시 전송)
            "X-Accel-Buffering": "no",
        },
    )

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
# 4) 히스토리 기반 스트리밍 (SSE)
# =========================
@app.post("/chat/history/stream")
async def chat_history_stream(req: HistoryReq):
    context = build_context(memory=req.memory, messages=req.messages)
    question = pick_last_user_question(req.messages)

    async def gen():
        token_stream = stream_prompt_only(context, question)
        async for frame in heartbeat_wrap(token_stream, ping_interval=15):
            yield frame

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

# =========================
# 5) 제목 (그대로 유지)
# =========================
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