# app/main.py
import os
import json
import re
import asyncio, time
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
import datetime as dt
import decimal
from zoneinfo import ZoneInfo

# DB 툴/헬스
from app.db import (
    db_health,
    get_season_items_by_month,
    get_recipes_by_season_item,
    get_daily_item_price,
    get_daily_category_avg,
    get_monthly_item_price,
    get_yearly_item_price,
    get_monthly_category_avg,
    get_yearly_category_avg,
)

load_dotenv()


def json_default(o):
    if isinstance(o, (dt.date, dt.datetime)):
        return o.isoformat()
    if isinstance(o, decimal.Decimal):
        return float(o)
    return str(o)

# ===== SSE 유틸 =====
def sse_data(payload: str) -> str:
    s = str(payload).replace("\r\n", "\n").replace("\r", "\n")
    return "data: " + "\ndata: ".join(s.split("\n")) + "\n\n"

async def heartbeat_wrap(token_aiter: AsyncIterator[str], ping_interval: int = 15):
    yield "retry: 2000\n\n"     # reconnection hint
    yield ": stream-open\n\n"   # comment frame (not delivered to client handler)

    aiter = token_aiter.__aiter__()
    while True:
        try:
            tok = await asyncio.wait_for(aiter.__anext__(), timeout=ping_interval)
            if tok:
                yield sse_data(tok)
        except asyncio.TimeoutError:
            yield f": ping {int(time.time())}\n\n"
        except StopAsyncIteration:
            break

    yield "event: done\ndata:\n\n"

# ===== FastAPI/CORS =====
app = FastAPI(title="LLM Chat Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ===== LLM / Prompt =====
MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
prompt = load_prompt("prompts/prompt.yaml", encoding="utf-8")
llm = ChatOpenAI(model=MODEL, temperature=0.2)

# ===== pydantic 요청 스키마 =====
class ChatReq(BaseModel):
    message: str
    memory: Optional[str] = ""

class Turn(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class HistoryReq(BaseModel):
    messages: List[Turn]
    memory: Optional[str] = ""

# ===== 컨텍스트/질문 헬퍼 =====
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

# ===== 날짜/시장/카테고리 감지 =====
KST = ZoneInfo("Asia/Seoul")
TODAY = dt.datetime.now(KST).date()

def _detect_market(q: str) -> str | None:
    if "소매" in q: return "소매"
    if "도매" in q: return "도매"
    return None

CATEGORY_SYNONYMS = {
    "식량작물": "식량작물", "곡물": "식량작물",
    "채소류": "채소류", "채소": "채소류",
    "특용작물": "특용작물", "특작": "특용작물",
    "과일류": "과일류", "과일": "과일류",
    "축산물": "축산물", "축산": "축산물", 
    "수산물": "수산물", "수산": "수산물",
    "과채": "채소류",
}

UNSUPPORTED_BY_MARKET = {
    ("도매", "축산물"),
}

def detect_category(q: str) -> Optional[str]:
    if not q: return None
    for k, v in CATEGORY_SYNONYMS.items():
        if k in q:
            return v
    return None

def _parse_year_month_from_text(q: str) -> tuple[Optional[int], Optional[int], Optional[str]]:
    """
    텍스트에서 연/월을 해석.
    반환: (year or None, month or None, mode)  # mode ∈ {"monthly","yearly",None}
    """
    q = (q or "").strip()

    # "YYYY년 MM월"
    m = re.search(r'(\d{4})\s*년\s*(\d{1,2})\s*월', q)
    if m:
        return int(m.group(1)), int(m.group(2)), "monthly"

    # "작년 MM월"
    m = re.search(r'작년\s*(\d{1,2})\s*월', q)
    if m:
        return TODAY.year - 1, int(m.group(1)), "monthly"

    # "지난달"/"이번달"
    if "지난달" in q:
        first = TODAY.replace(day=1)
        last_month = first - __import__("datetime").timedelta(days=1)
        return last_month.year, last_month.month, "monthly"
    if "이번달" in q or "이달" in q:
        return TODAY.year, TODAY.month, "monthly"

    # "YYYY년" (월 없음) → yearly
    m = re.search(r'(\d{4})\s*년(?!\s*\d+\s*월)', q)
    if m:
        return int(m.group(1)), None, "yearly"

    return None, None, None

# ===== DB 라우팅 =====
def fetch_db_results(question: str) -> str:
    """
    질문을 보고 필요한 경우 MySQL에서 값을 조회해 JSON 문자열로 반환.
    반환이 빈 문자열이면 CONTEXT에 주입하지 않음 → 프롬프트가 알아서 "소매/도매?" 같은 후속 질문을 하게 됨.
    """
    q = (question or "").strip()
    market = _detect_market(q)
    category = detect_category(q)

    # (A) n월 제철
    m = re.search(r'([1-9]|1[0-2])\s*월.*(제철|식재료)', q)
    if m:
        month = int(m.group(1))
        res = get_season_items_by_month.invoke({"month": month})
        payload = {"kind": "season_items_by_month", "month": month, "result": res}
        return json.dumps(payload, ensure_ascii=False, default=json_default)

    # (B) 레시피
    m = re.search(r'([가-힣A-Za-z0-9]+)\s*(레시피|요리|만드는\s*법)', q)
    if m:
        item = m.group(1)
        res = get_recipes_by_season_item.invoke({"season_item": item})
        payload = {"kind": "recipes_by_item", "item": item, "result": res}
        return json.dumps(payload, ensure_ascii=False, default=json_default)

    # (C) 오늘/어제/최근 - 품목 시세
    if re.search(r'(오늘|어제|최근|시세|가격)', q) and re.search(r'(시세|가격)', q):
        m = re.search(r'([가-힣A-Za-z0-9]{2,})\s*(시세|가격)', q)
        if m:
            item = m.group(1)
            res = get_daily_item_price.invoke({"item_name": item, "market": market})
            payload = {"kind": "daily_item_price", "item": item, "market": market, "result": res}
            return json.dumps(payload, ensure_ascii=False, default=json_default)

    # (D) 오늘/어제 - 카테고리 평균 (시장 미지정이면 DB조회 보류 → 프롬프트가 "소매/도매?" 질문)
    if category and re.search(r'(오늘|어제|최근|시세|가격|평균)', q):
        # 도매×축산물 → 소매로 대체하며 알림 표시
        note = None
        used_market = market
        if market and (market, category) in UNSUPPORTED_BY_MARKET:
            used_market = "소매"
            note = "도매에는 ‘축산물’ 카테고리 데이터가 없어 소매 기준으로 안내합니다."

        if not used_market:
            # 시장을 지정하지 않았으면 DB_RESULT를 넣지 않음 → 모델이 먼저 물어본다.
            return ""

        res = get_daily_category_avg.invoke({"category": category, "market": used_market})
        payload = {
            "kind": "daily_category_avg",
            "category": category,
            "normalizedCategory": category,
            "marketRequested": market,
            "marketUsed": used_market,
            "result": res
        }
        if note:
            payload["notice"] = note
        return json.dumps(payload, ensure_ascii=False, default=json_default)

    # (E/F) 월간/연간 - 품목
    y, mth, mode = _parse_year_month_from_text(q)
    if mode == "monthly" and y and mth:
        m = re.search(r'([가-힣A-Za-z0-9]{2,})\s*(시세|가격|평균|동향|추이)', q)
        if m:
            item = m.group(1)
            res = get_monthly_item_price.invoke({"item_name": item, "year": y, "month": mth, "market": market})
            payload = {"kind": "monthly_item_price", "item": item, "year": y, "month": mth, "market": market, "result": res}
            return json.dumps(payload, ensure_ascii=False, default=json_default)

    if mode == "yearly" and y:
        m = re.search(r'([가-힣A-Za-z0-9]{2,})\s*(시세|가격|평균|동향|추이)', q)
        if m:
            item = m.group(1)
            res = get_yearly_item_price.invoke({"item_name": item, "year": y, "market": market})
            payload = {"kind": "yearly_item_price", "item": item, "year": y, "market": market, "result": res}
            return json.dumps(payload, ensure_ascii=False, default=json_default)

    # (G) 월간/연간 - 카테고리 (시장 미지정이면 질문 유도)
    if mode == "monthly" and y and mth and category:
        if not market:
            return ""
        res = get_monthly_category_avg.invoke({"category": category, "year": y, "month": mth, "market": market})
        payload = {"kind": "monthly_category_avg", "category": category, "year": y, "month": mth, "market": market, "result": res}
        return json.dumps(payload, ensure_ascii=False, default=json_default)

    if mode == "yearly" and y and category:
        if not market:
            return ""
        res = get_yearly_category_avg.invoke({"category": category, "year": y, "market": market})
        payload = {"kind": "yearly_category_avg", "category": category, "year": y, "market": market, "result": res}
        return json.dumps(payload, ensure_ascii=False, default=json_default)

    return ""

# ===== LLM 실행 =====
async def run_prompt_only(context: str, question: str) -> str:
    db_blob = fetch_db_results(question)
    if db_blob:
        context = (context + "\n\n### DB_RESULTS\n" + db_blob).strip()
    rendered = prompt.format(question=question, context=context)
    resp = await llm.ainvoke([HumanMessage(content=rendered)])
    return (resp.content or "").strip()

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
        token_stream = stream_prompt_only(context, req.message)
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
# 5) 제목
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