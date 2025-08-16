# app/main.py
import os, json
from typing import List, Literal, Optional, AsyncIterator, Tuple
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import load_prompt
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app.db import DB_TOOLS, db_health

load_dotenv()

app = FastAPI(title="LLM Chat Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")

# (A) 레거시 완성응답 체인(필요 시 사용)
prompt = load_prompt("prompts/prompt.yaml", encoding="utf-8")
base_llm = ChatOpenAI(model=MODEL, temperature=0.2)
chain = prompt | base_llm | StrOutputParser()

# (B) 툴 시스템 프롬프트 (영문)
TOOLS_SYSTEM = """
You are an assistant that must respond to the user **in Korean only**.

You have the following tools. If your answer depends on database values, you **must** call the appropriate tool first to verify facts.

- get_season_items_by_month(month:int): Returns the list of seasonal ingredients for a given month (1–12).
- get_recipes_by_season_item(season_item:str): Takes a season item ID ('12') or a Korean name ('옥수수') and returns its recipes.

Rules:
1) If the question is about seasonal items or recipes, infer the tool arguments and call the tool. If a required argument is missing, ask exactly **one** short follow-up question in Korean to fill it.
2) Base the final answer on the tool results and output only a concise final answer in Korean (no JSON/raw data dumps).
3) If you don’t know, say so in Korean and suggest one next step.
"""

# (C) 툴 바인딩 LLM
llm_with_tools = ChatOpenAI(model=MODEL, temperature=0.2).bind_tools(DB_TOOLS)

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

# === 제목 생성 ===
class TitleReq(BaseModel):
    message: Optional[str] = None
    messages: Optional[List[Turn]] = None
    memory: Optional[str] = ""
    max_len: int = 20

def pick_last_user_question(msgs: List[Turn]) -> str:
    for m in reversed(msgs):
        if m.role == "user" and m.content.strip():
            return m.content.strip()
    return msgs[-1].content.strip() if msgs else ""

@app.post("/chat/history")
async def chat_history(req: HistoryReq):
    # 메모리 + 최근 턴들을 컨텍스트에 포함
    context = build_context(memory=req.memory, messages=req.messages)
    question = pick_last_user_question(req.messages)
    out = await run_with_tools(context, question)
    return {"reply": out}

@app.post("/chat/history/stream")
async def chat_history_stream(req: HistoryReq):
    context = build_context(memory=req.memory, messages=req.messages)
    question = pick_last_user_question(req.messages)

    async def gen():
        async for token in run_with_tools_stream(context, question):
            yield f"data: {token}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")

class TitleRes(BaseModel):
    title: str

# ===== 공통 컨텍스트 =====
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

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL}

# ===== 툴콜 안전 언패커(버전 의존성 방지) =====
def unpack_tool_call(call) -> Tuple[str, dict, Optional[str]]:
    """
    LangChain 버전에 따라 tool_calls 원소가 dict 또는 ToolCall 객체일 수 있다.
    항상 (name:str, args:dict, id:str|None) 튜플로 반환.
    """
    if isinstance(call, dict):
        return call.get("name"), call.get("args") or {}, call.get("id")
    # ToolCall-like 객체 지원
    name = getattr(call, "name", None)
    args = getattr(call, "args", None) or {}
    call_id = getattr(call, "id", None)
    return name, args, call_id

# =========================
# 1) 완성 응답 (툴 사용)
# =========================
async def run_with_tools(context: str, question: str) -> str:
    messages = [
        SystemMessage(content=TOOLS_SYSTEM),
        SystemMessage(content=f"=== CONTEXT ===\n{context or 'N/A'}"),
        HumanMessage(content=question),
    ]
    resp = await llm_with_tools.ainvoke(messages)

    while getattr(resp, "tool_calls", None):
        for call in resp.tool_calls:
            name, args, call_id = unpack_tool_call(call)
            tool_fn = next((t for t in DB_TOOLS if t.name == name), None)
            result = tool_fn.invoke(args) if tool_fn else {"error": "unknown tool", "name": name}
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)
            messages.append(ToolMessage(content=result, tool_call_id=call_id))
        resp = await llm_with_tools.ainvoke(messages)

    return (resp.content or "").strip()

@app.post("/chat")
async def chat(req: ChatReq):
    context = build_context(memory=req.memory)
    out = await run_with_tools(context, req.message)
    return {"reply": out}

# =========================
# 2) 스트리밍 (툴 결과 반영 후 최종턴만 스트림)
# =========================
FINALIZE_SYSTEM = """
역할: 한국어 어시스턴트.
아래 DB 조회 결과(툴 결과)를 근거로 간결하고 정확한 답변만 출력하세요.
- 근거를 나열하지 말고, 최종 결론/추천만 한국어 문장으로 정리.
- 불확실하면 모른다고 말하고, 다음 액션을 1가지 제안.
"""

async def stream_final_answer(context: str, question: str, tool_results_jsonl: List[str]) -> AsyncIterator[str]:
    finalize_llm = ChatOpenAI(model=MODEL, temperature=0.2)
    tool_blob = "\n".join(tool_results_jsonl) if tool_results_jsonl else "{}"

    messages = [
        SystemMessage(content=FINALIZE_SYSTEM),
        SystemMessage(content=f"=== CONTEXT ===\n{context or 'N/A'}"),
        SystemMessage(content=f"=== TOOL_RESULTS(JSON-LINES) ===\n{tool_blob}"),
        HumanMessage(content=question),
    ]

    async for chunk in finalize_llm.astream(messages):
        # ChatOpenAI는 BaseMessageChunk를 스트림으로 준다 → content 사용
        token = getattr(chunk, "content", None) or (chunk if isinstance(chunk, str) else None)
        if token:
            yield token

async def run_with_tools_stream(context: str, question: str) -> AsyncIterator[str]:
    messages = [
        SystemMessage(content=TOOLS_SYSTEM),
        SystemMessage(content=f"=== CONTEXT ===\n{context or 'N/A'}"),
        HumanMessage(content=question),
    ]
    resp = await llm_with_tools.ainvoke(messages)
    tool_jsonl: List[str] = []

    while getattr(resp, "tool_calls", None):
        for call in resp.tool_calls:
            name, args, call_id = unpack_tool_call(call)
            tool_fn = next((t for t in DB_TOOLS if t.name == name), None)
            result = tool_fn.invoke(args) if tool_fn else {"error": "unknown tool", "name": name}
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)
            tool_jsonl.append(result)
            messages.append(ToolMessage(content=result, tool_call_id=call_id))
        resp = await llm_with_tools.ainvoke(messages)

    # 툴 결과가 모두 반영된 뒤 최종 답변을 토큰 스트림으로 생성
    async for tok in stream_final_answer(context, question, tool_jsonl):
        yield tok

@app.post("/chat/stream")
async def chat_stream(req: ChatReq):
    context = build_context(memory=req.memory)

    async def gen():
        async for token in run_with_tools_stream(context, req.message):
            yield f"data: {token}\n\n"

    # SSE
    return StreamingResponse(gen(), media_type="text/event-stream")

# =========================
# 3) 제목 생성 (요청대로 그대로 유지)
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
title_chain = title_prompt | base_llm | StrOutputParser()

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
# 4) DB 헬스체크
# =========================
@app.get("/health/db")
async def health_db():
    return db_health()