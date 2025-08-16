# app/db.py
import os, re, json
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager

import pymysql  # requirements.txt에 PyMySQL 추가 필수!
from pydantic import BaseModel, Field
from langchain_core.tools import tool

# ==== DB ENV ====
DB_HOST = os.getenv("DB_HOST", os.getenv("MYSQL_HOST", "mysql"))  # ← 기본값 mysql 권장
DB_PORT = int(os.getenv("DB_PORT", os.getenv("MYSQL_PORT", "3306")))
DB_USER = os.getenv("DB_USER", os.getenv("MYSQL_USER", "root"))
DB_PASSWORD = os.getenv("DB_PASSWORD", os.getenv("MYSQL_PASSWORD", ""))
DB_NAME = os.getenv("DB_NAME", os.getenv("MYSQL_DATABASE", "pickl"))

@contextmanager
def get_conn():
    conn = pymysql.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD,
        database=DB_NAME, charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor, autocommit=True,
    )
    try:
        yield conn
    finally:
        conn.close()

# ==== 공통 ====
def safe_like(q: str) -> str:
    return f"%{q.strip()}%"

def season_item_name_col() -> str:
    # 네 스키마에선 'itemname'
    return "itemname"

def month_condition_sql(month: int) -> Tuple[str, List[Any]]:
    # in_season_month = int
    return "in_season_month = %s", [month]

def recipe_select_sql() -> Tuple[str, List[str]]:
    cols = [
        "id",
        "recipe_name AS recipeName",
        "ingredients",
        "instructions",
        "tip",
        "cooking_time_text AS cookingTime",
        "recommend_tags_csv",
    ]
    return ", ".join(cols), cols

# ==== 툴 스키마 ====
class MonthIn(BaseModel):
    month: int = Field(..., ge=1, le=12, description="1~12")

class RecipeIn(BaseModel):
    season_item: str = Field(..., description="식재료 ID('12') 또는 이름('옥수수')")

# ==== 헬퍼: 이름/ID 해석 ====
def resolve_season_item_id(conn, season_item: str) -> Optional[int]:
    if re.fullmatch(r"\d+", season_item.strip()):
        return int(season_item)
    name_col = season_item_name_col()
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT id FROM season_item WHERE {name_col} LIKE %s ORDER BY id LIMIT 1",
            (safe_like(season_item),),
        )
        row = cur.fetchone()
        return int(row["id"]) if row else None

# ==== 툴 1: 월별 제철 식재료 ====
@tool("get_season_items_by_month", args_schema=MonthIn)
def get_season_items_by_month(month: int) -> Dict[str, Any]:
    """
    해당 월(1~12)의 제철 식재료 목록을 반환합니다.
    반환: {"items":[{"id":1,"name":"옥수수"}], "count":N, "month":8}
    """
    with get_conn() as conn, conn.cursor() as cur:
        name_col = season_item_name_col()
        cond, params = month_condition_sql(month)
        sql = f"SELECT id, {name_col} AS name FROM season_item WHERE {cond} ORDER BY {name_col} ASC"
        cur.execute(sql, params)
        rows = cur.fetchall()
    return {"items": rows, "count": len(rows), "month": month}

# ==== 툴 2: 식재료 레시피 ====
@tool("get_recipes_by_season_item", args_schema=RecipeIn)
def get_recipes_by_season_item(season_item: str) -> Dict[str, Any]:
    """
    식재료 ID('12') 또는 이름('옥수수')를 받아 레시피 목록을 반환합니다.
    반환: {"season_item_id":12, "recipes":[{...}], "count":N}
    """
    with get_conn() as conn:
        sid = resolve_season_item_id(conn, season_item)
        if not sid:
            return {"error": "season_item_not_found", "q": season_item}

        sel_sql, _ = recipe_select_sql()
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {sel_sql} FROM season_item_recipe WHERE season_item_id=%s ORDER BY id ASC",
                (sid,),
            )
            rows = cur.fetchall()

    # recommend_tags_csv -> 배열로
    for r in rows:
        csv = r.pop("recommend_tags_csv", None)
        if csv:
            r["recommendTags"] = [t.strip() for t in csv.split(",") if t.strip()]

    return {"season_item_id": sid, "recipes": rows, "count": len(rows)}

# ==== LangChain 바인딩 목록 ====
DB_TOOLS = [get_season_items_by_month, get_recipes_by_season_item]

# ==== 헬스 ====
def db_health() -> Dict[str, Any]:
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT VERSION() AS v")
            ver = cur.fetchone()["v"]
            cur.execute("SELECT COUNT(*) AS c FROM season_item")
            c1 = cur.fetchone()["c"]
            cur.execute("SELECT COUNT(*) AS c FROM season_item_recipe")
            c2 = cur.fetchone()["c"]
        return {"ok": True, "version": ver, "season_item": c1, "season_item_recipe": c2}
    except Exception as e:
        return {"ok": False, "error": str(e)}