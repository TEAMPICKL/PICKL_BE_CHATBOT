import os, re, json
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager

import pymysql 
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# ==== DB ENV ====
DB_HOST = os.getenv("DB_HOST", os.getenv("MYSQL_HOST", "mysql")) 
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

# ============================================================
# ============== 💾 가격/요약 조회: 일·월·연 툴 =================
# ============================================================

# 공용: INFORMATION_SCHEMA로 컬럼명 자동 인식
def find_column(conn, table: str, candidates: List[str]) -> Optional[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COLUMN_NAME
              FROM INFORMATION_SCHEMA.COLUMNS
             WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s
            """,
            (DB_NAME, table),
        )
        cols = {row["COLUMN_NAME"].lower(): row["COLUMN_NAME"] for row in cur.fetchall()}
    for c in candidates:
        lc = c.lower()
        if lc in cols:
            return cols[lc]
    return None

def latest_value(conn, table: str, col_cands: List[str]) -> Optional[str]:
    col = find_column(conn, table, col_cands)
    if not col:
        return None
    with conn.cursor() as cur:
        cur.execute(f"SELECT MAX({col}) AS v FROM {table}")
        row = cur.fetchone()
        return row["v"] if row else None

# 입력 스키마
class DailyItemIn(BaseModel):
    item_name: str = Field(..., description="상품명(부분일치)")
    market: Optional[str] = Field(None, description="소매|도매 (옵션)")

class DailyCategoryIn(BaseModel):
    category: str = Field(..., description="카테고리명 또는 코드(부분일치 가능)")
    market: Optional[str] = Field(None, description="소매|도매 (옵션)")

class MonthlyItemIn(BaseModel):
    item_name: str = Field(..., description="상품명(부분일치)")
    year: int = Field(..., ge=2000, le=2100)
    month: int = Field(..., ge=1, le=12)
    market: Optional[str] = None

class YearlyItemIn(BaseModel):
    item_name: str = Field(..., description="상품명(부분일치)")
    year: int = Field(..., ge=2000, le=2100)
    market: Optional[str] = None

class MonthlyCategoryIn(BaseModel):
    category: str
    year: int = Field(..., ge=2000, le=2100)
    month: int = Field(..., ge=1, le=12)
    market: Optional[str] = None

class YearlyCategoryIn(BaseModel):
    category: str
    year: int = Field(..., ge=2000, le=2100)
    market: Optional[str] = None

# ---- (1) 오늘/어제: kamis_item_price ----
@tool("get_daily_item_price", args_schema=DailyItemIn)
def get_daily_item_price(item_name: str, market: Optional[str] = None) -> Dict[str, Any]:
    """
    최신 수집일의 품목 가격(오늘/어제 포함)을 조회. 테이블: kamis_item_price
    반환: {"table":"kamis_item_price","date":"YYYY-MM-DD","q":"배추","market":"소매","rows":[...], "count":N}
    """
    table = "kamis_item_price"
    with get_conn() as conn, conn.cursor() as cur:
        date_col   = find_column(conn, table, ["price_date", "pricedate", "date"])
        name_col   = find_column(conn, table, ["product_name", "item_name", "itemname", "productname"])
        market_col = find_column(conn, table, ["product_cls_name", "productclass", "market", "product_clsname"])
        if not (date_col and name_col):
            return {"error": "schema_mismatch", "table": table}

        latest = latest_value(conn, table, [date_col])
        if not latest:
            return {"error": "no_data", "table": table}

        where = [f"{date_col}=%s", f"{name_col} LIKE %s"]
        params = [latest, safe_like(item_name)]
        if market and market_col:
            where.append(f"{market_col}=%s")
            params.append(market)

        sql = f"SELECT * FROM {table} WHERE {' AND '.join(where)} ORDER BY id LIMIT 20"
        cur.execute(sql, params)
        rows = cur.fetchall()
        return {"table": table, "date": str(latest), "q": item_name, "market": market, "rows": rows, "count": len(rows)}

# ---- (2) 오늘/어제 카테고리 평균: kamis_category_summary ----
@tool("get_daily_category_avg", args_schema=DailyCategoryIn)
def get_daily_category_avg(category: str, market: Optional[str] = None) -> Dict[str, Any]:
    """
    최신 수집일의 카테고리 평균가(오늘/어제)를 조회. 테이블: kamis_category_summary
    """
    table = "kamis_category_summary"
    with get_conn() as conn, conn.cursor() as cur:
        date_col   = find_column(conn, table, ["price_date", "pricedate", "date"])
        code_col   = find_column(conn, table, ["category_code", "categorycode"])
        name_col   = find_column(conn, table, ["category_name", "categoryname", "name"])
        market_col = find_column(conn, table, ["product_cls_name", "market"])
        if not (date_col and (code_col or name_col)):
            return {"error": "schema_mismatch", "table": table}

        latest = latest_value(conn, table, [date_col])
        if not latest:
            return {"error": "no_data", "table": table}

        conds, params = [f"{date_col}=%s"], [latest]
        like = safe_like(category)
        if code_col and name_col:
            conds.append(f"({code_col} LIKE %s OR {name_col} LIKE %s)")
            params.extend([like, like])
        elif code_col:
            conds.append(f"{code_col} LIKE %s"); params.append(like)
        else:
            conds.append(f"{name_col} LIKE %s"); params.append(like)

        if market and market_col:
            conds.append(f"{market_col}=%s"); params.append(market)

        sql = f"SELECT * FROM {table} WHERE {' AND '.join(conds)} ORDER BY id LIMIT 20"
        cur.execute(sql, params)
        rows = cur.fetchall()
        return {"table": table, "date": str(latest), "q": category, "market": market, "rows": rows, "count": len(rows)}

# ---- (3) 월간 품목: monthly_item_price (24-09 ~ 25-08) ----
@tool("get_monthly_item_price", args_schema=MonthlyItemIn)
def get_monthly_item_price(item_name: str, year: int, month: int, market: Optional[str] = None) -> Dict[str, Any]:
    table = "monthly_item_price"
    ym6 = f"{year:04d}{month:02d}"   # 'YYYYMM'
    ym7 = f"{year:04d}-{month:02d}"  # 'YYYY-MM'
    with get_conn() as conn, conn.cursor() as cur:
        m_col     = find_column(conn, table, ["yyyymm", "price_month", "month", "stat_month", "pricemonth"])
        name_col  = find_column(conn, table, ["product_name", "item_name", "itemname", "productname"])
        market_col= find_column(conn, table, ["product_cls_name", "market"])
        if not (m_col and name_col):
            return {"error": "schema_mismatch", "table": table}

        conds = [f"({m_col} = %s OR LEFT({m_col}, 7) = %s)", f"{name_col} LIKE %s"]
        params = [ym6, ym7, safe_like(item_name)]
        if market and market_col:
            conds.append(f"{market_col}=%s")
            params.append(market)

        sql = f"SELECT * FROM {table} WHERE {' AND '.join(conds)} ORDER BY id LIMIT 50"
        cur.execute(sql, params)
        rows = cur.fetchall()
        return {"table": table, "year": year, "month": month, "q": item_name, "market": market, "rows": rows, "count": len(rows)}

# ---- (4) 월간 카테고리: monthly_category_summary ----
@tool("get_monthly_category_avg", args_schema=MonthlyCategoryIn)
def get_monthly_category_avg(category: str, year: int, month: int, market: Optional[str] = None) -> Dict[str, Any]:
    table = "monthly_category_summary"
    ym6 = f"{year:04d}{month:02d}"   # 'YYYYMM'
    ym7 = f"{year:04d}-{month:02d}"  # 'YYYY-MM'
    with get_conn() as conn, conn.cursor() as cur:
        m_col     = find_column(conn, table, ["yyyymm", "price_month", "month", "stat_month"])
        code_col  = find_column(conn, table, ["category_code", "categorycode"])
        name_col  = find_column(conn, table, ["category_name", "categoryname", "name"])
        market_col= find_column(conn, table, ["product_cls_name", "market"])
        if not m_col or not (code_col or name_col):
            return {"error": "schema_mismatch", "table": table}

        like = safe_like(category)
        conds, params = [f"({m_col} = %s OR LEFT({m_col}, 7) = %s)"], [ym6, ym7]
        if code_col and name_col:
            conds.append(f"({code_col} LIKE %s OR {name_col} LIKE %s)")
            params.extend([like, like])
        elif code_col:
            conds.append(f"{code_col} LIKE %s"); params.append(like)
        else:
            conds.append(f"{name_col} LIKE %s"); params.append(like)

        if market and market_col:
            conds.append(f"{market_col}=%s"); params.append(market)

        sql = f"SELECT * FROM {table} WHERE {' AND '.join(conds)} ORDER BY id LIMIT 50"
        cur.execute(sql, params)
        rows = cur.fetchall()
        return {"table": table, "year": year, "month": month, "q": category, "market": market, "rows": rows, "count": len(rows)}

# ---- (5) 연간 품목: yearly_item_price (5년치) ----
@tool("get_yearly_item_price", args_schema=YearlyItemIn)
def get_yearly_item_price(item_name: str, year: int, market: Optional[str] = None) -> Dict[str, Any]:
    table = "yearly_item_price"
    with get_conn() as conn, conn.cursor() as cur:
        y_col     = find_column(conn, table, ["yyyy", "price_year", "year", "stat_year"])
        name_col  = find_column(conn, table, ["product_name", "item_name", "itemname", "productname"])
        market_col= find_column(conn, table, ["product_cls_name", "market"])
        if not (y_col and name_col):
            return {"error": "schema_mismatch", "table": table}

        conds = [f"{y_col} = %s", f"{name_col} LIKE %s"]
        params = [str(year), safe_like(item_name)]
        if market and market_col:
            conds.append(f"{market_col}=%s")
            params.append(market)

        sql = f"SELECT * FROM {table} WHERE {' AND '.join(conds)} ORDER BY id LIMIT 50"
        cur.execute(sql, params)
        rows = cur.fetchall()
        return {"table": table, "year": year, "q": item_name, "market": market, "rows": rows, "count": len(rows)}

# ---- (6) 연간 카테고리: yearly_category_summary ----
@tool("get_yearly_category_avg", args_schema=YearlyCategoryIn)
def get_yearly_category_avg(category: str, year: int, market: Optional[str] = None) -> Dict[str, Any]:
    table = "yearly_category_summary"
    with get_conn() as conn, conn.cursor() as cur:
        y_col     = find_column(conn, table, ["yyyy", "price_year", "year", "stat_year"])
        code_col  = find_column(conn, table, ["category_code", "categorycode"])
        name_col  = find_column(conn, table, ["category_name", "categoryname", "name"])
        market_col= find_column(conn, table, ["product_cls_name", "market"])
        if not y_col or not (code_col or name_col):
            return {"error": "schema_mismatch", "table": table}

        like = safe_like(category)
        conds, params = [f"{y_col} = %s"], [str(year)]
        if code_col and name_col:
            conds.append(f"({code_col} LIKE %s OR {name_col} LIKE %s)")
            params.extend([like, like])
        elif code_col:
            conds.append(f"{code_col} LIKE %s"); params.append(like)
        else:
            conds.append(f"{name_col} LIKE %s"); params.append(like)

        if market and market_col:
            conds.append(f"{market_col}=%s"); params.append(market)

        sql = f"SELECT * FROM {table} WHERE {' AND '.join(conds)} ORDER BY id LIMIT 50"
        cur.execute(sql, params)
        rows = cur.fetchall()
        return {"table": table, "year": year, "q": category, "market": market, "rows": rows, "count": len(rows)}

# ---- LangChain 툴 목록 등록 ----
DB_TOOLS = [
    get_season_items_by_month,
    get_recipes_by_season_item,
    get_daily_item_price,
    get_daily_category_avg,
    get_monthly_item_price,
    get_yearly_item_price,
    get_monthly_category_avg,
    get_yearly_category_avg,
]

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