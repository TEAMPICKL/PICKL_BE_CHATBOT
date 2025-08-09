FROM python:3.11-slim
WORKDIR /app

# 의존성
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 코드
COPY . .

EXPOSE 8000

# 헬스체크용 엔드포인트가 main.py에 /health 있다고 가정
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# main.py가 루트에 있고 FastAPI 앱이 app 변수면 그대로, 
# 만약 app/main.py 구조면 "app.main:app"로 바꾸세요.
CMD ["uvicorn", "main:app", "--host","0.0.0.0","--port","8000"]