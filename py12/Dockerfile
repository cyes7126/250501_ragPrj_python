FROM python:3.12-bookworm

RUN apt update && apt install -y vim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "5001"]
