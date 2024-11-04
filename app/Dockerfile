FROM python:3.11-slim

RUN mkdir /app
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py"]