FROM python:3.11-slim

RUN mkdir /training
WORKDIR /training

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN prefect config set PREFECT_API_URL=http://host.docker.internal:4200/api
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV PREFECT_API_URL=http://host.docker.internal:4200/api
CMD ["sh","-c","prefect server start --host 0.0.0.0 & prefect deployment apply train-deployment.yaml & sleep 25 && prefect deployment run train/iris && prefect agent start -q default"]