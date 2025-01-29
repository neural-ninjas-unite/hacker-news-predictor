FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Define build argument with default value
ARG BACKEND_PORT=8000
# Make it available as env variable
ENV BACKEND_PORT=$BACKEND_PORT

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "${BACKEND_PORT}", "--reload"] 