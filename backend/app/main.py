from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
from datetime import datetime
from typing import List
import json
import os

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Create logs directory if it doesn't exist
if not os.path.exists("logs"):
    os.makedirs("logs")

LOG_FILE = "logs/prediction_logs.json"

version = "0.0.1"

class PredictionRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict_score(request: PredictionRequest):
    # Generate random score between 0 and 6000
    random_score = random.randint(0, 6000)
    return {"score": random_score}

@app.get("/ping")
async def health_check():
    return "ok"

@app.get("/version")
async def get_version():
    return {"version": version}

@app.get("/logs")
async def get_logs() -> dict[str, List[dict]]:
    if not os.path.exists(LOG_FILE):
        return {"logs": []}
        
    try:
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
        return {"logs": logs}
    except json.JSONDecodeError:
        return {"logs": []}

def save_log(latency: float, text: str, score: int):
    log_entry = {
        "latency": latency,
        "version": version, 
        "timestamp": datetime.now().isoformat(),
        "input": text,
        "prediction": score
    }
    
    # Load existing logs
    logs = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            logs = []
    
    # Append new log
    logs.append(log_entry)
    
    # Save updated logs
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f)

class PredicationRequest2(BaseModel):
    author: str
    title: str
    timestamp: str

@app.post("/how_many_upvotes")
async def predict_score(post: PredicationRequest2):
    # Generate random score between 0 and 6000
    number = random.randint(0, 6000)
    return {"upvotes": number}
