from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import List
import json
import os
from dotenv import load_dotenv
import time
from .utils import hello_world, predict_score
# from predict import predict_score

load_dotenv()

app = FastAPI()

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create logs directory if it doesn't exist
if not os.path.exists("logs"):
    os.makedirs("logs")

LOG_FILE = "logs/prediction_logs.json"

version = os.getenv("APP_VERSION", "1.0.0")

class PredictionRequest(BaseModel):
    author: str
    title: str
    timestamp: str

@app.post("/how_many_upvotes")
async def get_how_many_upvotes(post: PredictionRequest):
    # Record start time for latency calculation
    start_time = time.time()
    
    # Test utils import
    print(hello_world("HackerNews"))
    
    # Get prediction from model
    score = predict_score(post.title)
    # score = 7
    
    # Calculate latency
    latency = time.time() - start_time
    
    # Save prediction log
    save_log(
        latency=latency,
        text=post.title,
        score=score
    )
    return {"upvotes": score}

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
