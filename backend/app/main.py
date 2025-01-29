from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
