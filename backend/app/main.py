from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict_score(request: PredictionRequest):
    # Hardcoded response as specified
    return {"score": 10}

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 