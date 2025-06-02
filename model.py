# main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel

app = FastAPI()

# Allow all origins (for local development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData():
    price: float
    category: str
    product_name: str
    date: str

@app.post("/predict")
async def predict(data: InputData):
    # Echo back the input for debugging
    return {
        "message": "Data received successfully.",
        "price": data.price,
        "category": data.category,
        "product_name": data.product_name,
        "date": data.date,
        "probability": 0.42  # temporary dummy output
    }
