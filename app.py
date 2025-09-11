from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import random

# Define the request body schema to match your HTML form
class BookingData(BaseModel):
    lead_time: float
    total_nights: float
    adults: float
    children: float
    babies: float
    booking_changes: float
    adr: float
    total_of_special_requests: float
    is_canceled: int

# Initialize the FastAPI application
app = FastAPI(
    title="Hotel Customer Segment Predictor",
    description="An API to predict hotel customer segments based on booking data.",
    version="1.0.0",
)

# Add CORS middleware to allow requests from the HTML file
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Define the customer segments with their details and emojis
CUSTOMER_SEGMENTS = {
    "Leisure Travelers": {
        "emoji": "✈️",
        "summary": "This segment is characterized by bookings with a short lead time and a high total number of special requests, indicating spontaneous travel plans. They often book for shorter stays and may be families or couples.",
        "thresholds": lambda data: data.lead_time < 30 and data.total_of_special_requests > 1
    },
    "Corporate & Business": {
        "emoji": "💼",
        "summary": "This segment typically has a high booking changes count and a short stay duration, often booking for a single person. They value flexibility and are not always the ones making the booking.",
        "thresholds": lambda data: data.booking_changes >= 1 and data.total_nights < 5 and data.adults == 1
    },
    "High-Value Guests": {
        "emoji": "💎",
        "summary": "High-value guests have a high ADR (Average Daily Rate) and typically book with a high lead time, which suggests they are planning a significant event or vacation. They also tend to have a low cancellation rate.",
        "thresholds": lambda data: data.adr > 150 and data.lead_time > 60 and data.is_canceled == 0
    },
    "Family Stay": {
        "emoji": "👨‍👩‍👧‍👦",
        "summary": "This segment includes guests traveling with children or babies. They often book for longer stays and tend to have a moderate lead time to plan their vacation.",
        "thresholds": lambda data: (data.children > 0 or data.babies > 0) and data.total_nights >= 5
    },
    "Other": {
        "emoji": "👤",
        "summary": "This is the default segment for bookings that do not fit into any of the predefined categories. It represents a diverse group of customers with varied booking patterns.",
        "thresholds": lambda data: True
    }
}

def predict_segment(data: BookingData) -> dict:
    """
    Predicts the customer segment based on simple, predefined rules.
    In a real-world application, this is where your trained ML model would be called.
    """
    # Check the data against the predefined rules
    for segment_name, segment_details in CUSTOMER_SEGMENTS.items():
        if segment_details["thresholds"](data):
            return {
                "name": segment_name,
                "emoji": segment_details["emoji"],
                "summary": segment_details["summary"],
            }
    
    # Fallback to a default segment if no rules are met
    return {
        "name": "Other",
        "emoji": CUSTOMER_SEGMENTS["Other"]["emoji"],
        "summary": CUSTOMER_SEGMENTS["Other"]["summary"],
    }

@app.get("/")
def home():
    """Simple homepage for the API."""
    return {"message": "The Hotel Customer Segment API is running! Use the /predict endpoint to make a prediction."}

@app.post("/predict")
async def predict(request_data: BookingData):
    """
    Accepts customer booking details and returns a predicted segment.
    """
    try:
        prediction_result = predict_segment(request_data)
        return prediction_result
    
    except Exception as e:
        # Return a custom error message
        return {"error": f"An error occurred during prediction: {str(e)}"}
