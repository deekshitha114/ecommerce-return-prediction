
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional


APP_VERSION = "1.0.0"

app = FastAPI(
    title="E-Commerce Product Return Prediction",
    description="Predict product return status using Machine Learning",
    version=APP_VERSION
)

# Load trained model
try:
    model = joblib.load("model/return_model.pkl")
except Exception:
    model = None


# Input schema
class ProductData(BaseModel):

    Product_Category: str = "Electronics"

    Product_Price_INR: Optional[float] = Field(
        default=10000, gt=0
    )

    Discount_Percent: Optional[float] = Field(
        default=10, ge=0, le=100
    )

    Expected_Delivery_Days: Optional[float] = Field(
        default=5, gt=0
    )

    Actual_Delivery_Days: Optional[float] = Field(
        default=6, gt=0
    )

    Customer_Rating_PreOrder: Optional[float] = Field(
        default=4, ge=1, le=5
    )

    Previous_Returns: Optional[float] = Field(
        default=1, ge=0
    )

    Previous_Orders: Optional[float] = Field(
        default=10, gt=0
    )

    Payment_Method: str = "UPI"

    Damaged_Package_Flag: str = "No"

    Package_Size: str = "Medium"

    Promotion_Active: str = "Yes"


# Health endpoint
@app.get("/health")
def health():

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    return {
        "status": "healthy",
        "version": APP_VERSION,
        "model": "return_model.pkl"
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: ProductData):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available"
        )

    try:

        product = data.model_dump()

        # Feature engineering
        product["Delivery_Delay"] = (
            product["Actual_Delivery_Days"]
            - product["Expected_Delivery_Days"]
        )

        product["Customer_Return_Frequency"] = (
            product["Previous_Returns"]
            / product["Previous_Orders"]
        )

        product["Discount_Amount"] = (
            product["Product_Price_INR"]
            * product["Discount_Percent"] / 100
        )

        input_data = pd.DataFrame([product])

        prediction = int(
            model.predict(input_data)[0]
        )

        probability = float(
            model.predict_proba(input_data)[0][1]
        )

        return {
            "prediction": (
                "Returned"
                if prediction == 1
                else "Not Returned"
            ),
            "return_probability": round(
                probability, 4
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
