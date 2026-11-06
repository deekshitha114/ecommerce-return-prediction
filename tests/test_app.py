
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["status"] == "healthy"


def test_prediction():

    data = {
        "Product_Category": "Electronics",
        "Product_Price_INR": 15000,
        "Discount_Percent": 20,
        "Expected_Delivery_Days": 5,
        "Actual_Delivery_Days": 8,
        "Customer_Rating_PreOrder": 3.5,
        "Previous_Returns": 2,
        "Previous_Orders": 10,
        "Payment_Method": "UPI",
        "Damaged_Package_Flag": "No",
        "Package_Size": "Medium",
        "Promotion_Active": "Yes"
    }

    response = client.post(
        "/predict",
        json=data
    )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in [
        "Returned",
        "Not Returned"
    ]

    assert 0 <= result["return_probability"] <= 1


def test_invalid_price():

    data = {
        "Product_Price_INR": -100
    }

    response = client.post(
        "/predict",
        json=data
    )

    assert response.status_code == 422
