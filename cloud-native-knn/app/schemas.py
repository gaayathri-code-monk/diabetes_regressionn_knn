from pydantic import BaseModel, Field
from typing import List

class IrisRequest(BaseModel):
    features: List[float] = Field(..., description="Sepal length, sepal width, petal length, petal width")

class IrisResponse(BaseModel):
    predicted_class: int
    class_name: str
    probabilities: List[float]

class DiabetesRequest(BaseModel):
    features: List[float] = Field(..., description="10 numeric features as in sklearn diabetes dataset")

class DiabetesResponse(BaseModel):
    prediction: float
