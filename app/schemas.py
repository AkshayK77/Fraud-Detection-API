from pydantic import BaseModel, Field
from typing import Literal


class TransactionInput(BaseModel):
    amount: float = Field(example=149.62, description="Transaction amount in USD")
    hour_of_day: float = Field(example=14.5, ge=0, lt=24, description="Hour of day (0-23)")
    day_of_week: int = Field(example=2, ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    v1: float = Field(example=-1.3598071336738, description="PCA feature V1")
    v2: float = Field(example=-0.0727811733098497, description="PCA feature V2")
    v3: float = Field(example=2.53634673796914, description="PCA feature V3")
    v4: float = Field(example=1.37815522427443, description="PCA feature V4")
    v5: float = Field(example=-0.338320769942518, description="PCA feature V5")
    v6: float = Field(example=0.462387777762292, description="PCA feature V6")
    v7: float = Field(example=0.239598554061257, description="PCA feature V7")
    v8: float = Field(example=0.0986979012610507, description="PCA feature V8")
    v9: float = Field(example=0.363786969611213, description="PCA feature V9")
    v10: float = Field(example=0.0907941719789316, description="PCA feature V10")
    v11: float = Field(example=-0.551599533260813, description="PCA feature V11")
    v12: float = Field(example=-0.617800855762348, description="PCA feature V12")
    v13: float = Field(example=-0.991389847235408, description="PCA feature V13")
    v14: float = Field(example=-0.311169353699879, description="PCA feature V14")
    v15: float = Field(example=1.46817697209427, description="PCA feature V15")
    v16: float = Field(example=-0.470400525259478, description="PCA feature V16")
    v17: float = Field(example=0.207971241929242, description="PCA feature V17")
    v18: float = Field(example=0.0257905801985591, description="PCA feature V18")
    v19: float = Field(example=0.403992960255733, description="PCA feature V19")
    v20: float = Field(example=0.251412098239705, description="PCA feature V20")
    v21: float = Field(example=-0.018306777944153, description="PCA feature V21")
    v22: float = Field(example=0.277837575558899, description="PCA feature V22")
    v23: float = Field(example=-0.110473910188767, description="PCA feature V23")
    v24: float = Field(example=0.0669280749146731, description="PCA feature V24")
    v25: float = Field(example=0.128539358273528, description="PCA feature V25")
    v26: float = Field(example=-0.189114843888824, description="PCA feature V26")
    v27: float = Field(example=0.133558376740387, description="PCA feature V27")
    v28: float = Field(example=-0.0210530534538215, description="PCA feature V28")

    model_config = {
        "json_schema_extra": {
            "example": {
                "amount": 149.62,
                "hour_of_day": 14.5,
                "day_of_week": 2,
                "v1": -1.3598071336738,
                "v2": -0.0727811733098497,
                "v3": 2.53634673796914,
                "v4": 1.37815522427443,
                "v5": -0.338320769942518,
                "v6": 0.462387777762292,
                "v7": 0.239598554061257,
                "v8": 0.0986979012610507,
                "v9": 0.363786969611213,
                "v10": 0.0907941719789316,
                "v11": -0.551599533260813,
                "v12": -0.617800855762348,
                "v13": -0.991389847235408,
                "v14": -0.311169353699879,
                "v15": 1.46817697209427,
                "v16": -0.470400525259478,
                "v17": 0.207971241929242,
                "v18": 0.0257905801985591,
                "v19": 0.403992960255733,
                "v20": 0.251412098239705,
                "v21": -0.018306777944153,
                "v22": 0.277837575558899,
                "v23": -0.110473910188767,
                "v24": 0.0669280749146731,
                "v25": 0.128539358273528,
                "v26": -0.189114843888824,
                "v27": 0.133558376740387,
                "v28": -0.0210530534538215,
            }
        }
    }


class PredictionResponse(BaseModel):
    is_fraud: bool = Field(description="Whether the transaction is predicted as fraudulent")
    confidence: float = Field(description="Model confidence score (0.0 to 1.0)")
    risk_level: Literal["LOW", "MEDIUM", "HIGH"] = Field(description="Risk level based on confidence")
    inference_time_ms: float = Field(description="Model inference time in milliseconds")
