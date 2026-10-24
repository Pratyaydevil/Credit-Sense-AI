import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

# Import model inference and LangGraph workflow from credit.py
import credit

app = FastAPI(
    title="CreditSense AI - Credit Risk Profiling & Rehabilitation Engine",
    description="Full-stack AI Underwriting and Multi-Agent Credit Risk Decision Support API",
    version="2.0.0"
)

# Enable CORS for local and web testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ApplicantInput(BaseModel):
    Monthly_Inhand_Salary: float = Field(..., ge=0, description="Monthly in-hand net salary in USD")
    Outstanding_Debt: float = Field(..., ge=0, description="Total outstanding debt in USD")
    Credit_Utilization_Ratio: float = Field(..., ge=0, le=100, description="Credit card utilization percentage (0-100)")
    Delay_from_due_date: float = Field(..., ge=0, description="Average payment latency past due date in days")
    Num_Credit_Inquiries: float = Field(..., ge=0, description="Number of hard credit inquiries in past 6 months")
    Annual_Income: Optional[float] = Field(None, ge=0, description="Annual gross income in USD")
    Total_EMI_per_month: Optional[float] = Field(None, ge=0, description="Total monthly installment burden in USD")
    Num_of_Loan: Optional[int] = Field(2, ge=0, description="Number of active loans")
    Credit_History_Age: Optional[str] = Field("5 Years and 6 Months", description="Credit tenure string")
    mode: Optional[str] = Field("FIVE_QUESTION", description="Inference mode: 'FIVE_QUESTION' or 'FULL'")
    api_key: Optional[str] = Field(None, description="Optional Google Gemini API key for live Gemini 2.5 Flash synthesis")

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "engine": "LangGraph Multi-Agent Workflow",
        "ml_model": "XGBoost Dual-Model Pipeline",
        "genai_available": credit.GENAI_AVAILABLE
    }

@app.get("/api/metrics")
def get_model_metrics():
    """Returns dual-model comparison, candidate model benchmarks, hazard analysis, and thresholds."""
    try:
        metrics = credit.get_system_metrics()
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/samples")
def get_sample_applicants():
    """Provides curated applicant profiles for 1-click interactive demo testing."""
    samples = {
        "preconfigured": {
            "name": "Pre-Configured Test Profile (Benchmark)",
            "description": "Standard risk profile with moderate latency and utilization.",
            "data": {
                "Monthly_Inhand_Salary": 3800.0,
                "Outstanding_Debt": 1650.0,
                "Credit_Utilization_Ratio": 54.0,
                "Delay_from_due_date": 12.0,
                "Num_Credit_Inquiries": 4.0,
                "Annual_Income": 48000.0,
                "Total_EMI_per_month": 420.0,
                "Num_of_Loan": 3,
                "Credit_History_Age": "6 Years and 2 Months"
            }
        },
        "prime": {
            "name": "Tier-1 Prime Borrower (Good / Low Risk)",
            "description": "High liquidity, pristine repayment discipline, minimal utilization.",
            "data": {
                "Monthly_Inhand_Salary": 7500.0,
                "Outstanding_Debt": 420.0,
                "Credit_Utilization_Ratio": 14.5,
                "Delay_from_due_date": 0.0,
                "Num_Credit_Inquiries": 1.0,
                "Annual_Income": 90000.0,
                "Total_EMI_per_month": 85.0,
                "Num_of_Loan": 1,
                "Credit_History_Age": "9 Years and 8 Months"
            }
        },
        "moderate": {
            "name": "Tier-2 Borderline Borrower (Standard / Moderate Risk)",
            "description": "Fair debt handling but borderline credit inquiries and payment delays.",
            "data": {
                "Monthly_Inhand_Salary": 4200.0,
                "Outstanding_Debt": 1950.0,
                "Credit_Utilization_Ratio": 42.0,
                "Delay_from_due_date": 7.0,
                "Num_Credit_Inquiries": 3.0,
                "Annual_Income": 52000.0,
                "Total_EMI_per_month": 310.0,
                "Num_of_Loan": 2,
                "Credit_History_Age": "4 Years and 5 Months"
            }
        },
        "distressed": {
            "name": "Subprime Delinquent Borrower (Poor / High Risk)",
            "description": "Severe revolving utilization, heavy delinquency latency, aggressive inquiry velocity.",
            "data": {
                "Monthly_Inhand_Salary": 2100.0,
                "Outstanding_Debt": 4800.0,
                "Credit_Utilization_Ratio": 84.0,
                "Delay_from_due_date": 25.0,
                "Num_Credit_Inquiries": 7.0,
                "Annual_Income": 25200.0,
                "Total_EMI_per_month": 620.0,
                "Num_of_Loan": 5,
                "Credit_History_Age": "2 Years and 1 Month"
            }
        }
    }
    return samples

@app.post("/api/assess")
def assess_applicant_endpoint(payload: ApplicantInput):
    """
    Executes the full LangGraph credit underwriting multi-agent workflow:
    1. Validation node
    2. ML Prediction node (XGBoost)
    3. Health Analytics & Recommendations node
    4. Gemini 2.5 Flash Synthesis & Rehabilitation node
    """
    try:
        # Prepare applicant dictionary matching feature expectations
        data = payload.dict()
        if not data.get("Annual_Income"):
            data["Annual_Income"] = data["Monthly_Inhand_Salary"] * 12.0
        if not data.get("Total_EMI_per_month"):
            data["Total_EMI_per_month"] = data["Outstanding_Debt"] * 0.15

        mode = data.get("mode", "FIVE_QUESTION")
        api_key = data.get("api_key")

        # Execute LangGraph pipeline
        result = credit.assess_applicant(data, mode=mode, api_key=api_key)

        return {
            "success": True,
            "prediction": result.get("prediction", {}),
            "health": result.get("health", {}),
            "health_index": result.get("health_index", 0.0),
            "recommendations": result.get("recommendations", []),
            "final_narrative": result.get("final_narrative", ""),
            "mode": mode,
            "raw_input": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assessment error: {str(e)}")

@app.get("/api/plots/{plot_name}")
def get_plot(plot_name: str):
    file_map = {
        "eda": "models/eda_distribution.png",
        "confusion_matrix": "models/confusion_matrix.png"
    }
    path = file_map.get(plot_name)
    if path and os.path.exists(path):
        return FileResponse(path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Plot not found")

# Ensure static directory exists
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("Starting CreditSense AI Full-Stack Server on http://localhost:8000")
    print("="*70)
    uvicorn.run(app, host="127.0.0.1", port=8000)
