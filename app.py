import logging
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from classifier import is_relevant

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

app = FastAPI(title="relevant-priors-v1")

# Simple in-process cache keyed on (current_study_id, prior_study_id).
# Avoids re-running the same pair on retries or duplicated requests.
_cache: dict[tuple[str, str], bool] = {}


# ── Pydantic models ──────────────────────────────────────────────────────────

class Study(BaseModel):
    study_id: str
    study_description: str
    study_date: str


class Case(BaseModel):
    case_id: str
    patient_id: str | None = None
    patient_name: str | None = None
    current_study: Study
    prior_studies: list[Study]


class PredictionRequest(BaseModel):
    challenge_id: str | None = None
    schema_version: int | None = None
    generated_at: str | None = None
    cases: list[Case]


class Prediction(BaseModel):
    case_id: str
    study_id: str
    predicted_is_relevant: bool


class PredictionResponse(BaseModel):
    predictions: list[Prediction]


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "service": "relevant-priors-v1"}


@app.post("/predict", response_model=PredictionResponse)
def predict(body: PredictionRequest, request: Request):
    req_id = str(uuid.uuid4())[:8]
    total_priors = sum(len(c.prior_studies) for c in body.cases)
    log.info(
        "req=%s  cases=%d  total_priors=%d",
        req_id, len(body.cases), total_priors,
    )

    t0 = time.perf_counter()
    predictions: list[Prediction] = []

    for case in body.cases:
        cur = case.current_study
        for prior in case.prior_studies:
            cache_key = (cur.study_id, prior.study_id)
            if cache_key in _cache:
                result = _cache[cache_key]
            else:
                result = is_relevant(cur.study_description, prior.study_description)
                _cache[cache_key] = result

            predictions.append(
                Prediction(
                    case_id=case.case_id,
                    study_id=prior.study_id,
                    predicted_is_relevant=result,
                )
            )

    elapsed = time.perf_counter() - t0
    log.info(
        "req=%s  predictions=%d  elapsed=%.3fs",
        req_id, len(predictions), elapsed,
    )

    return PredictionResponse(predictions=predictions)


# Evaluator might POST to the root or to /predict – handle both just in case.
@app.post("/")
async def predict_root(request: Request) -> Any:
    body_json = await request.json()
    req = PredictionRequest(**body_json)
    return predict(req, request)
