"""HTTP API for Oscar and Laya checkpoints."""

import hmac
import json
import logging
import os
from contextlib import asynccontextmanager
from threading import Lock
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, StrictStr, model_validator

CHECKPOINTS = {
    "oscar-1": "mgoeckel/oscar-1-400m",
    "laya": "convaiinnovations/laya",
}
Text = Annotated[StrictStr, Field(min_length=1, max_length=4096)]
Criteria = (
    Annotated[dict[Text, Text], Field(min_length=2, max_length=255)]
    | Annotated[list[Text], Field(min_length=2, max_length=255)]
)


class Question(BaseModel):
    type: Literal["noul", "choice", "score"]
    instructions: Text
    criteria: Criteria | None = None

    @model_validator(mode="after")
    def check_criteria(self):
        if self.type == "choice" and self.criteria is None:
            raise ValueError("choice requires criteria")
        if self.type == "score" and not isinstance(self.criteria, list):
            raise ValueError("score requires a list of criteria")
        if self.type == "choice" and isinstance(self.criteria, list):
            if len(set(self.criteria)) != len(self.criteria):
                raise ValueError("choice criteria must be unique")
        if self.type == "noul" and self.criteria is not None:
            if not isinstance(self.criteria, dict) or set(self.criteria) != {"true", "false"}:
                raise ValueError("noul criteria must use 'true' and 'false' keys")
        return self


class DecisionRequest(BaseModel):
    model: Literal["oscar-1", "mgoeckel/oscar-1-400m", "laya", "convaiinnovations/laya"] = "oscar-1"
    state: Annotated[StrictStr, Field(min_length=1, max_length=50000)] | dict[str, object]
    questions: Annotated[dict[Text, Question], Field(min_length=1, max_length=64)]

    @model_validator(mode="after")
    def check_state_size(self):
        if isinstance(self.state, dict) and len(json.dumps(self.state, ensure_ascii=False)) > 50000:
            raise ValueError("state must not exceed 50000 characters")
        return self


@asynccontextmanager
async def lifespan(app: FastAPI):
    import laya

    selected = os.getenv("MODEL") or None
    if selected is not None and selected not in CHECKPOINTS:
        raise ValueError("MODEL must be 'laya' or 'oscar-1'")
    app.state.api_key = os.getenv("API_KEY")
    app.state.agents = {
        name: laya.load(checkpoint, device=os.getenv("LAYA_DEVICE") or None)
        for name, checkpoint in CHECKPOINTS.items()
        if selected is None or name == selected
    }
    # ponytail: one inference at a time; batch requests if throughput matters.
    app.state.inference_lock = Lock()
    try:
        yield
    finally:
        del app.state.agents


def require_api_key(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(HTTPBearer(auto_error=False))],
):
    key = app.state.api_key
    if key and not hmac.compare_digest(
        (credentials.credentials if credentials else "").encode("utf-8"), key.encode("utf-8")
    ):
        raise HTTPException(
            status_code=401, detail="invalid or missing access token",
            headers={"WWW-Authenticate": "Bearer"},
        )


app = FastAPI(title="Laya and Oscar API", lifespan=lifespan, dependencies=[Depends(require_api_key)])


@app.get("/health")
def health():
    return {"status": "ok", "models": list(app.state.agents)}


@app.post("/v1/decisions")
def decisions(body: DecisionRequest):
    questions = {key: question.model_dump(exclude_none=True) for key, question in body.questions.items()}
    model = "laya" if body.model in ("laya", CHECKPOINTS["laya"]) else "oscar-1"
    if model not in app.state.agents:
        raise HTTPException(status_code=422, detail=f"model '{model}' is not enabled on this server")
    try:
        with app.state.inference_lock:
            result = app.state.agents[model].predict(body.state, questions)
        return {**result, "model": model}
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logging.exception("%s inference failed", model)
        raise HTTPException(status_code=500, detail="inference failed") from error


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8001")))
