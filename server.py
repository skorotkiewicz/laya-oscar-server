"""HTTP API for the Oscar checkpoint used in example.py."""

import logging
import os
from contextlib import asynccontextmanager
from threading import Lock
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, StrictStr, model_validator

CHECKPOINT = "mgoeckel/oscar-1-400m"
Text = Annotated[StrictStr, Field(min_length=1)]
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
    model: Literal["oscar-1", "mgoeckel/oscar-1-400m"] = "oscar-1"
    state: Text | dict[str, object]
    questions: Annotated[dict[Text, Question], Field(min_length=1, max_length=64)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    import laya

    app.state.agent = laya.load(CHECKPOINT, device=os.getenv("LAYA_DEVICE") or None)
    # ponytail: one inference at a time; batch requests if throughput matters.
    app.state.inference_lock = Lock()
    try:
        yield
    finally:
        del app.state.agent


app = FastAPI(title="Laya Oscar API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": "oscar-1"}


@app.post("/v1/decisions")
def decisions(body: DecisionRequest):
    questions = {key: question.model_dump(exclude_none=True) for key, question in body.questions.items()}
    try:
        with app.state.inference_lock:
            result = app.state.agent.predict(body.state, questions)
        return {**result, "model": "oscar-1"}
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logging.exception("Oscar inference failed")
        raise HTTPException(status_code=500, detail="inference failed") from error


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)
