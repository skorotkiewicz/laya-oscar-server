"""HTTP API for Oscar and Laya checkpoints."""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import signal
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, StrictStr, model_validator
from redis import Redis, RedisError
from redis.backoff import NoBackoff
from redis.retry import Retry

CHECKPOINTS = {
    # "oscar-1-17m": "mgoeckel/oscar-1-17m",
    # "oscar-1-32m": "mgoeckel/oscar-1-32m",
    # "oscar-1-68m": "mgoeckel/oscar-1-68m",
    # "oscar-1-150m": "mgoeckel/oscar-1-150m",
    "oscar-1": "mgoeckel/oscar-1-400m",
    "laya": "convaiinnovations/laya",
    # "laya-multilingual": "convaiinnovations/laya-multilingual",
    # "laya-typed-decisions": "convaiinnovations/laya-typed-decisions",
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
    model: Literal[
        "oscar-1", "mgoeckel/oscar-1-400m",
        "laya", "convaiinnovations/laya",
        # "oscar-1-small", "mgoeckel/oscar-1-17m",
        # "laya-multilingual", "convaiinnovations/laya-multilingual",
        # "laya-typed-decisions", "convaiinnovations/laya-typed-decisions"
    ]
    state: Annotated[StrictStr, Field(min_length=1, max_length=50000)] | dict[str, object]
    questions: Annotated[dict[Text, Question], Field(min_length=1, max_length=64)]

    @model_validator(mode="after")
    def check_state_size(self):
        if isinstance(self.state, dict) and len(json.dumps(self.state, ensure_ascii=False)) > 50000:
            raise ValueError("state must not exceed 50000 characters")
        return self


def redis_unavailable():
    if not app.state.redis_failed:
        app.state.redis_failed = True
        logging.error("Redis unavailable; stopping server")
        os.kill(os.getpid(), signal.SIGTERM)
    return HTTPException(status_code=503, detail="Redis unavailable; server is shutting down")


async def watch_redis():
    while not app.state.redis_failed:
        # ponytail: 5s polling; shorten the interval if faster idle-outage detection matters.
        await asyncio.sleep(5)
        try:
            await asyncio.to_thread(app.state.cache.ping)
        except RedisError:
            redis_unavailable()
            return


@asynccontextmanager
async def lifespan(app: FastAPI):
    selected = os.getenv("MODEL") or None
    if selected is not None and selected not in CHECKPOINTS:
        raise ValueError(f"MODEL must be one of {', '.join(CHECKPOINTS)}")
    app.state.api_key = os.getenv("API_KEY")
    redis_url = os.getenv("REDIS_URL")
    app.state.cache = Redis.from_url(
        redis_url, decode_responses=True, socket_connect_timeout=0.5,
        socket_timeout=0.5, retry=Retry(NoBackoff(), 0),
    ) if redis_url else None
    app.state.redis_failed = False
    watcher = None
    try:
        if app.state.cache is not None:
            try:
                app.state.cache.ping()
            except RedisError as error:
                raise RuntimeError("Redis unavailable; server startup aborted") from error
        import laya

        app.state.agents = {
            name: laya.load(checkpoint, device=os.getenv("LAYA_DEVICE") or None)
            for name, checkpoint in CHECKPOINTS.items()
            if selected is None or name == selected
        }
        # ponytail: global cache/inference lock; move hit lookups outside if cache latency matters.
        app.state.inference_lock = Lock()
        if app.state.cache is not None:
            watcher = asyncio.create_task(watch_redis())
        yield
    finally:
        if watcher is not None:
            watcher.cancel()
            with suppress(asyncio.CancelledError):
                await watcher
        if app.state.cache is not None:
            app.state.cache.close()
        if hasattr(app.state, "agents"):
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


app = FastAPI(title="Laya and Oscar API", lifespan=lifespan)


@app.get("/floppybird.html", include_in_schema=False)
def floppybird():
    return FileResponse(Path(__file__).with_name("floppybird.html"))


@app.get("/health", dependencies=[Depends(require_api_key)])
def health():
    if app.state.redis_failed:
        raise redis_unavailable()
    if app.state.cache is not None:
        try:
            app.state.cache.ping()
        except RedisError as error:
            raise redis_unavailable() from error
    return {"status": "ok", "models": list(app.state.agents)}


@app.post("/v1/decisions", dependencies=[Depends(require_api_key)])
def decisions(body: DecisionRequest, response: Response):
    questions = {key: question.model_dump(exclude_none=True) for key, question in body.questions.items()}
    model = next(name for name, checkpoint in CHECKPOINTS.items() if body.model in (name, checkpoint))
    if model not in app.state.agents:
        raise HTTPException(status_code=422, detail=f"model '{model}' is not enabled on this server")
    response.headers["X-Cache"] = "BYPASS"
    try:
        cache = app.state.cache
        if cache is not None:
            payload = {"model": model, "checkpoint": CHECKPOINTS[model], "state": body.state, "questions": questions}
            # Preserve mapping order: choice option order can affect predictions.
            digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
            # ponytail: source-based identity; bump v1 if weights change at the same source.
            cache_key = "laya:decisions:v1:" + digest
        with app.state.inference_lock:
            if app.state.redis_failed:
                raise redis_unavailable()
            if cache is not None:
                try:
                    cached = cache.get(cache_key)
                    if cached is not None:
                        result = json.loads(cached)
                        if not isinstance(result, dict) or result.get("model") != model or not isinstance(result.get("answers"), dict):
                            raise ValueError("invalid cached response")
                        response.headers["X-Cache"] = "HIT"
                        return result
                except RedisError as error:
                    raise redis_unavailable() from error
                except (ValueError, TypeError):
                    logging.warning("Invalid cached response; using model")
            if app.state.redis_failed:
                raise redis_unavailable()
            result = {**app.state.agents[model].predict(body.state, questions), "model": model}
            if cache is not None:
                try:
                    cache.set(cache_key, json.dumps(result, ensure_ascii=False, allow_nan=False), ex=3600)
                    response.headers["X-Cache"] = "MISS"
                except RedisError as error:
                    raise redis_unavailable() from error
                except (ValueError, TypeError):
                    logging.warning("Cache serialization failed; returning model response")
        return result
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logging.exception("%s inference failed", model)
        raise HTTPException(status_code=500, detail="inference failed") from error


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8001")))
