# Laya and Oscar HTTP API

Serves `mgoeckel/oscar-1-400m` and `convaiinnovations/laya` via `POST /v1/decisions`.
Both models load once at startup. First startup downloads the checkpoints
from Hugging Face and requires internet access and enough memory for both models.

## Run

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python server.py
```

Listens on `127.0.0.1:8001`. Interactive API docs: <http://127.0.0.1:8001/docs>.
Health check: `curl http://127.0.0.1:8001/health`.
Set `LAYA_DEVICE=cpu` or `LAYA_DEVICE=cuda` to request a device; otherwise Laya chooses.

## Request

```sh
curl http://127.0.0.1:8001/v1/decisions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "oscar-1",
    "state": "I was charged twice, please return the money.",
    "questions": {
      "intent": {
        "type": "choice",
        "instructions": "What does the customer want?",
        "criteria": ["refund", "cancel", "information", "other"]
      },
      "urgency": {
        "type": "score",
        "instructions": "Rate urgency.",
        "criteria": ["routine", "low", "moderate", "high", "critical"]
      },
      "sensitive": {
        "type": "noul",
        "instructions": "Is this a fraud/security issue?"
      }
    }
  }'
```

`model` is optional and defaults to `oscar-1`. Change `"model": "oscar-1"` to
`"model": "laya"` to use `convaiinnovations/laya`. Both full checkpoint IDs are also accepted.
`state` accepts a string or a JSON object. Choice criteria accept a label-to-description
object or a list; score criteria require a list. Answers and usage pass through from
Laya unchanged, with `model` set to `"oscar-1"` or `"laya"` to match the selected model.
No fabricated billing or provider metadata.

This is a local API without authentication. Use an authenticated reverse proxy before
exposing it to other machines. Run one worker to avoid duplicate model loads.

## Check without downloading weights

```sh
pip install httpx
python -m unittest test_server.py
```

The check stubs Laya and exercises the real HTTP routes, validation, and error handling.
It does not verify real-model inference.
