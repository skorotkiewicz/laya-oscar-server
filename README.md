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

Defaults to `HOST=127.0.0.1` and `PORT=8001`. Override with:

```sh
HOST=127.0.0.1 PORT=8080 python server.py
```

Interactive API docs (default address): <http://127.0.0.1:8001/docs>.
Health check: `curl http://127.0.0.1:8001/health`.
Set `LAYA_DEVICE=cpu` or `LAYA_DEVICE=cuda` to request a device; otherwise Laya chooses.

## Optional API key

```sh
API_KEY="demo" python server.py
```

When `API_KEY` is set to a non-empty value, `/v1/decisions` and `/health` require
an `Authorization: Bearer <API_KEY>` header. Add it to your curl requests:

```sh
curl http://127.0.0.1:8001/health -H 'Authorization: Bearer demo'
```

Missing or invalid tokens return HTTP 401. Unset or empty `API_KEY` disables
authentication. The key is read at startup. Interactive docs and the OpenAPI schema
remain public; use the docs' **Authorize** button to enter your token.

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

For access from other machines, set a strong `API_KEY` and use HTTPS through a reverse
proxy to protect the token in transit. Run one worker to avoid duplicate model loads.

## Check without downloading weights

```sh
pip install httpx
python -m unittest test_server.py
```

The check stubs Laya and exercises the real HTTP routes, validation, and error handling.
It does not verify real-model inference.
