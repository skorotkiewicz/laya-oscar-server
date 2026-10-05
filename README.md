# Laya and Oscar HTTP API

Serves `mgoeckel/oscar-1-400m` and `convaiinnovations/laya` via `POST /v1/decisions`.
By default, both models load once at startup. Set `MODEL` to load only one.
First startup downloads the enabled checkpoints from Hugging Face and requires
internet access and enough memory for those models.

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

## Optional model selection

```sh
MODEL="laya" python server.py
# Or load only Oscar:
MODEL="oscar-1" python server.py
```

Unset or empty `MODEL` keeps the current behavior: both models load.
Requests for a disabled model return HTTP 422; `/health` lists only loaded models.
An unsupported `MODEL` value fails at startup before loading any checkpoint.
The request `model` still defaults to `oscar-1`, so include `"model": "laya"`
when running in Laya-only mode. Configuration is read at startup.

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
authentication. The key is read at startup. Interactive docs, the OpenAPI schema,
and the game page remain public; use the docs' **Authorize** button to enter your token.

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

## Floppybird

Open <http://127.0.0.1:8001/floppybird.html> on the running server (use its actual
host and port if different). Select a model, enter the API key if required, and
click **Start flight**. The page detects enabled models from `/health` and sends
two `noul` questions to `/v1/decisions`: is the bird below the target height, and is
it falling? The target is the next pipe opening's center. The explicit rule is
**flap only when both probabilities are at least 0.5; otherwise wait**. Height and
motion are also described in words. Requests use the same origin, so no CORS setup is needed.

The simulation pauses during each API request, then advances 0.15 seconds.
Only one request is in flight; **Stop flight** cancels it. API keys stay in the page's
memory and are never stored. A model may crash the bird; there is no fallback autopilot.

## Check without downloading weights

```sh
pip install httpx
python -m unittest test_server.py
```

The check stubs Laya and exercises the real HTTP routes, validation, and error handling.
It does not verify real-model inference. With Node.js installed, check game physics,
HTTP payloads, cancellation, and error handling without a browser or model:

```sh
node test_floppybird.js
# Simulate a flight with the real API and reproducible pipe positions:
node test_floppybird.js --live http://192.168.0.124:8001
```

The live check stops on a crash or after 200 turns. Set `API_KEY` for an authenticated
server. This checks a simple rule driven by model judgments, not a learned flight policy.

## Evaluate real models

With the server running, compare both models against `cases.jsonl`:

```sh
python check_cases.py --url http://192.168.0.124:8001
# If authentication is enabled:
API_KEY="demo" python check_cases.py --url http://192.168.0.124:8001
```

Use `--models laya` or `--models oscar-1` to check one model, and `--cases PATH`
for another JSONL file. The default server URL is `http://127.0.0.1:8001`.
Each row reports the expected answer, raw prediction, and probability assigned to the
expected answer. A choice passes on an exact label match; noul uses a 0.5 threshold;
score passes when it is less than 0.5 away from the expected zero-based rubric index.
A passing answer can still have weak confidence, so inspect the probabilities too.
Exit codes: 0 for all cases passing, 1 for mismatches, 2 for HTTP/connection errors.
