# Laya, Oscar, and Hobson HTTP API

Run `mgoeckel/oscar-1-400m`, `convaiinnovations/laya`, and
`StrandsAgents/strands-decider-2B-hobson-v21` through `POST /v1/decisions`.
The server loads all three models once at startup unless you select one with `MODEL`.

The first run downloads the enabled checkpoints from Hugging Face. Hobson also
downloads its base model, `Qwen/Qwen3.5-2B-Base`. You'll need internet access and
enough memory to load the selected models.

## Run

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python server.py
```

If your GPU needs the CUDA 11.8 PyTorch build, install it before the other
requirements:

```sh
python3.13 -m venv .venv-cu118
. .venv-cu118/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python server.py
```

Hobson uses `strands-decider` and requires PyTorch 2.7 or newer and Transformers
5.15 or newer. Installing the requirements may upgrade an older PyTorch build.
Check that the installed build supports your GPU.

The default address is `http://127.0.0.1:8001`. To change it:

```sh
HOST=127.0.0.1 PORT=8080 python server.py
```

At the default address, open <http://127.0.0.1:8001/docs> for interactive API docs
or check the server with:

```sh
curl http://127.0.0.1:8001/health
```

Set `LAYA_DEVICE=cpu` or `LAYA_DEVICE=cuda` to choose a device for every model.
Otherwise, Laya chooses the device for Oscar and Laya. Hobson uses CUDA if
available, then MPS, then CPU.

## Choose a model

```sh
MODEL=laya python server.py
# Load only Oscar instead:
MODEL=oscar-1 python server.py
# Load only Hobson instead:
MODEL=hobson-v21 python server.py
```

Unset or empty `MODEL` loads all three models. An unsupported value stops startup
before any checkpoint loads. `/health` lists the loaded models, and requests
for a disabled model return HTTP 422.

Every decision request needs a `model`, even if the server has only one model
loaded. The server reads configuration at startup, so restart it after changing
environment variables.

## Send a request

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

Use `"model": "laya"` for Laya or `"model": "hobson-v21"` for Hobson. The full
checkpoint IDs also work, including
`"model": "StrandsAgents/strands-decider-2B-hobson-v21"`. Omitting `model` returns
HTTP 422.

`state` can be a string or a JSON object. For `choice`, supply criteria as a list
or an object mapping labels to descriptions. For `score`, supply a list. Hobson
accepts 2 to 10 score levels; longer rubrics return HTTP 422.

The server returns the selected backend's answers and usage unchanged, with
`model` set to `"oscar-1"`, `"laya"`, or `"hobson-v21"`. It adds no billing or
provider metadata.

## Require an API key

```sh
API_KEY="demo" python server.py
```

A non-empty `API_KEY` protects `/v1/decisions` and `/health`. Send it as a bearer
token:

```sh
curl http://127.0.0.1:8001/health -H 'Authorization: Bearer demo'
```

Add the same header to decision requests. Missing or invalid tokens return
HTTP 401. Unset or empty `API_KEY` turns authentication off.

The docs, OpenAPI schema, and game page stay public. Use the docs' **Authorize**
button to enter your token.

For access from other machines, set a strong key and put the server behind an
HTTPS reverse proxy. Run one worker to avoid loading duplicate copies of the
models.

## Cache responses with Redis

```sh
REDIS_URL="redis://127.0.0.1:6379/0" python server.py
```

Unset or empty `REDIS_URL` disables caching. When enabled, the server caches
successful responses for one hour. SHA-256 keys include the resolved model,
checkpoint source, state, and validated questions. Request formatting and model
aliases normalize to the same key. Criteria order stays intact because it can
affect predictions. The server checks authentication and model availability
before looking up a cached response.

Redis is required once you set `REDIS_URL`. The server checks the connection
before loading models and aborts startup if it fails. It also checks Redis every
five seconds, on `/health`, and during cache operations. A Redis error shuts down
the server rather than falling back to inference. Affected API requests return
HTTP 503. Connections use short timeouts with no automatic retries.

Successful responses include an `X-Cache` header with `HIT`, `MISS`, or `BYPASS`.
Errors are never cached. Cache hits return the original response, including its
usage.

Keep Redis private or authenticated. If you change weights or inference behavior
without changing the checkpoint source, clear this cache's keys or bump the
`v1` prefix in `server.py`.

To check caching, run:

```sh
python check_cache.py --url http://127.0.0.1:8001 --model oscar-1
```

The check sends eight identical requests with `Authorization: Bearer demo`.
If the server requires authentication, start it with `API_KEY="demo"`. Choose
an enabled model, such as `--model laya` for a Laya-only server.

The check expects one MISS followed by seven HITs, all with identical responses.
It prints timings and response bodies, and exits nonzero on failure. Each run
uses a unique marker so it starts with a cold cache key.

## Floppybird

Open <http://127.0.0.1:8001/floppybird.html> on the running server. Use your
server's host and port if they differ. Select a model, enter the API key if
required, and click **Start flight**.

The page checks `/health` for available models. Each turn, it sends two `noul`
questions to `/v1/decisions`: is the bird below the target height, and is it
falling? The target is the center of the next pipe opening. The state includes
height and motion in words as well as numbers.

The bird flaps only when both probabilities are at least 0.5. Otherwise it waits.
The simulation pauses during each API request, then advances 0.15 seconds. Only
one request runs at a time, and **Stop flight** cancels it.

Requests use the same origin, so no CORS setup is needed. API keys stay in page
memory and are never stored. There is no fallback autopilot. A model can crash
the bird.

## Check without downloading weights

```sh
pip install httpx
python -m unittest test_server.py
```

These tests stub model loading and inference, and exercise the real HTTP routes,
validation, and error handling. They also check Hobson's request and response
schemas. They do not test real-model inference.

With Node.js installed, check game physics, request payloads, cancellation, and
error handling without a browser or model:

```sh
node test_floppybird.js
```

To simulate a flight against the running API with reproducible pipe positions:

```sh
node test_floppybird.js --live http://127.0.0.1:8001
```

The live check stops on a crash or after 200 turns. Set `API_KEY` if the server
requires authentication. It tests a fixed rule driven by model judgments, not a
learned flight policy.

## Evaluate real models

With the server running, compare all three models against `cases.jsonl`:

```sh
python check_cases.py --url http://127.0.0.1:8001
# If the server requires authentication:
API_KEY="demo" python check_cases.py --url http://127.0.0.1:8001
```

Use `--models laya`, `--models oscar-1`, or `--models hobson-v21` to check one
model, and `--cases PATH` to use another JSONL file. The default URL is
`http://127.0.0.1:8001`.

Each row reports the expected answer, raw prediction, and probability assigned to
the expected answer. A `choice` answer passes on an exact label match. A `noul`
answer uses a 0.5 threshold. A `score` answer passes when it is less than 0.5 away
from the expected zero-based rubric index.

Check the probabilities too. A passing answer can still have weak confidence.
The check exits with 0 if all cases pass, 1 for mismatches, or 2 for HTTP or
connection errors.
