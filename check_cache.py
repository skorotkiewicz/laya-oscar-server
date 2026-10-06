"""Send eight requests with Bearer demo and verify one cache miss followed by seven hits."""

import argparse
import json
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="oscar-1")
    args = parser.parse_args()
    payload = {
        "model": args.model,
        # A new marker per run guarantees the first request should be a cache miss.
        "state": {"text": "The red ball is inside the blue box.", "cache_test": uuid4().hex},
        "questions": {"box": {"type": "noul", "instructions": "Is the box blue?"}},
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": "Bearer demo"}
    statuses, results = [], []
    for i in range(8):
        started = perf_counter()
        request = Request(args.url.rstrip("/") + "/v1/decisions", data=data, headers=headers)
        try:
            with urlopen(request, timeout=60) as response:
                statuses.append(response.headers.get("X-Cache", "MISSING"))
                result = json.load(response)
        except HTTPError as error:
            print(f"HTTP {error.code}: {error.read().decode('utf-8', errors='replace')}")
            return 2
        except (URLError, TimeoutError, ValueError) as error:
            print(f"Request failed: {error}")
            return 2
        results.append(result)
        elapsed = (perf_counter() - started) * 1000
        print(f"{i + 1}/8 {statuses[-1]:7} {elapsed:8.1f} ms  {json.dumps(result, ensure_ascii=False)}")
    if statuses == ["MISS"] + ["HIT"] * 7 and all(result == results[0] for result in results):
        print("PASS: one MISS, seven HITs, all responses identical.")
        return 0
    print("FAIL: expected one MISS, seven HITs, and identical responses. Check REDIS_URL and restart the server.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
