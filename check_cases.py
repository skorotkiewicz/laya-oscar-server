"""Evaluate real HTTP predictions against cases.jsonl, without downloading models."""

import argparse
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def assess(case, answer):
    kind, expected = case["type"], case["expected"]
    actual = answer[kind]
    if kind == "noul":
        return (actual >= 0.5) == expected, actual, actual if expected else 1 - actual
    probability = answer["probabilities"][str(expected)]
    if kind == "choice":
        return actual == expected, actual, probability
    # ponytail: fixed half-level tolerance; use task-specific tolerances for subjective rubrics.
    return abs(actual - expected) < 0.5, actual, probability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.jsonl"))
    parser.add_argument("--models", nargs="+", choices=["oscar-1", "laya"], default=["oscar-1", "laya"])
    args = parser.parse_args()
    with args.cases.open(encoding="utf-8") as file:
        cases = [(line_number, json.loads(line)) for line_number, line in enumerate(file, 1) if line.strip()]
    if not cases:
        parser.error("cases file is empty")
    headers = {"Content-Type": "application/json"}
    if key := os.getenv("API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    failed = False
    for model in args.models:
        passed = 0
        for line_number, case in cases:
            question = {key: case[key] for key in ("type", "instructions", "criteria") if key in case}
            payload = {"model": model, "state": case["state"], "questions": {"case": question}}
            request = Request(args.url.rstrip("/") + "/v1/decisions",
                              data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers=headers)
            try:
                with urlopen(request, timeout=60) as response:
                    result = json.load(response)
            except HTTPError as error:
                print(f"{model} case {line_number}: HTTP {error.code}: {error.read().decode('utf-8', errors='replace')}")
                return 2
            except (URLError, TimeoutError) as error:
                print(f"{model} case {line_number}: request failed: {error}")
                return 2
            if result["model"] != model:
                print(f"Requested {model}, but server returned {result['model']}")
                return 2
            ok, actual, probability = assess(case, result["answers"]["case"])
            passed += ok
            print(f"{model:7} #{line_number:02} {'PASS' if ok else 'FAIL'} {case['type']:6} "
                  f"expected={case['expected']!r} got={actual!r} p(expected)={probability:.4f} "
                  f"| {case['instructions']}")
        print(f"\n{model}: {passed}/{len(cases)} passed ({passed / len(cases):.1%})\n")
        failed |= passed != len(cases)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
