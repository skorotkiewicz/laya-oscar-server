**Laya 0.3.28 already includes a trainer**, so we don’t need a custom PyTorch loop.

But our **30 cases are too small for meaningful general training**. They’re useful for testing the pipeline; ideally keep them as a benchmark and collect separate training examples.

### 1. Convert the format

The trainer expects `state`, `questions`, and an `expected` mapping:

```json
{
  "state": "Do not refund the order. I want to keep it.",
  "questions": {
    "refund": {
      "type": "noul",
      "instructions": "Does the customer request a refund?"
    }
  },
  "expected": {"refund": false}
}
```

This converts our existing file without changing it:

```python
import json

with open("cases.jsonl", encoding="utf-8") as source, \
     open("train.jsonl", "x", encoding="utf-8") as output:
    for i, line in enumerate(source):
        if not line.strip():
            continue
        case = json.loads(line)
        qid = f"case_{i}"
        question = {
            k: case[k]
            for k in ("type", "instructions", "criteria")
            if k in case
        }
        row = {
            "state": case["state"],
            "questions": {qid: question},
            "expected": {qid: case["expected"]},
        }
        output.write(json.dumps(row, ensure_ascii=False) + "\n")
```

### 2. Fine-tune Oscar

A cheap first experiment trains only the decision heads:

```sh
python -m laya.train_cli \
  --base mgoeckel/oscar-1-400m \
  --data train.jsonl \
  --out ./oscar-local \
  --loss soft-ce \
  --freeze-encoder \
  --shuffle-options \
  --label-smoothing 0.05 \
  --epochs 3 \
  --micro-batch 2 \
  --grad-accum 1 \
  --device auto
```

`--shuffle-options` helps discourage option-position bias. Freezing the encoder reduces training cost, but limits what the model can learn.

### 3. Load the result

```python
agent = laya.load("./oscar-local")
```

To serve it, point the existing Oscar entry in `CHECKPOINTS` at that directory.

### For actual improvement

- Collect varied examples, especially negation and object relationships.
- Include counterexamples, renamed labels, and different option orders.
- Split by underlying scenario, not individual question, to avoid leakage.
- Evaluate on untouched cases. Better results on examples used for training mostly show memorization.

With more data, we can unfreeze the encoder using a small learning rate. **Fixing probabilities and improving reasoning are separate goals.**
