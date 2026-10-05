"""Run with: python -m unittest test_server.py (requires httpx)."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from server import CHECKPOINT, app


class ServerTest(unittest.TestCase):
    def test_http_api(self):
        questions = {
            "intent": {"type": "choice", "instructions": "What does the customer want?",
                       "criteria": ["refund", "cancel"]},
            "urgency": {"type": "score", "instructions": "Rate urgency.",
                        "criteria": ["routine", "critical"]},
            "sensitive": {"type": "noul", "instructions": "Is this a security issue?"},
        }
        result = {
            "answers": {
                "intent": {"type": "choice", "choice": "refund", "probabilities": {"refund": 0.9, "cancel": 0.1}},
                "urgency": {"type": "score", "score": 0.2},
                "sensitive": {"type": "noul", "noul": 0.1},
            },
            "usage": {"input_tokens": 42, "output_tokens": 0},
        }
        agent = Mock()
        agent.predict.return_value = result
        load = Mock(return_value=agent)
        with patch.dict(sys.modules, {"laya": SimpleNamespace(load=load)}):
            with TestClient(app) as client:
                load.assert_called_once()
                self.assertEqual(load.call_args.args, (CHECKPOINT,))
                self.assertEqual(client.get("/health").json(), {"status": "ok", "model": "oscar-1"})
                for state in ("Zoë needs a refund.", {"text": "Zoë needs a refund."}):
                    response = client.post("/v1/decisions", json={"state": state, "questions": questions})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json(), {**result, "model": "oscar-1"})
                    agent.predict.assert_called_with(state, questions)

                calls = agent.predict.call_count
                for invalid in (
                    {"model": "needle-research", "state": "hello", "questions": questions},
                    {"state": "hello", "questions": {}},
                    {"state": "hello", "questions": {"x": {"type": "choice", "instructions": "Pick one"}}},
                    {"state": "hello", "questions": {"x": {"type": "score", "instructions": "Rate", "criteria": {"a": "low", "b": "high"}}}},
                    {"state": "x" * 50001, "questions": questions},
                    {"state": {"text": "x" * 50001}, "questions": questions},
                    {"state": 123, "questions": questions},
                ):
                    self.assertEqual(client.post("/v1/decisions", json=invalid).status_code, 422)
                self.assertEqual(client.post("/v1/decisions", content="not JSON").status_code, 422)
                self.assertEqual(agent.predict.call_count, calls)

                agent.predict.side_effect = ValueError("invalid question")
                self.assertEqual(client.post("/v1/decisions", json={"state": "hello", "questions": questions}).status_code, 422)
                agent.predict.side_effect = RuntimeError("private checkpoint path")
                with self.assertLogs(level="ERROR"):
                    response = client.post("/v1/decisions", json={"state": "hello", "questions": questions})
                self.assertEqual(response.status_code, 500)
                self.assertEqual(response.json(), {"detail": "inference failed"})
        self.assertFalse(hasattr(app.state, "agent"))


if __name__ == "__main__":
    unittest.main()
