"""Run with: python -m unittest test_server.py (requires httpx)."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from server import CHECKPOINTS, app


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
        laya_agent = Mock()
        laya_agent.predict.return_value = {**result, "usage": {"input_tokens": 100, "output_tokens": 0}}
        agents = {"oscar-1": agent, "laya": laya_agent}
        load = Mock(side_effect=lambda checkpoint, **kwargs: agents[next(
            name for name, source in CHECKPOINTS.items() if source == checkpoint
        )])
        with patch.dict(os.environ, {"API_KEY": ""}), patch.dict(sys.modules, {"laya": SimpleNamespace(load=load)}):
            with TestClient(app) as client:
                self.assertEqual(load.call_count, 2)
                self.assertEqual([call.args for call in load.call_args_list], [(source,) for source in CHECKPOINTS.values()])
                self.assertEqual(client.get("/health").json(), {"status": "ok", "models": ["oscar-1", "laya"]})
                for state in ("Zoë needs a refund.", {"text": "Zoë needs a refund."}):
                    response = client.post("/v1/decisions", json={"state": state, "questions": questions})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json(), {**result, "model": "oscar-1"})
                    agent.predict.assert_called_with(state, questions)

                for model, name in (
                    ("oscar-1", "oscar-1"), ("mgoeckel/oscar-1-400m", "oscar-1"),
                    ("laya", "laya"), ("convaiinnovations/laya", "laya"),
                    ("oscar-1", "oscar-1"),
                ):
                    selected, other = agents[name], agents["laya" if name == "oscar-1" else "oscar-1"]
                    other_calls = other.predict.call_count
                    response = client.post("/v1/decisions", json={"model": model, "state": "hello", "questions": questions})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json(), {**selected.predict.return_value, "model": name})
                    selected.predict.assert_called_with("hello", questions)
                    self.assertEqual(other.predict.call_count, other_calls)
                self.assertEqual(load.call_count, 2)

                calls = [agent.predict.call_count for agent in agents.values()]
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
                self.assertEqual([agent.predict.call_count for agent in agents.values()], calls)

                agent.predict.side_effect = ValueError("invalid question")
                self.assertEqual(client.post("/v1/decisions", json={"state": "hello", "questions": questions}).status_code, 422)
                agent.predict.side_effect = RuntimeError("private checkpoint path")
                with self.assertLogs(level="ERROR"):
                    response = client.post("/v1/decisions", json={"state": "hello", "questions": questions})
                self.assertEqual(response.status_code, 500)
                self.assertEqual(response.json(), {"detail": "inference failed"})
        self.assertFalse(hasattr(app.state, "agents"))

    def test_optional_api_key(self):
        body = {"state": "hello", "questions": {"x": {"type": "noul", "instructions": "Is this a greeting?"}}}
        for key in (None, "", "demo"):
            agent = Mock()
            agent.predict.return_value = {"answers": {"x": {"type": "noul", "noul": 0.9}}}
            with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {"laya": SimpleNamespace(load=Mock(return_value=agent))}):
                if key is not None:
                    os.environ["API_KEY"] = key
                with TestClient(app) as client:
                    # The key is captured at startup, not re-read for each request.
                    os.environ["API_KEY"] = "changed"
                    for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic demo"},
                                    {"Authorization": "demo"}, {"Authorization": "Bearer"},
                                    {"Authorization": b"Bearer \xff"}):
                        response = client.post("/v1/decisions", json=body, headers=headers)
                        self.assertEqual(response.status_code, 401 if key else 200)
                        self.assertEqual(client.get("/health", headers=headers).status_code, 401 if key else 200)
                        if key:
                            self.assertEqual(response.headers["www-authenticate"], "Bearer")
                            self.assertEqual(response.json(), {"detail": "invalid or missing access token"})
                    if key:
                        agent.predict.assert_not_called()
                    headers = {"Authorization": "Bearer demo"}
                    self.assertEqual(client.post("/v1/decisions", json=body, headers=headers).status_code, 200)
                    self.assertEqual(client.get("/health", headers=headers).status_code, 200)


if __name__ == "__main__":
    unittest.main()
