"""Run with: python -m unittest test_server.py (requires httpx)."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from check_cases import assess
from server import CHECKPOINTS, app


class ServerTest(unittest.TestCase):
    def test_case_assessment(self):
        for expected, probability, passes in ((True, 0.5866, True), (False, 0.1, True),
                                               (True, 0.1, False), (False, 0.8, False)):
            ok, actual, p_expected = assess({"type": "noul", "expected": expected}, {"noul": probability})
            self.assertEqual(ok, passes)
            self.assertEqual(actual, probability)
            self.assertAlmostEqual(p_expected, probability if expected else 1 - probability)
        for choice in ("r", "b"):
            self.assertEqual(assess({"type": "choice", "expected": "r"},
                                   {"choice": choice, "probabilities": {"r": 0.9, "b": 0.1}}),
                             (choice == "r", choice, 0.9))
        for score, passes in ((3.7, True), (0.9, False), (3.5, False)):
            self.assertEqual(assess({"type": "score", "expected": 4},
                                   {"score": score, "probabilities": {"4": 0.7}}),
                             (passes, score, 0.7))

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
        agents = {name: Mock() for name in CHECKPOINTS}
        for i, selected in enumerate(agents.values()):
            selected.predict.return_value = {**result, "usage": {"input_tokens": 42 + i, "output_tokens": 0}}
        agent = agents["oscar-1"]
        load = Mock(side_effect=lambda checkpoint, **kwargs: agents[next(
            name for name, source in CHECKPOINTS.items() if source == checkpoint
        )])
        with patch.dict(os.environ, {"API_KEY": "", "MODEL": ""}), patch.dict(sys.modules, {"laya": SimpleNamespace(load=load)}):
            with TestClient(app) as client:
                self.assertEqual(load.call_count, len(CHECKPOINTS))
                self.assertEqual([call.args for call in load.call_args_list], [(source,) for source in CHECKPOINTS.values()])
                self.assertEqual(client.get("/health").json(), {"status": "ok", "models": list(CHECKPOINTS)})
                schema = client.get("/openapi.json")
                self.assertEqual(schema.status_code, 200)
                model_schema = schema.json()["components"]["schemas"]["DecisionRequest"]["properties"]["model"]
                self.assertEqual(set(model_schema["enum"]), set(CHECKPOINTS) | set(CHECKPOINTS.values()))
                for state in ("Zoë needs a refund.", {"text": "Zoë needs a refund."}):
                    response = client.post("/v1/decisions", json={"state": state, "questions": questions})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json(), {**result, "model": "oscar-1"})
                    agent.predict.assert_called_with(state, questions)

                for model, name in ((alias, name) for name, source in CHECKPOINTS.items() for alias in (name, source)):
                    selected = agents[name]
                    other_calls = {key: value.predict.call_count for key, value in agents.items() if key != name}
                    response = client.post("/v1/decisions", json={"model": model, "state": "hello", "questions": questions})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json(), {**selected.predict.return_value, "model": name})
                    selected.predict.assert_called_with("hello", questions)
                    self.assertEqual({key: value.predict.call_count for key, value in agents.items() if key != name}, other_calls)
                self.assertEqual(load.call_count, len(CHECKPOINTS))

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

    def test_model_selection(self):
        body = {"state": "hello", "questions": {"x": {"type": "noul", "instructions": "Is this a greeting?"}}}
        for selection in (None, "", *CHECKPOINTS, "unknown"):
            agent = Mock()
            agent.predict.return_value = {"answers": {"x": {"type": "noul", "noul": 0.9}}}
            load = Mock(return_value=agent)
            with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {"laya": SimpleNamespace(load=load)}):
                if selection is not None:
                    os.environ["MODEL"] = selection
                if selection == "unknown":
                    with self.assertRaisesRegex(ValueError, "MODEL must be"):
                        with TestClient(app):
                            pass
                    load.assert_not_called()
                    continue
                enabled = [selection] if selection else list(CHECKPOINTS)
                with TestClient(app) as client:
                    self.assertEqual([call.args for call in load.call_args_list], [(CHECKPOINTS[name],) for name in enabled])
                    self.assertEqual(client.get("/health").json(), {"status": "ok", "models": enabled})
                    for name, source in CHECKPOINTS.items():
                        for model in (name, source):
                            calls = agent.predict.call_count
                            response = client.post("/v1/decisions", json={**body, "model": model})
                            self.assertEqual(response.status_code, 200 if name in enabled else 422)
                            if name in enabled:
                                self.assertEqual(response.json()["model"], name)
                            else:
                                self.assertEqual(response.json(), {"detail": f"model '{name}' is not enabled on this server"})
                                self.assertEqual(agent.predict.call_count, calls)
                    # Omitting the request model still defaults to Oscar.
                    self.assertEqual(client.post("/v1/decisions", json=body).status_code, 200 if "oscar-1" in enabled else 422)
                    self.assertEqual(load.call_count, len(enabled))

    def test_optional_api_key(self):
        body = {"state": "hello", "questions": {"x": {"type": "noul", "instructions": "Is this a greeting?"}}}
        for key in (None, "", "demo"):
            agent = Mock()
            agent.predict.return_value = {"answers": {"x": {"type": "noul", "noul": 0.9}}}
            with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {"laya": SimpleNamespace(load=Mock(return_value=agent))}):
                if key is not None:
                    os.environ["API_KEY"] = key
                with TestClient(app) as client:
                    page = client.get("/floppybird.html")
                    self.assertEqual(page.status_code, 200)
                    self.assertTrue(page.headers["content-type"].startswith("text/html"))
                    self.assertIn('id="game"', page.text)
                    self.assertIn('"/v1/decisions"', page.text)
                    self.assertEqual(client.get("/server.py").status_code, 404)
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
