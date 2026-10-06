"""Run with: python -m unittest test_server.py (requires httpx)."""

import asyncio
import json
import os
import signal
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient
from redis import RedisError

from check_cases import assess
from server import CHECKPOINTS, app, watch_redis


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
        with patch.dict(os.environ, {"API_KEY": "", "MODEL": "", "REDIS_URL": ""}), patch.dict(sys.modules, {"laya": SimpleNamespace(load=load)}):
            with TestClient(app) as client:
                self.assertEqual(load.call_count, len(CHECKPOINTS))
                self.assertEqual([call.args for call in load.call_args_list], [(source,) for source in CHECKPOINTS.values()])
                self.assertEqual(client.get("/health").json(), {"status": "ok", "models": list(CHECKPOINTS)})
                schema = client.get("/openapi.json")
                self.assertEqual(schema.status_code, 200)
                model_schema = schema.json()["components"]["schemas"]["DecisionRequest"]["properties"]["model"]
                self.assertEqual(set(model_schema["enum"]), set(CHECKPOINTS) | set(CHECKPOINTS.values()))
                self.assertIn("model", schema.json()["components"]["schemas"]["DecisionRequest"]["required"])
                for state in ("Zoë needs a refund.", {"text": "Zoë needs a refund."}):
                    response = client.post("/v1/decisions", json={"model": "oscar-1", "state": state, "questions": questions})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json(), {**result, "model": "oscar-1"})
                    self.assertEqual(response.headers["x-cache"], "BYPASS")
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
                    {"state": "hello", "questions": questions},
                    {"model": "needle-research", "state": "hello", "questions": questions},
                    {"model": "oscar-1", "state": "hello", "questions": {}},
                    {"model": "oscar-1", "state": "hello", "questions": {"x": {"type": "choice", "instructions": "Pick one"}}},
                    {"model": "oscar-1", "state": "hello", "questions": {"x": {"type": "score", "instructions": "Rate", "criteria": {"a": "low", "b": "high"}}}},
                    {"model": "oscar-1", "state": "x" * 50001, "questions": questions},
                    {"model": "oscar-1", "state": {"text": "x" * 50001}, "questions": questions},
                    {"model": "oscar-1", "state": 123, "questions": questions},
                ):
                    self.assertEqual(client.post("/v1/decisions", json=invalid).status_code, 422)
                self.assertEqual(client.post("/v1/decisions", content="not JSON").status_code, 422)
                self.assertEqual([agent.predict.call_count for agent in agents.values()], calls)

                agent.predict.side_effect = ValueError("invalid question")
                self.assertEqual(client.post("/v1/decisions", json={"model": "oscar-1", "state": "hello", "questions": questions}).status_code, 422)
                agent.predict.side_effect = RuntimeError("private checkpoint path")
                with self.assertLogs(level="ERROR"):
                    response = client.post("/v1/decisions", json={"model": "oscar-1", "state": "hello", "questions": questions})
                self.assertEqual(response.status_code, 500)
                self.assertEqual(response.json(), {"detail": "inference failed"})
        self.assertFalse(hasattr(app.state, "agents"))

    def test_model_selection(self):
        body = {"model": "oscar-1", "state": "hello", "questions": {"x": {"type": "noul", "instructions": "Is this a greeting?"}}}
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
                    # The model is required, even with only one checkpoint enabled.
                    self.assertEqual(client.post("/v1/decisions", json={key: value for key, value in body.items() if key != "model"}).status_code, 422)
                    self.assertEqual(load.call_count, len(enabled))

    def test_redis_availability(self):
        cache, load = Mock(), Mock()
        cache.ping.side_effect = RedisError("offline")
        with patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379/0"}, clear=True), \
             patch.dict(sys.modules, {"laya": SimpleNamespace(load=load)}), \
             patch("server.Redis.from_url", return_value=cache):
            with self.assertRaisesRegex(RuntimeError, "Redis unavailable; server startup aborted"):
                with TestClient(app):
                    pass
            load.assert_not_called()
            cache.close.assert_called_once()

            app.state.redis_failed = False
            with patch("server.asyncio.sleep", new=AsyncMock()), patch("server.os.kill") as stop:
                with self.assertLogs(level="ERROR"):
                    asyncio.run(watch_redis())
                self.assertTrue(app.state.redis_failed)
                stop.assert_called_once_with(os.getpid(), signal.SIGTERM)

            cache.ping.side_effect = [True, RedisError("offline")]
            with patch("server.os.kill") as stop:
                with TestClient(app) as client:
                    with self.assertLogs(level="ERROR"):
                        self.assertEqual(client.get("/health").status_code, 503)
                    stop.assert_called_once_with(os.getpid(), signal.SIGTERM)

    def test_redis_cache(self):
        body = {"model": "oscar-1", "state": "Zoë sees a red ball.", "questions": {
            "color": {"type": "choice", "instructions": "What color is the ball?", "criteria": {"r": "Red", "b": "Blue"}}}}
        agent = Mock()
        agent.predict.return_value = {"answers": {"color": {"type": "choice", "choice": "r"}}}
        store = {}
        cache = Mock()
        cache.ping.return_value = True
        cache.get.side_effect = store.get
        cache.set.side_effect = lambda key, value, **kwargs: store.__setitem__(key, value)
        with patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379/0", "API_KEY": "demo"}, clear=True), \
             patch.dict(sys.modules, {"laya": SimpleNamespace(load=Mock(return_value=agent))}), \
             patch("server.Redis.from_url", return_value=cache) as connect, \
             patch("server.os.kill") as stop:
            with TestClient(app, headers={"Authorization": "Bearer demo"}) as client:
                first = client.post("/v1/decisions", json=body)
                self.assertEqual(first.status_code, 200)
                self.assertEqual(first.headers["x-cache"], "MISS")
                agent.predict.assert_called_once()
                cache_key = next(iter(store))
                self.assertRegex(cache_key, r"^laya:decisions:v1:[0-9a-f]{64}$")
                self.assertEqual(cache.set.call_args.kwargs, {"ex": 3600})
                equivalent = {"questions": body["questions"], "state": body["state"], "model": CHECKPOINTS["oscar-1"]}
                hit = client.post("/v1/decisions", content=json.dumps(equivalent, indent=4), headers={"Content-Type": "application/json"})
                self.assertEqual(hit.headers["x-cache"], "HIT")
                self.assertEqual(hit.json(), first.json())
                agent.predict.assert_called_once()

                reordered = {**body, "questions": {"color": {**body["questions"]["color"], "criteria": {"b": "Blue", "r": "Red"}}}}
                for changed in (reordered, {**body, "model": "laya"}, {**body, "state": "Another state"}):
                    self.assertEqual(client.post("/v1/decisions", json=changed).headers["x-cache"], "MISS")
                self.assertEqual(agent.predict.call_count, 4)
                self.assertEqual(len(store), 4)

                reads = cache.get.call_count
                self.assertEqual(client.post("/v1/decisions", json=body, headers={"Authorization": "Bearer wrong"}).status_code, 401)
                self.assertEqual(client.post("/v1/decisions", json={key: value for key, value in body.items() if key != "model"}).status_code, 422)
                self.assertEqual(client.post("/v1/decisions", json={**body, "model": "unknown"}).status_code, 422)
                self.assertEqual(cache.get.call_count, reads)

                for corrupt in ("not JSON", "null", json.dumps({"model": "laya", "answers": {}})):
                    store[cache_key] = corrupt
                    with self.assertLogs(level="WARNING"):
                        repaired = client.post("/v1/decisions", json=body)
                    self.assertEqual(repaired.headers["x-cache"], "MISS")
                    self.assertEqual(repaired.json(), first.json())

                writes = cache.set.call_count
                agent.predict.side_effect = ValueError("invalid question")
                self.assertEqual(client.post("/v1/decisions", json={**body, "state": "Model failure"}).status_code, 422)
                agent.predict.side_effect = RuntimeError("model failure")
                with self.assertLogs(level="ERROR"):
                    self.assertEqual(client.post("/v1/decisions", json={**body, "state": "Model failure"}).status_code, 500)
                self.assertEqual(cache.set.call_count, writes)
                agent.predict.side_effect = None

                cache.set.side_effect = RedisError("offline")
                with self.assertLogs(level="ERROR"):
                    failed = client.post("/v1/decisions", json={**body, "state": "Write failure"})
                self.assertEqual(failed.status_code, 503)
                self.assertEqual(failed.json(), {"detail": "Redis unavailable; server is shutting down"})
                stop.assert_called_once_with(os.getpid(), signal.SIGTERM)
                calls, reads = agent.predict.call_count, cache.get.call_count
                self.assertEqual(client.post("/v1/decisions", json=body).status_code, 503)
                self.assertEqual(agent.predict.call_count, calls)
                self.assertEqual(cache.get.call_count, reads)
                self.assertEqual(stop.call_count, 1)
            cache.close.assert_called_once()
            self.assertEqual(connect.call_args.args, ("redis://localhost:6379/0",))

            cache.get.side_effect = RedisError("offline")
            with TestClient(app, headers={"Authorization": "Bearer demo"}) as client:
                calls, writes = agent.predict.call_count, cache.set.call_count
                with self.assertLogs(level="ERROR"):
                    self.assertEqual(client.post("/v1/decisions", json=body).status_code, 503)
                self.assertEqual(agent.predict.call_count, calls)
                self.assertEqual(cache.set.call_count, writes)
                self.assertEqual(stop.call_count, 2)
            # A cached answer must not enable a model disabled on the next startup.
            with patch.dict(os.environ, {"MODEL": "oscar-1"}):
                with TestClient(app, headers={"Authorization": "Bearer demo"}) as client:
                    reads = cache.get.call_count
                    self.assertEqual(client.post("/v1/decisions", json={**body, "model": "laya"}).status_code, 422)
                    self.assertEqual(cache.get.call_count, reads)

    def test_optional_api_key(self):
        body = {"model": "oscar-1", "state": "hello", "questions": {"x": {"type": "noul", "instructions": "Is this a greeting?"}}}
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
