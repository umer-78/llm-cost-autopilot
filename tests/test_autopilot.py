import json

import httpx
import numpy as np
from fastapi.testclient import TestClient

from autopilot.router import Registry, Router, pick
from autopilot.service import create_app

CONFIG = {"frontier": "big", "models": {
    "big": {"input_per_million": 2.5, "output_per_million": 10.0},
    "small": {"input_per_million": 0.1, "output_per_million": 0.4},
    "mid": {"input_per_million": 0.5, "output_per_million": 1.5}}}


def test_registry_orders_models_cheapest_first_and_prices_tokens():
    reg = Registry.from_config(CONFIG)
    assert reg.names == ["small", "mid", "big"]
    assert abs(reg.cost("big", 1000, 200) - (1000 * 2.5 + 200 * 10.0) / 1e6) < 1e-12


def test_pick_takes_the_cheapest_confident_model_else_the_most_trusted():
    reg = Registry.from_config(CONFIG)
    chances = {"small": np.array([0.9, 0.2, 0.3]), "mid": np.array([0.95, 0.85, 0.4]), "big": np.array([0.99, 0.9, 0.6])}
    assert list(pick(reg, chances, 0.8)) == ["small", "mid", "big"]


def test_router_learns_which_rows_a_model_gets_right():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 3))
    reg = Registry.from_config(CONFIG)
    outcomes = {"small": (x[:, 0] > 0).astype(float), "mid": (x[:, 0] > -0.5).astype(float), "big": np.ones(400)}
    outcomes["big"][:5] = 0
    router = Router(reg, 0.8).fit(x, outcomes)
    easy, hard = np.array([[3.0, 0, 0]]), np.array([[-3.0, 0, 0]])
    assert router.route(easy)[0][0] == "small" and router.route(hard)[0][0] == "big"


def test_service_routes_forwards_and_counts_the_saving(monkeypatch):
    reg = Registry.from_config(CONFIG)

    class Fixed:
        registry, confidence = reg, 0.8

        def route(self, x):
            return [("small", "test")]

    calls = []

    def upstream(request):
        body = json.loads(request.content)
        calls.append(body["model"])
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 1000, "completion_tokens": 200}})

    monkeypatch.setenv("AUTOPILOT_SMALL_URL", "http://small/v1")
    client = TestClient(create_app(Fixed(), lambda p, t: [0.0], ["qa"], transport=httpx.MockTransport(upstream)))
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.headers["x-routed-model"] == "small" and calls == ["small"]
    stats = client.get("/v1/stats").json()
    assert stats["requests"] == 1 and 0.9 < stats["saving"] < 1.0
    assert client.put("/v1/routing-config", json={"confidence": 0.9}).json()["confidence"] == 0.9
