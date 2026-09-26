"""The router as a service: an OpenAI-style chat completions endpoint where the caller does
not pick the model. Each registry model is reached through an OpenAI-compatible base URL
set in AUTOPILOT_<NAME>_URL and AUTOPILOT_<NAME>_KEY (NAME upper-cased, dashes and dots as
underscores). Not measured here; the benchmark runs the router on recorded answers."""
import os
import time

import httpx
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def env_name(model):
    return model.upper().replace("-", "_").replace(".", "_")


def create_app(router, featurize, task_names, transport=None):
    """router: a fitted Router; featurize(prompt, task) -> feature row."""
    app = FastAPI(title="llm-cost-autopilot")
    stats = {"requests": 0, "cost": 0.0, "frontier_cost": 0.0, "by_model": {}}
    reg = router.registry

    @app.get("/v1/models")
    async def models():
        return {"models": [{"name": n, "input_per_million": reg.price[n][0] * 1e6, "output_per_million": reg.price[n][1] * 1e6}
                           for n in reg.names], "confidence": router.confidence}

    @app.put("/v1/routing-config")
    async def config(request: Request):
        body = await request.json()
        if "confidence" in body:
            router.confidence = float(body["confidence"])
        return {"confidence": router.confidence}

    @app.get("/v1/stats")
    async def get_stats():
        saved = stats["frontier_cost"] - stats["cost"]
        return {**stats, "saved": round(saved, 6), "saving": round(saved / stats["frontier_cost"], 4) if stats["frontier_cost"] else 0.0}

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        body = await request.json()
        prompt = " ".join(str(m.get("content", "")) for m in body.get("messages", []) if m.get("role") == "user")
        task = request.headers.get("x-task", task_names[0])
        model, why = router.route(np.asarray([featurize(prompt, task)]))[0]
        url = os.environ.get(f"AUTOPILOT_{env_name(model)}_URL", "")
        key = os.environ.get(f"AUTOPILOT_{env_name(model)}_KEY", "")
        started = time.perf_counter()
        async with httpx.AsyncClient(base_url=url, transport=transport, timeout=120) as client:
            reply = await client.post("/chat/completions", json={**body, "model": model},
                                      headers={"Authorization": f"Bearer {key}"} if key else {})
        data = reply.json()
        usage = data.get("usage") or {}
        pin, pout = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        stats["requests"] += 1
        stats["cost"] += reg.cost(model, pin, pout)
        stats["frontier_cost"] += reg.cost(reg.frontier, pin, pout)
        stats["by_model"][model] = stats["by_model"].get(model, 0) + 1
        return JSONResponse(data, status_code=reply.status_code, headers={
            "X-Routed-Model": model, "X-Route-Reason": why, "X-Latency-Ms": str(round(1000 * (time.perf_counter() - started)))})

    return app
