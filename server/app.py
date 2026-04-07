"""FastAPI server for DataForge-Env.

Exposes:
  POST /reset  — start a new episode
  POST /step   — apply an action
  GET  /state  — inspect current env state
  GET  /health — liveness probe
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from env.env import DataForgeEnv
from env.models import (
    Action,
    ResetRequest,
    ResetResponse,
    StepRequest,
    StepResult,
)

app = FastAPI(
    title="DataForge-Env",
    description="OpenEnv data-cleaning environment API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton environment instance
_env = DataForgeEnv()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/reset", response_model=ResetResponse)
async def reset(req: ResetRequest | None = None):
    """Reset the environment to a fresh episode."""
    task_id = req.task_id if req else "easy"
    try:
        obs = await _env.reset(task_id=task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return ResetResponse(
        observation=obs,
        info={"task_id": task_id, "message": "Environment reset successfully."},
    )


@app.post("/step", response_model=StepResult)
async def step(req: StepRequest):
    """Apply an action and return observation, reward, done, info."""
    try:
        result = await _env.step(req.action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@app.get("/state")
async def state():
    """Return a snapshot of the environment state (debugging / observability)."""
    return _env.state()


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
