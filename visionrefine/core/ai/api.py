"""Shared image/video model settings and job routes."""
from fastapi import APIRouter

from ..video.api import errors
from .contracts import Bindings, Configuration, JobInput, Provider, ReviewInput
from .providers import check
from .service import AIService


def create_ai_router(get_store):
    router = APIRouter(prefix="/api/ai", tags=["ai"])

    @router.get("/config")
    def config():
        return AIService(get_store()).configuration()

    @router.put("/config")
    def save_config(payload: Configuration):
        with errors():
            return AIService(get_store()).save_configuration(payload)

    @router.post("/test")
    def test_provider(payload: Provider):
        with errors():
            return check(payload.model_dump())

    @router.get("/projects/{pid}/bindings")
    def bindings(pid: str):
        with errors():
            return AIService(get_store()).bindings(pid)

    @router.put("/projects/{pid}/bindings")
    def save_bindings(pid: str, payload: Bindings):
        with errors():
            return AIService(get_store()).save_bindings(pid, payload)

    @router.get("/projects/{pid}/jobs")
    def jobs(pid: str):
        with errors():
            return AIService(get_store()).jobs(pid)

    @router.post("/projects/{pid}/jobs", status_code=202)
    def start(pid: str, payload: JobInput):
        with errors():
            return AIService(get_store()).start(pid, payload)

    @router.get("/projects/{pid}/jobs/{jid}")
    def job(pid: str, jid: str):
        with errors():
            return AIService(get_store()).job(pid, jid)

    @router.post("/projects/{pid}/jobs/{jid}/cancel")
    def cancel(pid: str, jid: str):
        with errors():
            return AIService(get_store()).cancel(pid, jid)

    @router.post("/projects/{pid}/jobs/{jid}/retry", status_code=202)
    def retry(pid: str, jid: str):
        with errors():
            return AIService(get_store()).retry(pid, jid)

    @router.post("/projects/{pid}/jobs/{jid}/review")
    def review(pid: str, jid: str, payload: ReviewInput):
        with errors():
            return AIService(get_store()).review(pid, jid, payload)

    return router
