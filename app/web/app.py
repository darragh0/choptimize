from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, cast  # noqa: TC003

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.engine import Engine
from app.engine.llm import list_models
from app.engine.types import LLMService

_here = Path(__file__).parent
_VALID_SERVICES: frozenset[str] = frozenset({"ollama", "openai", "gemini"})
_engines: dict[tuple[LLMService, str | None, str | None], Engine] = {}


def _get_engine(
    service: LLMService = "ollama",
    model: str | None = None,
    api_key: str | None = None,
) -> Engine:
    key = (service, model, api_key)
    if key not in _engines:
        _engines[key] = Engine(service=service, model=model, api_key=api_key)
    return _engines[key]


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    try:
        _get_engine()
    except Exception:
        pass
    yield


app = FastAPI(lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=str(_here / "static")), name="static")
templates = Jinja2Templates(directory=str(_here / "templates"))


@app.get("/")
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


@app.get("/models")
def models(service: str = "ollama", api_key: str = "") -> dict[str, list[str] | str]:
    svc: LLMService = cast("LLMService", service) if service in _VALID_SERVICES else "ollama"
    try:
        return {"models": list_models(svc, api_key or None)}
    except ValueError as e:
        return {"error": str(e)}


@app.post("/optimize", response_class=HTMLResponse)
def optimize(
    request: Request,
    prompt: Annotated[str, Form()],
    *,
    service: Annotated[str, Form()] = "ollama",
    model: Annotated[str, Form()] = "",
    api_key: Annotated[str, Form()] = "",
    improve: Annotated[bool, Form()] = False,
) -> HTMLResponse:
    from app.engine.validator import validate  # noqa: PLC0415

    if msg := validate(prompt):
        return templates.TemplateResponse(request, "partials/error.html", {"message": msg})

    svc: LLMService = cast("LLMService", service) if service in _VALID_SERVICES else "ollama"
    try:
        engine = _get_engine(service=svc, model=model or None, api_key=api_key or None)
        result = engine.analyze(prompt, improve=improve, show_raw=False)
    except ValueError as e:
        return templates.TemplateResponse(request, "partials/error.html", {"message": str(e)})
    return templates.TemplateResponse(
        request,
        "partials/result.html",
        {"prompt": prompt, "result": result},
    )
