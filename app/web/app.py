from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.engine import Engine

_here = Path(__file__).parent

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(_here / "static")), name="static")
templates = Jinja2Templates(directory=str(_here / "templates"))

_engine = Engine()


@app.get("/")
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


@app.post("/optimize", response_class=HTMLResponse)
def optimize(
    request: Request,
    prompt: Annotated[str, Form()],
    improve: Annotated[bool, Form()] = False,
) -> HTMLResponse:
    result = _engine.analyze(prompt, improve=improve)
    return templates.TemplateResponse(
        request,
        "partials/result.html",
        {"prompt": prompt, "result": result},
    )
