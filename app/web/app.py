from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/submit")
def submit(text: str = Form(...)):
    return {"you_sent": text}
