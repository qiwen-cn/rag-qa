from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag_core import ask

app = FastAPI()


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask_endpoint(req: AskRequest):
    return ask(req.question)


@app.get("/")
def index():
    return FileResponse("rag.html")
