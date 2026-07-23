from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.config import get_settings
from app.models import (
    AskRequest,
    AskResponse,
    DocumentInput,
    IngestResult,
    SearchHit,
    SearchRequest,
)
from app.rag import answer_question
from app.weaviate_store import (
    health,
    initialize_collection,
    semantic_search,
    upsert_document,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_collection()
    yield


settings = get_settings()
app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)


@app.get("/health/live")
def liveness_check() -> dict[str, str]:
    return {"status": "alive"}

@app.get("/health")
def health_check() -> dict:
    try:
        status = health()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    if not status["ready"]:
        raise HTTPException(
            status_code=503,
            detail=status,
        )

    return {
        "status": "ok",
        "weaviate": status,
    }


@app.post("/admin/initialize")
def initialize() -> dict[str, str]:
    initialize_collection()
    return {"status": "initialized"}


@app.post("/documents", response_model =IngestResult)
def ingest_document(document: DocumentInput) -> IngestResult:
    try:
        count = upsert_document(document)
        return IngestResult(doc_id=document.doc_id, chunks_written=count)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/documents/upload", response_model=IngestResult)
async def upload_document(
    file: UploadFile = File(...),
    department: str = Form(default="General"),
    security_level: str = Form(default="internal"),
) -> IngestResult:
    filename = file.filename or "uploaded.txt"
    suffix = Path(filename).suffix.lower()
    if suffix not in {".txt", ".md", ".csv", ".json"}:
        raise HTTPException(
            status_code=415,
            detail="Supported upload types: .txt, .md, .csv, .json",
        )

    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file must be UTF-8 text.",
        ) from exc

    document = DocumentInput(
        doc_id=filename,
        title=Path(filename).stem,
        text=text,
        source=filename,
        department=department,
        document_type=suffix.lstrip(".") or "text",
        security_level=security_level,
    )
    count = upsert_document(document)
    return IngestResult(doc_id=document.doc_id, chunks_written=count)


@app.post("/search", response_model=list[SearchHit])
def search(request: SearchRequest) -> list[SearchHit]:
    try:
        return semantic_search(
            request.query,
            request.limit,
            request.department,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        return answer_question(
            request.question,
            request.limit,
            request.department,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
