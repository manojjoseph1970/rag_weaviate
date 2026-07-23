from pydantic import BaseModel, Field


class DocumentInput(BaseModel):
    doc_id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=1)
    source: str = Field(default="api", max_length=1000)
    department: str = Field(default="General", max_length=200)
    document_type: str = Field(default="text", max_length=100)
    security_level: str = Field(default="internal", max_length=100)
    version: str = Field(default="1", max_length=100)


class IngestResult(BaseModel):
    doc_id: str
    chunks_written: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=20)
    department: str | None = None


class SearchHit(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    content: str
    source: str
    department: str
    chunk_index: int
    distance: float | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=20)
    department: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SearchHit]
    generation_enabled: bool