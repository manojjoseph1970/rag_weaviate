from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.api import app
from app.models import AskResponse,  SearchHit


@pytest.fixture

def client():
    """
    Disable the real Weaviate initialization that normally runs during
    the FastAPI lifespan startup.
    """
    with patch("app.api.initialize_collection") as mock_initialize:
        with TestClient(app) as test_client:
            yield test_client

        mock_initialize.assert_called_once()


# -------------------------------------------------------------------
# Health endpoint
# -------------------------------------------------------------------

def test_liveness_endpoint(client) -> None:
    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}

@patch("app.api.health")
def test_health_endpoint(mock_health,client) -> None:
    mock_health.return_value = {
        "ready": True,
        "message": "Weaviate is ready",
    }

    response = client.get("/health")

    assert response.status_code == 200

    mock_health.assert_called_once_with()


@patch("app.api.health")
def test_health_endpoint_not_ready(mock_health, client):
    mock_health.return_value = {
        "ready": False,
        "message": "Weaviate is unavailable",
    }

    response = client.get("/health")

    assert response.status_code == 503

    # Because health_check catches HTTPException as a generic Exception,
    # the detail may be converted to a string.
    body = response.json()

    assert body["detail"]["ready"] is False
    assert body["detail"]["message"] == "Weaviate is unavailable"

    mock_health.assert_called_once_with()


@patch("app.api.health")
def test_health_endpoint_exception(mock_health, client):
    mock_health.side_effect = RuntimeError("Connection failed")

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"detail": "Connection failed"}


# -------------------------------------------------------------------
# Admin initialization endpoint
# -------------------------------------------------------------------


@patch("app.api.initialize_collection")
def test_initialize_endpoint(mock_initialize, client):
    response = client.post("/admin/initialize")

    assert response.status_code == 200
    assert response.json() == {"status": "initialized"}

    mock_initialize.assert_called_once_with()


# -------------------------------------------------------------------
# Document ingestion endpoint
# -------------------------------------------------------------------


@patch("app.api.upsert_document")
def test_ingest_document(mock_upsert, client):
    mock_upsert.return_value = 3

    payload = {
        "doc_id": "policy-001",
        "title": "Customer Renewal Policy",
        "text": "This is a sample renewal policy document.",
        "source": "unit-test",
        "department": "Customer Success",
        "document_type": "policy",
        "security_level": "internal",
        "version": "1",
    }

    response = client.post("/documents", json=payload)

    assert response.status_code == 200
    assert response.json() == {
        "doc_id": "policy-001",
        "chunks_written": 3,
    }

    mock_upsert.assert_called_once()

    document = mock_upsert.call_args.args[0]
    assert document.doc_id == "policy-001"
    assert document.title == "Customer Renewal Policy"
    assert document.text == "This is a sample renewal policy document."


@patch("app.api.upsert_document")
def test_ingest_document_failure(mock_upsert, client):
    mock_upsert.side_effect = RuntimeError("Weaviate insert failed")

    payload = {
        "doc_id": "policy-001",
        "title": "Customer Renewal Policy",
        "text": "Sample document",
        "source": "unit-test",
        "department": "Customer Success",
        "document_type": "policy",
        "security_level": "internal",
        "version": "1",
    }

    response = client.post("/documents", json=payload)

    assert response.status_code == 500
    assert response.json() == {"detail": "Weaviate insert failed"}


def test_ingest_document_validation_error(client):
    payload = {
        "title": "Missing document ID",
        "text": "Invalid request",
    }

    response = client.post("/documents", json=payload)

    assert response.status_code == 422


# -------------------------------------------------------------------
# File upload endpoint
# -------------------------------------------------------------------


@patch("app.api.upsert_document")
def test_upload_text_document(mock_upsert, client):
    mock_upsert.return_value = 2

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "renewal_policy.txt",
                b"Renewal policy text",
                "text/plain",
            )
        },
        data={
            "department": "Customer Success",
            "security_level": "internal",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "doc_id": "renewal_policy.txt",
        "chunks_written": 2,
    }

    mock_upsert.assert_called_once()

    document = mock_upsert.call_args.args[0]

    assert document.doc_id == "renewal_policy.txt"
    assert document.title == "renewal_policy"
    assert document.text == "Renewal policy text"
    assert document.source == "renewal_policy.txt"
    assert document.department == "Customer Success"
    assert document.document_type == "txt"
    assert document.security_level == "internal"


@patch("app.api.upsert_document")
def test_upload_markdown_document(mock_upsert, client):
    mock_upsert.return_value = 1

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "guide.md",
                b"# Customer Guide",
                "text/markdown",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["doc_id"] == "guide.md"
    assert response.json()["chunks_written"] == 1

    document = mock_upsert.call_args.args[0]
    assert document.department == "General"
    assert document.security_level == "internal"
    assert document.document_type == "md"


def test_upload_unsupported_file_type(client):
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "document.pdf",
                b"fake PDF content",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 415
    assert response.json() == {
        "detail": "Supported upload types: .txt, .md, .csv, .json"
    }


def test_upload_non_utf8_file(client):
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "invalid.txt",
                b"\xff\xfe\xfd",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "The uploaded file must be UTF-8 text."
    }


def test_upload_without_file(client):
    response = client.post(
        "/documents/upload",
        data={
            "department": "Customer Success",
            "security_level": "internal",
        },
    )

    assert response.status_code == 422


# -------------------------------------------------------------------
# Search endpoint
# -------------------------------------------------------------------


@patch("app.api.semantic_search")
def test_search(mock_semantic_search, client):
    mock_semantic_search.return_value = [
    SearchHit(
        doc_id="policy-001",
        chunk_id="chunk-001",
        title="Renewal Policy",
        content="Customers must submit renewal requests.",
        chunk_index=0,
        #score=0.95,
        source="policy.txt",
        department="Customer Success",
    )
    ]   

    payload = {
        "query": "renewal policy",
        "limit": 5,
        "department": "Customer Success",
    }

    response = client.post("/search", json=payload)

    assert response.status_code == 200

    body = response.json()
    assert len(body) == 1
    assert body[0]["doc_id"] == "policy-001"
    

    mock_semantic_search.assert_called_once_with(
        "renewal policy",
        5,
        "Customer Success",
    )


@patch("app.api.semantic_search")
def test_search_empty_results(mock_semantic_search, client):
    mock_semantic_search.return_value = []

    response = client.post(
        "/search",
        json={
            "query": "unknown policy",
            "limit": 5,
            "department": None,
        },
    )

    assert response.status_code == 200
    assert response.json() == []

    mock_semantic_search.assert_called_once_with(
        "unknown policy",
        5,
        None,
    )


@patch("app.api.semantic_search")
def test_search_failure(mock_semantic_search, client):
    mock_semantic_search.side_effect = RuntimeError("Search failed")

    response = client.post(
        "/search",
        json={
            "query": "renewal policy",
            "limit": 5,
            "department": "Customer Success",
        },
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Search failed"}


# -------------------------------------------------------------------
# Ask endpoint
# -------------------------------------------------------------------


@patch("app.api.answer_question")
def test_ask_question(mock_answer_question, client):
    mock_answer_question.return_value = AskResponse(
        answer="The renewal period is 30 days.",
        sources=[],
         generation_enabled=True,
    )

    payload = {
        "question": "What is the renewal period?",
        "limit": 5,
        "department": "Customer Success",
    }

    response = client.post("/ask", json=payload)

    assert response.status_code == 200
    assert response.json()["answer"] == "The renewal period is 30 days."

    mock_answer_question.assert_called_once_with(
        "What is the renewal period?",
        5,
        "Customer Success",
    )


@patch("app.api.answer_question")
def test_ask_question_failure(mock_answer_question, client):
    mock_answer_question.side_effect = RuntimeError("LLM request failed")

    response = client.post(
        "/ask",
        json={
            "question": "What is the renewal period?",
            "limit": 5,
            "department": "Customer Success",
        },
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "LLM request failed"}