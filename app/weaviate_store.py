from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from uuid import UUID

import weaviate
import weaviate.classes as wvc
from weaviate.auth import Auth
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter, MetadataQuery

from app.config import get_settings
from app.embeddings import embed_query, embed_texts
from app.ingestion import split_text
from app.models import DocumentInput, SearchHit


def deterministic_uuid(hex_digest: str) -> UUID:
    return UUID(hex=hex_digest[:32])


@contextmanager
def get_client() -> Iterator[weaviate.WeaviateClient]:
    settings = get_settings()
    if settings.weaviate_api_key:
        client = weaviate.connect_to_custom(
                http_host=settings.weaviate_http_host,
                http_port=settings.weaviate_http_port,
                http_secure=False,
                grpc_host=settings.weaviate_grpc_host,
                grpc_port=settings.weaviate_grpc_port,
                grpc_secure=False,
                auth_credentials=Auth.api_key(settings.weaviate_api_key),
        )
    else:
        client = weaviate.connect_to_custom(
            http_host=settings.weaviate_http_host,
            http_port=settings.weaviate_http_port,
            http_secure=False,
            grpc_host=settings.weaviate_grpc_host,
            grpc_port=settings.weaviate_grpc_port,
            grpc_secure=False,
        )
        # client = weaviate.connect_to_custom(
        #     http_host=settings.weaviate_http_host,
        #     http_port=settings.weaviate_http_port,
        #     http_secure=settings.weaviate_secure,
        #     grpc_host=settings.weaviate_grpc_host,
        #     grpc_port=settings.weaviate_grpc_port,
        #     grpc_secure=settings.weaviate_secure,
        #     auth_credentials=Auth.api_key(settings.weaviate_api_key),
        # )
    try:
        yield client
    finally:
        client.close()


def initialize_collection() -> None:
    settings = get_settings()
    with get_client() as client:
        if client.collections.exists(settings.weaviate_collection):
            return

        client.collections.create(
            name=settings.weaviate_collection,
            vector_config=Configure.Vectors.self_provided(
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=wvc.config.VectorDistances.COSINE
                )
            ),
            properties=[
                Property(name="chunk_id", data_type=DataType.TEXT),
                Property(name="doc_id", data_type=DataType.TEXT),
                Property(name="title", data_type=DataType.TEXT),
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="department", data_type=DataType.TEXT),
                Property(name="document_type", data_type=DataType.TEXT),
                Property(name="security_level", data_type=DataType.TEXT),
                Property(name="version", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
                Property(name="ingested_at", data_type=DataType.DATE),
            ],
        )


def upsert_document(document: DocumentInput) -> int:
    settings = get_settings()
    chunks = split_text(
        document.text,
        document.doc_id,
        settings.chunk_size,
        settings.chunk_overlap,
    )
    vectors = embed_texts([chunk.content for chunk in chunks])
    now = datetime.now(timezone.utc)

    with get_client() as client:
        collection = client.collections.use(settings.weaviate_collection)

        # Replace the document atomically from the application's perspective:
        # remove old chunks, then insert the new deterministic chunk set.
        collection.data.delete_many(
            where=Filter.by_property("doc_id").equal(document.doc_id)
        )

        with collection.batch.fixed_size(batch_size=100) as batch:
            for chunk, vector in zip(chunks, vectors, strict=True):
                batch.add_object(
                    uuid=deterministic_uuid(chunk.chunk_id),
                    properties={
                        "chunk_id": chunk.chunk_id,
                        "doc_id": document.doc_id,
                        "title": document.title,
                        "content": chunk.content,
                        "source": document.source,
                        "department": document.department,
                        "document_type": document.document_type,
                        "security_level": document.security_level,
                        "version": document.version,
                        "chunk_index": chunk.chunk_index,
                        "ingested_at": now,
                    },
                    vector=vector,
                )

        if collection.batch.failed_objects:
            errors = "; ".join(
                str(item.message) for item in collection.batch.failed_objects[:5]
            )
            raise RuntimeError(f"Weaviate batch insert failed: {errors}")

    return len(chunks)


def semantic_search(
    query: str,
    limit: int,
    department: str | None = None,
) -> list[SearchHit]:
    settings = get_settings()
    vector = embed_query(query)
    filters = (
        Filter.by_property("department").equal(department)
        if department
        else None
    )

    with get_client() as client:
        collection = client.collections.use(settings.weaviate_collection)
        response = collection.query.near_vector(
            near_vector=vector,
            limit=limit,
            filters=filters,
            return_metadata=MetadataQuery(distance=True),
        )

    hits: list[SearchHit] = []
    for obj in response.objects:
        props = obj.properties
        hits.append(
            SearchHit(
                chunk_id=str(props["chunk_id"]),
                doc_id=str(props["doc_id"]),
                title=str(props["title"]),
                content=str(props["content"]),
                source=str(props["source"]),
                department=str(props["department"]),
                chunk_index=int(props["chunk_index"]),
                distance=(
                    float(obj.metadata.distance)
                    if obj.metadata and obj.metadata.distance is not None
                    else None
                ),
            )
        )
    return hits


def health() -> dict[str, bool]:
    with get_client() as client:
        return {
            "live": client.is_live(),
            "ready": client.is_ready(),
        }
