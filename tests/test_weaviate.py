
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from app.models import DocumentInput, SearchHit
from app.weaviate_store import (
    deterministic_uuid,
    get_client,
    health,
    initialize_collection,
    semantic_search,
    upsert_document,
)


def create_settings(
    *,
    api_key: str | None = None,
) -> SimpleNamespace:
    """Create fake application settings for unit tests."""
    return SimpleNamespace(
        weaviate_api_key=api_key,
        weaviate_http_host="weaviate",
        weaviate_http_port=8080,
        weaviate_grpc_host="weaviate",
        weaviate_grpc_port=50051,
        weaviate_collection="DocumentChunk",
        chunk_size=500,
        chunk_overlap=50,
    )


def create_document() -> DocumentInput:
    """Create a valid test document."""
    return DocumentInput(
        doc_id="policy-001",
        title="Renewal Policy",
        text=(
            "Enterprise renewal planning should begin "
            "120 days before contract expiration."
        ),
        source="pubsub",
        department="Customer Success",
        document_type="policy",
        security_level="internal",
        version="1",
    )


class TestDeterministicUuid:
    def test_returns_uuid_from_first_32_hex_characters(self) -> None:
        digest = (
            "1234567890abcdef1234567890abcdef"
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        )

        result = deterministic_uuid(digest)

        assert isinstance(result, UUID)
        assert result.hex == "1234567890abcdef1234567890abcdef"

    def test_same_digest_returns_same_uuid(self) -> None:
        digest = "a" * 64

        first = deterministic_uuid(digest)
        second = deterministic_uuid(digest)

        assert first == second

    def test_different_digests_return_different_uuids(self) -> None:
        first = deterministic_uuid("a" * 64)
        second = deterministic_uuid("b" * 64)

        assert first != second

    def test_invalid_hex_digest_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            deterministic_uuid("not-a-valid-hex-digest")


class TestGetClient:
    @patch("app.weaviate_store.weaviate.connect_to_custom")
    @patch("app.weaviate_store.get_settings")
    def test_connects_without_api_key(
        self,
        mock_get_settings,
        mock_connect,
    ) -> None:
        mock_get_settings.return_value = create_settings(api_key=None)
        client = MagicMock()
        mock_connect.return_value = client

        with get_client() as returned_client:
            assert returned_client is client

        mock_connect.assert_called_once_with(
            http_host="weaviate",
            http_port=8080,
            http_secure=False,
            grpc_host="weaviate",
            grpc_port=50051,
            grpc_secure=False,
        )
        client.close.assert_called_once_with()

    @patch("app.weaviate_store.Auth.api_key")
    @patch("app.weaviate_store.weaviate.connect_to_custom")
    @patch("app.weaviate_store.get_settings")
    def test_connects_with_api_key(
        self,
        mock_get_settings,
        mock_connect,
        mock_auth_api_key,
    ) -> None:
        mock_get_settings.return_value = create_settings(
            api_key="test-weaviate-key"
        )

        credentials = MagicMock()
        mock_auth_api_key.return_value = credentials

        client = MagicMock()
        mock_connect.return_value = client

        with get_client() as returned_client:
            assert returned_client is client

        mock_auth_api_key.assert_called_once_with("test-weaviate-key")

        mock_connect.assert_called_once_with(
            http_host="weaviate",
            http_port=8080,
            http_secure=False,
            grpc_host="weaviate",
            grpc_port=50051,
            grpc_secure=False,
            auth_credentials=credentials,
        )

        client.close.assert_called_once_with()

    @patch("app.weaviate_store.weaviate.connect_to_custom")
    @patch("app.weaviate_store.get_settings")
    def test_closes_client_when_context_body_raises(
        self,
        mock_get_settings,
        mock_connect,
    ) -> None:
        mock_get_settings.return_value = create_settings()
        client = MagicMock()
        mock_connect.return_value = client

        with pytest.raises(RuntimeError, match="Test failure"):
            with get_client():
                raise RuntimeError("Test failure")

        client.close.assert_called_once_with()


class TestInitializeCollection:
    @patch("app.weaviate_store.get_client")
    @patch("app.weaviate_store.get_settings")
    def test_does_not_create_existing_collection(
        self,
        mock_get_settings,
        mock_get_client,
    ) -> None:
        mock_get_settings.return_value = create_settings()

        client = MagicMock()
        client.collections.exists.return_value = True
        mock_get_client.return_value.__enter__.return_value = client

        initialize_collection()

        client.collections.exists.assert_called_once_with(
            "DocumentChunk"
        )
        client.collections.create.assert_not_called()

    @patch("app.weaviate_store.Property")
    @patch("app.weaviate_store.Configure.VectorIndex.hnsw")
    @patch("app.weaviate_store.Configure.Vectors.self_provided")
    @patch("app.weaviate_store.get_client")
    @patch("app.weaviate_store.get_settings")
    def test_creates_collection_when_missing(
        self,
        mock_get_settings,
        mock_get_client,
        mock_self_provided,
        mock_hnsw,
        mock_property,
    ) -> None:
        mock_get_settings.return_value = create_settings()

        client = MagicMock()
        client.collections.exists.return_value = False
        mock_get_client.return_value.__enter__.return_value = client

        hnsw_config = MagicMock()
        vector_config = MagicMock()

        mock_hnsw.return_value = hnsw_config
        mock_self_provided.return_value = vector_config
        mock_property.side_effect = (
            lambda **kwargs: kwargs
        )

        initialize_collection()

        client.collections.exists.assert_called_once_with(
            "DocumentChunk"
        )

        mock_hnsw.assert_called_once()

        mock_self_provided.assert_called_once_with(
            vector_index_config=hnsw_config
        )

        client.collections.create.assert_called_once()

        create_kwargs = client.collections.create.call_args.kwargs

        assert create_kwargs["name"] == "DocumentChunk"
        assert create_kwargs["vector_config"] is vector_config
        assert len(create_kwargs["properties"]) == 11

        property_names = [
            prop["name"]
            for prop in create_kwargs["properties"]
        ]

        assert property_names == [
            "chunk_id",
            "doc_id",
            "title",
            "content",
            "source",
            "department",
            "document_type",
            "security_level",
            "version",
            "chunk_index",
            "ingested_at",
        ]


class TestUpsertDocument:
    @patch("app.weaviate_store.datetime")
    @patch("app.weaviate_store.embed_texts")
    @patch("app.weaviate_store.split_text")
    @patch("app.weaviate_store.get_client")
    @patch("app.weaviate_store.get_settings")
    def test_upserts_document_chunks(
        self,
        mock_get_settings,
        mock_get_client,
        mock_split_text,
        mock_embed_texts,
        mock_datetime,
    ) -> None:
        mock_get_settings.return_value = create_settings()

        document = create_document()

        chunks = [
            SimpleNamespace(
                chunk_id="a" * 64,
                content="First chunk",
                chunk_index=0,
            ),
            SimpleNamespace(
                chunk_id="b" * 64,
                content="Second chunk",
                chunk_index=1,
            ),
        ]

        vectors = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]

        mock_split_text.return_value = chunks
        mock_embed_texts.return_value = vectors

        fixed_time = datetime(
            2026,
            7,
            22,
            12,
            0,
            0,
        )
        mock_datetime.now.return_value = fixed_time

        client = MagicMock()
        collection = MagicMock()
        batch = MagicMock()

        client.collections.use.return_value = collection
        collection.batch.fixed_size.return_value.__enter__.return_value = (
            batch
        )
        collection.batch.failed_objects = []

        mock_get_client.return_value.__enter__.return_value = client

        result = upsert_document(document)

        assert result == 2

        mock_split_text.assert_called_once_with(
            document.text,
            document.doc_id,
            500,
            50,
        )

        mock_embed_texts.assert_called_once_with(
            ["First chunk", "Second chunk"]
        )

        client.collections.use.assert_called_once_with(
            "DocumentChunk"
        )

        collection.data.delete_many.assert_called_once()

        delete_filter = (
            collection.data.delete_many.call_args.kwargs["where"]
        )
        assert delete_filter is not None

        collection.batch.fixed_size.assert_called_once_with(
            batch_size=100
        )

        assert batch.add_object.call_count == 2

        first_call = batch.add_object.call_args_list[0].kwargs
        second_call = batch.add_object.call_args_list[1].kwargs

        assert first_call["uuid"] == UUID(hex="a" * 32)
        assert first_call["vector"] == [0.1, 0.2, 0.3]
        assert first_call["properties"] == {
            "chunk_id": "a" * 64,
            "doc_id": "policy-001",
            "title": "Renewal Policy",
            "content": "First chunk",
            "source": "pubsub",
            "department": "Customer Success",
            "document_type": "policy",
            "security_level": "internal",
            "version": "1",
            "chunk_index": 0,
            "ingested_at": fixed_time,
        }

        assert second_call["uuid"] == UUID(hex="b" * 32)
        assert second_call["vector"] == [0.4, 0.5, 0.6]
        assert second_call["properties"]["content"] == "Second chunk"
        assert second_call["properties"]["chunk_index"] == 1

    @patch("app.weaviate_store.embed_texts")
    @patch("app.weaviate_store.split_text")
    @patch("app.weaviate_store.get_client")
    @patch("app.weaviate_store.get_settings")
    def test_raises_when_batch_contains_failed_objects(
        self,
        mock_get_settings,
        mock_get_client,
        mock_split_text,
        mock_embed_texts,
    ) -> None:
        mock_get_settings.return_value = create_settings()

        document = create_document()

        mock_split_text.return_value = [
            SimpleNamespace(
                chunk_id="a" * 64,
                content="Chunk content",
                chunk_index=0,
            )
        ]
        mock_embed_texts.return_value = [[0.1, 0.2]]

        client = MagicMock()
        collection = MagicMock()
        batch = MagicMock()

        client.collections.use.return_value = collection
        collection.batch.fixed_size.return_value.__enter__.return_value = (
            batch
        )

        collection.batch.failed_objects = [
            SimpleNamespace(message="First insert failed"),
            SimpleNamespace(message="Second insert failed"),
        ]

        mock_get_client.return_value.__enter__.return_value = client

        with pytest.raises(
            RuntimeError,
            match=(
                "Weaviate batch insert failed: "
                "First insert failed; Second insert failed"
            ),
        ):
            upsert_document(document)

    @patch("app.weaviate_store.embed_texts")
    @patch("app.weaviate_store.split_text")
    @patch("app.weaviate_store.get_client")
    @patch("app.weaviate_store.get_settings")
    def test_raises_when_chunk_and_vector_counts_do_not_match(
        self,
        mock_get_settings,
        mock_get_client,
        mock_split_text,
        mock_embed_texts,
    ) -> None:
        mock_get_settings.return_value = create_settings()

        mock_split_text.return_value = [
            SimpleNamespace(
                chunk_id="a" * 64,
                content="First chunk",
                chunk_index=0,
            ),
            SimpleNamespace(
                chunk_id="b" * 64,
                content="Second chunk",
                chunk_index=1,
            ),
        ]

        mock_embed_texts.return_value = [
            [0.1, 0.2],
        ]

        client = MagicMock()
        collection = MagicMock()
        batch = MagicMock()

        client.collections.use.return_value = collection
        collection.batch.fixed_size.return_value.__enter__.return_value = (
            batch
        )
        collection.batch.failed_objects = []

        mock_get_client.return_value.__enter__.return_value = client

        with pytest.raises(ValueError):
            upsert_document(create_document())


class TestSemanticSearch:
    @patch("app.weaviate_store.embed_query")
    @patch("app.weaviate_store.get_client")
    @patch("app.weaviate_store.get_settings")
    def test_searches_without_department_filter(
        self,
        mock_get_settings,
        mock_get_client,
        mock_embed_query,
    ) -> None:
        mock_get_settings.return_value = create_settings()
        mock_embed_query.return_value = [0.1, 0.2, 0.3]

        client = MagicMock()
        collection = MagicMock()

        client.collections.use.return_value = collection

        search_object = SimpleNamespace(
            properties={
                "chunk_id": "chunk-001",
                "doc_id": "policy-001",
                "title": "Renewal Policy",
                "content": "Begin renewal planning 120 days early.",
                "source": "policy.txt",
                "department": "Customer Success",
                "chunk_index": 0,
            },
            metadata=SimpleNamespace(distance=0.12),
        )

        collection.query.near_vector.return_value = (
            SimpleNamespace(objects=[search_object])
        )

        mock_get_client.return_value.__enter__.return_value = client

        results = semantic_search(
            query="When should renewal planning begin?",
            limit=5,
        )

        assert len(results) == 1
        assert isinstance(results[0], SearchHit)
        assert results[0].chunk_id == "chunk-001"
        assert results[0].doc_id == "policy-001"
        assert results[0].title == "Renewal Policy"
        assert results[0].distance == 0.12

        mock_embed_query.assert_called_once_with(
            "When should renewal planning begin?"
        )

        client.collections.use.assert_called_once_with(
            "DocumentChunk"
        )

        call_kwargs = (
            collection.query.near_vector.call_args.kwargs
        )

        assert call_kwargs["near_vector"] == [0.1, 0.2, 0.3]
        assert call_kwargs["limit"] == 5
        assert call_kwargs["filters"] is None

    @patch("app.weaviate_store.Filter.by_property")
    @patch("app.weaviate_store.embed_query")
    @patch("app.weaviate_store.get_client")
    @patch("app.weaviate_store.get_settings")
    def test_searches_with_department_filter(
        self,
        mock_get_settings,
        mock_get_client,
        mock_embed_query,
        mock_by_property,
    ) -> None:
        mock_get_settings.return_value = create_settings()
        mock_embed_query.return_value = [0.1, 0.2]

        property_filter = MagicMock()
        department_filter = MagicMock()

        mock_by_property.return_value = property_filter
        property_filter.equal.return_value = department_filter

        client = MagicMock()
        collection = MagicMock()

        client.collections.use.return_value = collection
        collection.query.near_vector.return_value = (
            SimpleNamespace(objects=[])
        )

        mock_get_client.return_value.__enter__.return_value = client

        results = semantic_search(
            query="What is the renewal policy?",
            limit=3,
            department="Customer Success",
        )

        assert results == []

        mock_by_property.assert_called_once_with("department")
        property_filter.equal.assert_called_once_with(
            "Customer Success"
        )

        call_kwargs = (
            collection.query.near_vector.call_args.kwargs
        )
        assert call_kwargs["filters"] is department_filter

    @patch("app.weaviate_store.embed_query")
    @patch("app.weaviate_store.get_client")
    @patch("app.weaviate_store.get_settings")
    def test_returns_none_when_distance_is_missing(
        self,
        mock_get_settings,
        mock_get_client,
        mock_embed_query,
    ) -> None:
        mock_get_settings.return_value = create_settings()
        mock_embed_query.return_value = [0.1, 0.2]

        client = MagicMock()
        collection = MagicMock()

        client.collections.use.return_value = collection

        search_object = SimpleNamespace(
            properties={
                "chunk_id": "chunk-001",
                "doc_id": "doc-001",
                "title": "Test Document",
                "content": "Test content",
                "source": "test.txt",
                "department": "Finance",
                "chunk_index": 0,
            },
            metadata=SimpleNamespace(distance=None),
        )

        collection.query.near_vector.return_value = (
            SimpleNamespace(objects=[search_object])
        )

        mock_get_client.return_value.__enter__.return_value = client

        results = semantic_search(
            query="test",
            limit=1,
        )

        assert results[0].distance is None

    @patch("app.weaviate_store.embed_query")
    @patch("app.weaviate_store.get_client")
    @patch("app.weaviate_store.get_settings")
    def test_returns_none_when_metadata_is_missing(
        self,
        mock_get_settings,
        mock_get_client,
        mock_embed_query,
    ) -> None:
        mock_get_settings.return_value = create_settings()
        mock_embed_query.return_value = [0.1, 0.2]

        client = MagicMock()
        collection = MagicMock()

        client.collections.use.return_value = collection

        search_object = SimpleNamespace(
            properties={
                "chunk_id": "chunk-001",
                "doc_id": "doc-001",
                "title": "Test Document",
                "content": "Test content",
                "source": "test.txt",
                "department": "Finance",
                "chunk_index": 0,
            },
            metadata=None,
        )

        collection.query.near_vector.return_value = (
            SimpleNamespace(objects=[search_object])
        )

        mock_get_client.return_value.__enter__.return_value = client

        results = semantic_search(
            query="test",
            limit=1,
        )

        assert results[0].distance is None


class TestHealth:
    @patch("app.weaviate_store.get_client")
    def test_returns_live_and_ready_status(
        self,
        mock_get_client,
    ) -> None:
        client = MagicMock()
        client.is_live.return_value = True
        client.is_ready.return_value = False

        mock_get_client.return_value.__enter__.return_value = client

        result = health()

        assert result == {
            "live": True,
            "ready": False,
        }

        client.is_live.assert_called_once_with()
        client.is_ready.assert_called_once_with() 