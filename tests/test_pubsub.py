import json
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from app.models import DocumentInput
from app.pubsub_worker import callback, main, request_stop, stop_event


class TestPubSubCallback:
    """Unit tests for Pub/Sub message processing."""

    @staticmethod
    def create_message(payload: dict) -> MagicMock:
        message = MagicMock()
        message.data = json.dumps(payload).encode("utf-8")
        return message

    @staticmethod
    def valid_document_payload() -> dict:
        return {
            "doc_id": "policy-001",
            "title": "Renewal Policy",
            "text": "Renewal planning should begin 120 days early.",
            "source": "pubsub",
            "department": "Customer Success",
            "document_type": "policy",
            "security_level": "internal",
            "version": "1",
        }

    @patch("app.pubsub_worker.upsert_document")
    def test_callback_ingests_valid_message_and_acknowledges(
        self,
        mock_upsert_document,
    ) -> None:
        message = self.create_message(self.valid_document_payload())
        mock_upsert_document.return_value = 3

        callback(message)

        mock_upsert_document.assert_called_once()

        document = mock_upsert_document.call_args.args[0]
        assert isinstance(document, DocumentInput)
        assert document.doc_id == "policy-001"
        assert document.title == "Renewal Policy"
        assert document.department == "Customer Success"

        message.ack.assert_called_once_with()
        message.nack.assert_not_called()

    @patch("app.pubsub_worker.upsert_document")
    def test_callback_acknowledges_invalid_json(
        self,
        mock_upsert_document,
    ) -> None:
        message = MagicMock()
        message.data = b'{"doc_id": "broken"'

        callback(message)

        mock_upsert_document.assert_not_called()
        message.ack.assert_called_once_with()
        message.nack.assert_not_called()

    @patch("app.pubsub_worker.upsert_document")
    def test_callback_acknowledges_invalid_document_payload(
        self,
        mock_upsert_document,
    ) -> None:
        message = self.create_message(
            {
                "title": "Missing required fields",
            }
        )

        callback(message)

        mock_upsert_document.assert_not_called()
        message.ack.assert_called_once_with()
        message.nack.assert_not_called()

    @patch("app.pubsub_worker.upsert_document")
    def test_callback_nacks_transient_ingestion_failure(
        self,
        mock_upsert_document,
    ) -> None:
        message = self.create_message(self.valid_document_payload())
        mock_upsert_document.side_effect = RuntimeError(
            "Weaviate unavailable"
        )

        callback(message)

        mock_upsert_document.assert_called_once()
        message.nack.assert_called_once_with()
        message.ack.assert_not_called()

    @patch("app.pubsub_worker.logger")
    @patch("app.pubsub_worker.upsert_document")
    def test_callback_logs_successful_ingestion(
        self,
        mock_upsert_document,
        mock_logger,
    ) -> None:
        message = self.create_message(self.valid_document_payload())
        mock_upsert_document.return_value = 4

        callback(message)

        mock_logger.info.assert_called_once_with(
            "Ingested doc_id=%s chunks=%s",
            "policy-001",
            4,
        )

    @patch("app.pubsub_worker.logger")
    @patch("app.pubsub_worker.upsert_document")
    def test_callback_logs_invalid_message(
        self,
        mock_upsert_document,
        mock_logger,
    ) -> None:
        message = MagicMock()
        message.data = b"invalid-json"

        callback(message)

        mock_upsert_document.assert_not_called()
        mock_logger.error.assert_called_once()

        log_message = mock_logger.error.call_args.args[0]
        assert log_message == "Permanent invalid message: %s"

    @patch("app.pubsub_worker.logger")
    @patch("app.pubsub_worker.upsert_document")
    def test_callback_logs_transient_failure(
        self,
        mock_upsert_document,
        mock_logger,
    ) -> None:
        message = self.create_message(self.valid_document_payload())
        mock_upsert_document.side_effect = RuntimeError(
            "Temporary failure"
        )

        callback(message)

        mock_logger.exception.assert_called_once_with(
            "Transient ingestion failure; message will be retried"
        )


class TestRequestStop:
    """Tests for graceful shutdown signaling."""

    def setup_method(self) -> None:
        stop_event.clear()

    def teardown_method(self) -> None:
        stop_event.clear()

    def test_request_stop_sets_stop_event(self) -> None:
        assert stop_event.is_set() is False

        request_stop()

        assert stop_event.is_set() is True

    def test_request_stop_accepts_signal_arguments(self) -> None:
        request_stop(15, object())

        assert stop_event.is_set() is True


class TestMain:
    """Unit tests for Pub/Sub worker startup and shutdown."""

    def setup_method(self) -> None:
        stop_event.clear()

    def teardown_method(self) -> None:
        stop_event.clear()

    @patch("app.pubsub_worker.get_settings")
    def test_main_raises_when_gcp_project_is_missing(
        self,
        mock_get_settings,
    ) -> None:
        mock_get_settings.return_value = SimpleNamespace(
            gcp_project_id=None,
            pubsub_subscription="document-ingestion-sub",
        )

        with pytest.raises(
            RuntimeError,
            match="GCP_PROJECT_ID must be configured",
        ):
            main()

    @patch("app.pubsub_worker.pubsub_v1.SubscriberClient")
    @patch("app.pubsub_worker.initialize_collection")
    @patch("app.pubsub_worker.get_settings")
    def test_main_initializes_collection_and_subscription(
        self,
        mock_get_settings,
        mock_initialize_collection,
        mock_subscriber_class,
    ) -> None:
        mock_get_settings.return_value = SimpleNamespace(
            gcp_project_id="test-project",
            pubsub_subscription="document-ingestion-sub",
        )

        subscriber = MagicMock()
        streaming_pull = MagicMock()

        subscriber.subscription_path.return_value = (
            "projects/test-project/subscriptions/"
            "document-ingestion-sub"
        )
        subscriber.subscribe.return_value = streaming_pull
        streaming_pull.done.return_value = False

        mock_subscriber_class.return_value = subscriber

        with patch(
            "app.pubsub_worker.stop_event.wait",
            return_value=True,
        ):
            main()

        mock_initialize_collection.assert_called_once_with()
        mock_subscriber_class.assert_called_once_with()

        subscriber.subscription_path.assert_called_once_with(
            "test-project",
            "document-ingestion-sub",
        )

        subscriber.subscribe.assert_called_once()

        subscribe_arguments = subscriber.subscribe.call_args

        assert subscribe_arguments.args[0] == (
            "projects/test-project/subscriptions/"
            "document-ingestion-sub"
        )
        assert subscribe_arguments.kwargs["callback"] is callback
        assert (
            subscribe_arguments.kwargs["flow_control"].max_messages
            == 20
        )

        streaming_pull.cancel.assert_called_once_with()
        streaming_pull.result.assert_called_once_with(timeout=30)
        subscriber.close.assert_called_once_with()

    @patch("app.pubsub_worker.signal.signal")
    @patch("app.pubsub_worker.pubsub_v1.SubscriberClient")
    @patch("app.pubsub_worker.initialize_collection")
    @patch("app.pubsub_worker.get_settings")
    def test_main_registers_shutdown_signals(
        self,
        mock_get_settings,
        mock_initialize_collection,
        mock_subscriber_class,
        mock_signal,
    ) -> None:
        mock_get_settings.return_value = SimpleNamespace(
            gcp_project_id="test-project",
            pubsub_subscription="document-ingestion-sub",
        )

        subscriber = MagicMock()
        streaming_pull = MagicMock()

        subscriber.subscription_path.return_value = (
            "projects/test-project/subscriptions/"
            "document-ingestion-sub"
        )
        subscriber.subscribe.return_value = streaming_pull

        mock_subscriber_class.return_value = subscriber

        with patch(
            "app.pubsub_worker.stop_event.wait",
            return_value=True,
        ):
            main()

        assert mock_signal.call_count == 2

        
        assert call(
            __import__("signal").SIGINT,
            request_stop,
        ) in mock_signal.call_args_list

    @patch("app.pubsub_worker.pubsub_v1.SubscriberClient")
    @patch("app.pubsub_worker.initialize_collection")
    @patch("app.pubsub_worker.get_settings")
    def test_main_checks_streaming_result_when_future_is_done(
        self,
        mock_get_settings,
        mock_initialize_collection,
        mock_subscriber_class,
    ) -> None:
        mock_get_settings.return_value = SimpleNamespace(
            gcp_project_id="test-project",
            pubsub_subscription="document-ingestion-sub",
        )

        subscriber = MagicMock()
        streaming_pull = MagicMock()

        subscriber.subscription_path.return_value = (
            "projects/test-project/subscriptions/"
            "document-ingestion-sub"
        )
        subscriber.subscribe.return_value = streaming_pull
        streaming_pull.done.return_value = True

        mock_subscriber_class.return_value = subscriber

        wait_results = iter([False, True])

        with patch(
            "app.pubsub_worker.stop_event.wait",
            side_effect=lambda timeout: next(wait_results),
        ):
            main()

        assert streaming_pull.done.call_count == 1

        assert streaming_pull.result.call_args_list == [
            call(),
            call(timeout=30),
        ]

    @patch("app.pubsub_worker.pubsub_v1.SubscriberClient")
    @patch("app.pubsub_worker.initialize_collection")
    @patch("app.pubsub_worker.get_settings")
    def test_main_handles_streaming_timeout(
        self,
        mock_get_settings,
        mock_initialize_collection,
        mock_subscriber_class,
    ) -> None:
        from concurrent.futures import TimeoutError

        mock_get_settings.return_value = SimpleNamespace(
            gcp_project_id="test-project",
            pubsub_subscription="document-ingestion-sub",
        )

        subscriber = MagicMock()
        streaming_pull = MagicMock()

        subscriber.subscription_path.return_value = (
            "projects/test-project/subscriptions/"
            "document-ingestion-sub"
        )
        subscriber.subscribe.return_value = streaming_pull
        streaming_pull.done.return_value = True

        streaming_pull.result.side_effect = [
            TimeoutError(),
            None,
        ]

        mock_subscriber_class.return_value = subscriber

        with patch(
            "app.pubsub_worker.stop_event.wait",
            return_value=False,
        ):
            main()

        streaming_pull.cancel.assert_called_once_with()
        subscriber.close.assert_called_once_with()

    @patch("app.pubsub_worker.pubsub_v1.SubscriberClient")
    @patch("app.pubsub_worker.initialize_collection")
    @patch("app.pubsub_worker.get_settings")
    def test_main_always_closes_resources(
        self,
        mock_get_settings,
        mock_initialize_collection,
        mock_subscriber_class,
    ) -> None:
        mock_get_settings.return_value = SimpleNamespace(
            gcp_project_id="test-project",
            pubsub_subscription="document-ingestion-sub",
        )

        subscriber = MagicMock()
        streaming_pull = MagicMock()

        subscriber.subscription_path.return_value = (
            "projects/test-project/subscriptions/"
            "document-ingestion-sub"
        )
        subscriber.subscribe.return_value = streaming_pull
        streaming_pull.done.side_effect = RuntimeError(
            "Streaming failure"
        )

        mock_subscriber_class.return_value = subscriber

        with patch(
            "app.pubsub_worker.stop_event.wait",
            return_value=False,
        ):
            with pytest.raises(
                RuntimeError,
                match="Streaming failure",
            ):
                main()

        streaming_pull.cancel.assert_called_once_with()
        streaming_pull.result.assert_called_once_with(timeout=30)
        subscriber.close.assert_called_once_with() 
