import json
import logging
import signal
import threading
from concurrent.futures import TimeoutError

from google.cloud import pubsub_v1
from pydantic import ValidationError

from app.config import get_settings
from app.models import DocumentInput
from app.weaviate_store import initialize_collection, upsert_document


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("pubsub-worker")
stop_event = threading.Event()


def callback(message: pubsub_v1.subscriber.message.Message) -> None:
    try:
        payload = json.loads(message.data.decode("utf-8"))
        document = DocumentInput.model_validate(payload)
        count = upsert_document(document)
        logger.info("Ingested doc_id=%s chunks=%s", document.doc_id, count)
        message.ack()
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.error("Permanent invalid message: %s", exc)
        # Ack malformed input so it does not poison the subscription forever.
        message.ack()
    except Exception:
        logger.exception("Transient ingestion failure; message will be retried")
        message.nack()


def request_stop(*_: object) -> None:
    stop_event.set()


def main() -> None:
    settings = get_settings()
    if not settings.gcp_project_id:
        raise RuntimeError("GCP_PROJECT_ID must be configured")

    initialize_collection()
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(
        settings.gcp_project_id,
        settings.pubsub_subscription,
    )

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    streaming_pull = subscriber.subscribe(
        subscription_path,
        callback=callback,
        flow_control=pubsub_v1.types.FlowControl(max_messages=20),
    )
    logger.info("Listening on %s", subscription_path)

    try:
        while not stop_event.wait(timeout=1):
            if streaming_pull.done():
                streaming_pull.result()
    except TimeoutError:
        pass
    finally:
        streaming_pull.cancel()
        streaming_pull.result(timeout=30)
        subscriber.close()


if __name__ == "__main__":
    main()
