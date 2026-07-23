import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    chunk_index: int
    content: str


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_text(
    text: str,
    doc_id: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    """Split text on word boundaries with deterministic chunk identifiers."""
    normalized = normalize_text(text)
    if not normalized:
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    words = normalized.split(" ")
    chunks: list[TextChunk] = []
    start = 0
    index = 0

    while start < len(words):
        current: list[str] = []
        length = 0
        end = start

        while end < len(words):
            word = words[end]
            extra = len(word) + (1 if current else 0)
            if current and length + extra > chunk_size:
                break
            current.append(word)
            length += extra
            end += 1

        content = " ".join(current)
        digest = hashlib.sha256(
            f"{doc_id}:{index}:{content}".encode()
        ).hexdigest()
        chunks.append(
            TextChunk(
                chunk_id=digest,
                chunk_index=index,
                content=content,
            )
        )

        if end >= len(words):
            break

        overlap_chars = 0
        next_start = end
        while chunk_overlap > 0 and next_start > start and overlap_chars < chunk_overlap:
            next_start -= 1
            overlap_chars += len(words[next_start]) + 1

        start = max(next_start, start + 1)
        index += 1

    return chunks
#print(split_text("This is a test for checkinh Chunk",1,6,3))