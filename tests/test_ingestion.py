import hashlib
import pytest

from app.ingestion import TextChunk, normalize_text, split_text


# -------------------------------------------------------------------
# normalize_text tests
# -------------------------------------------------------------------


def test_normalize_text_removes_extra_whitespace() -> None:
    text = "  Customer   renewal\nplanning\tshould begin early.  "

    result = normalize_text(text)

    assert result == "Customer renewal planning should begin early."


def test_normalize_text_empty_string() -> None:
    assert normalize_text("") == ""


def test_normalize_text_only_whitespace() -> None:
    assert normalize_text(" \n\t   ") == ""


def test_normalize_text_preserves_normal_text() -> None:
    text = "This text is already normalized."

    assert normalize_text(text) == text


# -------------------------------------------------------------------
# split_text basic behavior
# -------------------------------------------------------------------


def test_split_text_returns_empty_list_for_empty_text() -> None:
    result = split_text(
        text="",
        doc_id="doc-001",
        chunk_size=50,
        chunk_overlap=10,
    )

    assert result == []


def test_split_text_returns_empty_list_for_whitespace_text() -> None:
    result = split_text(
        text="   \n\t ",
        doc_id="doc-001",
        chunk_size=50,
        chunk_overlap=10,
    )

    assert result == []


def test_split_text_single_chunk() -> None:
    result = split_text(
        text="Renewal planning should begin early.",
        doc_id="doc-001",
        chunk_size=100,
        chunk_overlap=10,
    )

    assert len(result) == 1
    assert isinstance(result[0], TextChunk)
    assert result[0].chunk_index == 0
    assert result[0].content == "Renewal planning should begin early."


def test_split_text_normalizes_text_before_chunking() -> None:
    result = split_text(
        text="  Renewal   planning\nshould\tbegin early. ",
        doc_id="doc-001",
        chunk_size=100,
        chunk_overlap=10,
    )

    assert len(result) == 1
    assert result[0].content == "Renewal planning should begin early."


# -------------------------------------------------------------------
# chunk size and overlap validation
# -------------------------------------------------------------------


def test_split_text_raises_when_overlap_equals_chunk_size() -> None:
    with pytest.raises(
        ValueError,
        match="chunk_overlap must be smaller than chunk_size",
    ):
        split_text(
            text="Some document text",
            doc_id="doc-001",
            chunk_size=20,
            chunk_overlap=20,
        )


def test_split_text_raises_when_overlap_exceeds_chunk_size() -> None:
    with pytest.raises(
        ValueError,
        match="chunk_overlap must be smaller than chunk_size",
    ):
        split_text(
            text="Some document text",
            doc_id="doc-001",
            chunk_size=20,
            chunk_overlap=25,
        )


# -------------------------------------------------------------------
# multiple chunk behavior
# -------------------------------------------------------------------


def test_split_text_creates_multiple_chunks() -> None:
    text = "one two three four five six"

    result = split_text(
        text=text,
        doc_id="doc-001",
        chunk_size=13,
        chunk_overlap=0,
    )

    assert len(result) == 2
    assert [chunk.content for chunk in result] == [
        "one two three",
        "four five six",
    ]


def test_split_text_assigns_sequential_chunk_indexes() -> None:
    result = split_text(
        text="one two three four five six seven",
        doc_id="doc-001",
        chunk_size=13,
        chunk_overlap=0,
    )

    assert [chunk.chunk_index for chunk in result] == list(
        range(len(result))
    )


def test_split_text_uses_overlap() -> None:
    result = split_text(
        text="one two three four five six",
        doc_id="doc-001",
        chunk_size=13,
        chunk_overlap=5,
    )

    contents = [chunk.content for chunk in result]

    assert contents[0] == "one two three"
    assert contents[1].startswith("three")
    assert "four" in contents[1]


def test_split_text_overlap_repeats_previous_words() -> None:
    result = split_text(
        text="alpha beta gamma delta epsilon",
        doc_id="doc-001",
        chunk_size=16,
        chunk_overlap=6,
    )

    assert len(result) >= 2

    first_words = result[0].content.split()
    second_words = result[1].content.split()

    assert any(word in second_words for word in first_words)


# -------------------------------------------------------------------
# chunk identifier tests
# -------------------------------------------------------------------


def test_split_text_generates_expected_chunk_id() -> None:
    doc_id = "doc-001"
    content = "Renewal planning should begin early."

    result = split_text(
        text=content,
        doc_id=doc_id,
        chunk_size=100,
        chunk_overlap=10,
    )

    expected_id = hashlib.sha256(
        f"{doc_id}:0:{content}".encode("utf-8")
    ).hexdigest()

    assert result[0].chunk_id == expected_id


def test_split_text_chunk_ids_are_deterministic() -> None:
    arguments = {
        "text": "one two three four five six",
        "doc_id": "doc-001",
        "chunk_size": 13,
        "chunk_overlap": 5,
    }

    first_result = split_text(**arguments)
    second_result = split_text(**arguments)

    assert first_result == second_result


def test_split_text_different_doc_ids_create_different_chunk_ids() -> None:
    first_result = split_text(
        text="Same document content",
        doc_id="doc-001",
        chunk_size=100,
        chunk_overlap=10,
    )

    second_result = split_text(
        text="Same document content",
        doc_id="doc-002",
        chunk_size=100,
        chunk_overlap=10,
    )

    assert first_result[0].content == second_result[0].content
    assert first_result[0].chunk_id != second_result[0].chunk_id


def test_split_text_different_content_creates_different_chunk_ids() -> None:
    first_result = split_text(
        text="First document content",
        doc_id="doc-001",
        chunk_size=100,
        chunk_overlap=10,
    )

    second_result = split_text(
        text="Second document content",
        doc_id="doc-001",
        chunk_size=100,
        chunk_overlap=10,
    )

    assert first_result[0].chunk_id != second_result[0].chunk_id


def test_each_chunk_has_unique_id() -> None:
    result = split_text(
        text="one two three four five six seven eight",
        doc_id="doc-001",
        chunk_size=13,
        chunk_overlap=5,
    )

    chunk_ids = [chunk.chunk_id for chunk in result]

    assert len(chunk_ids) == len(set(chunk_ids))


# -------------------------------------------------------------------
# edge cases
# -------------------------------------------------------------------


def test_split_text_handles_word_longer_than_chunk_size() -> None:
    long_word = "extraordinarylongword"

    result = split_text(
        text=long_word,
        doc_id="doc-001",
        chunk_size=5,
        chunk_overlap=0,
    )

    assert len(result) == 1
    assert result[0].content == long_word


def test_split_text_does_not_split_inside_words() -> None:
    result = split_text(
        text="customer renewal planning",
        doc_id="doc-001",
        chunk_size=10,
        chunk_overlap=0,
    )

    contents = [chunk.content for chunk in result]

    assert "customer" in contents
    assert "renewal" in contents
    assert "planning" in contents


def test_text_chunk_is_immutable() -> None:
    chunk = TextChunk(
        chunk_id="chunk-001",
        chunk_index=0,
        content="Sample content",
    )

    with pytest.raises(AttributeError):
        chunk.content = "Changed content"

