from openai import OpenAI

from app.config import get_settings
from app.models import AskResponse
from app.weaviate_store import semantic_search


SYSTEM_PROMPT = """You are a grounded enterprise knowledge assistant.
Answer only from the supplied context.
If the context does not contain the answer, say that you do not know.
Cite sources in square brackets using the source number, for example [1].
Do not invent policies, dates, people, metrics, or links."""


def answer_question(
    question: str,
    limit: int,
    department: str | None = None,
) -> AskResponse:
    settings = get_settings()
    hits = semantic_search(question, limit, department)

    if not hits:
        return AskResponse(
            answer="I do not know. No relevant passages were retrieved.",
            sources=[],
            generation_enabled=bool(settings.openai_api_key),
        )

    if not settings.openai_api_key:
        return AskResponse(
            answer=(
                "Generation is disabled because OPENAI_API_KEY is not configured. "
                "The relevant passages are returned in sources."
            ),
            sources=hits,
            generation_enabled=False,
        )

    context = "\n\n".join(
        f"[{index}] Title: {hit.title}\n"
        f"Source: {hit.source}\n"
        f"Content: {hit.content}"
        for index, hit in enumerate(hits, start=1)
    )

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.responses.create(
        model=settings.openai_model,
        instructions=SYSTEM_PROMPT,
        input=f"Question:\n{question}\n\nContext:\n{context}",
        temperature=0,
    )

    return AskResponse(
        answer=response.output_text.strip(),
        sources=hits,
        generation_enabled=True,
    )
