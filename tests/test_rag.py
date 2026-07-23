from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.models import AskResponse, SearchHit
from app.rag import SYSTEM_PROMPT, answer_question


class TestAnswerQuestion:
    """Unit tests for app.rag.answer_question."""

    @staticmethod
    def create_hit(
        doc_id: str = "policy-001",
        chunk_id: str = "policy-001-chunk-0",
        title: str = "Renewal Policy",
        content: str = "Enterprise renewal planning should begin 120 days early.",
        chunk_index: int = 0,
        score: float = 0.95,
        source: str = "renewal-policy.txt",
        department: str = "Customer Success",
    ) -> SearchHit:
        return SearchHit(
            doc_id=doc_id,
            chunk_id=chunk_id,
            title=title,
            content=content,
            chunk_index=chunk_index,
            score=score,
            source=source,
            department=department,
        )

    @patch("app.rag.semantic_search")
    @patch("app.rag.get_settings")
    def test_returns_unknown_when_no_hits_and_api_key_exists(
        self,
        mock_get_settings,
        mock_semantic_search,
    ) -> None:
        mock_get_settings.return_value = SimpleNamespace(
            openai_api_key="test-api-key",
            openai_model="gpt-test",
        )
        mock_semantic_search.return_value = []

        result = answer_question(
            question="When should renewal planning begin?",
            limit=5,
            department="Customer Success",
        )

        assert isinstance(result, AskResponse)
        assert result.answer == (
            "I do not know. No relevant passages were retrieved."
        )
        assert result.sources == []
        assert result.generation_enabled is True

        mock_semantic_search.assert_called_once_with(
            "When should renewal planning begin?",
            5,
            "Customer Success",
        )

    @patch("app.rag.semantic_search")
    @patch("app.rag.get_settings")
    def test_returns_unknown_when_no_hits_and_api_key_missing(
        self,
        mock_get_settings,
        mock_semantic_search,
    ) -> None:
        mock_get_settings.return_value = SimpleNamespace(
            openai_api_key=None,
            openai_model="gpt-test",
        )
        mock_semantic_search.return_value = []

        result = answer_question(
            question="What is the policy?",
            limit=3,
        )

        assert result.answer == (
            "I do not know. No relevant passages were retrieved."
        )
        assert result.sources == []
        assert result.generation_enabled is False

        mock_semantic_search.assert_called_once_with(
            "What is the policy?",
            3,
            None,
        )

    @patch("app.rag.semantic_search")
    @patch("app.rag.get_settings")
    def test_returns_sources_when_generation_is_disabled(
        self,
        mock_get_settings,
        mock_semantic_search,
    ) -> None:
        hit = self.create_hit()

        mock_get_settings.return_value = SimpleNamespace(
            openai_api_key="",
            openai_model="gpt-test",
        )
        mock_semantic_search.return_value = [hit]

        result = answer_question(
            question="When should renewal planning begin?",
            limit=5,
            department="Customer Success",
        )

        assert result.answer == (
            "Generation is disabled because OPENAI_API_KEY is not configured. "
            "The relevant passages are returned in sources."
        )
        assert result.sources == [hit]
        assert result.generation_enabled is False

    @patch("app.rag.OpenAI")
    @patch("app.rag.semantic_search")
    @patch("app.rag.get_settings")
    def test_generates_answer_using_openai(
        self,
        mock_get_settings,
        mock_semantic_search,
        mock_openai,
    ) -> None:
        hit = self.create_hit()

        mock_get_settings.return_value = SimpleNamespace(
            openai_api_key="test-api-key",
            openai_model="gpt-test-model",
        )
        mock_semantic_search.return_value = [hit]

        mock_client = MagicMock()
        mock_response = SimpleNamespace(
            output_text="  Renewal planning should begin 120 days early [1].  "
        )
        mock_client.responses.create.return_value = mock_response
        mock_openai.return_value = mock_client

        result = answer_question(
            question="When should renewal planning begin?",
            limit=5,
            department="Customer Success",
        )

        assert result.answer == (
            "Renewal planning should begin 120 days early [1]."
        )
        assert result.sources == [hit]
        assert result.generation_enabled is True

        mock_openai.assert_called_once_with(
            api_key="test-api-key",
        )

        mock_client.responses.create.assert_called_once_with(
            model="gpt-test-model",
            instructions=SYSTEM_PROMPT,
            input=(
                "Question:\n"
                "When should renewal planning begin?\n\n"
                "Context:\n"
                "[1] Title: Renewal Policy\n"
                "Source: renewal-policy.txt\n"
                "Content: Enterprise renewal planning should begin "
                "120 days early."
            ),
            temperature=0,
        )

    @patch("app.rag.OpenAI")
    @patch("app.rag.semantic_search")
    @patch("app.rag.get_settings")
    def test_builds_context_from_multiple_hits(
        self,
        mock_get_settings,
        mock_semantic_search,
        mock_openai,
    ) -> None:
        first_hit = self.create_hit()

        second_hit = self.create_hit(
            doc_id="playbook-001",
            chunk_id="playbook-001-chunk-0",
            title="Renewal Playbook",
            content="The account team should review renewal risk.",
            source="renewal-playbook.md",
            chunk_index=0,
            score=0.90,
        )

        mock_get_settings.return_value = SimpleNamespace(
            openai_api_key="test-api-key",
            openai_model="gpt-test-model",
        )
        mock_semantic_search.return_value = [
            first_hit,
            second_hit,
        ]

        mock_client = MagicMock()
        mock_client.responses.create.return_value = SimpleNamespace(
            output_text="Planning should begin early [1].",
        )
        mock_openai.return_value = mock_client

        result = answer_question(
            question="How should renewal planning work?",
            limit=2,
            department="Customer Success",
        )

        assert result.generation_enabled is True
        assert result.sources == [first_hit, second_hit]

        call_arguments = mock_client.responses.create.call_args.kwargs
        prompt_input = call_arguments["input"]

        assert "[1] Title: Renewal Policy" in prompt_input
        assert "Source: renewal-policy.txt" in prompt_input
        assert (
            "Content: Enterprise renewal planning should begin "
            "120 days early."
            in prompt_input
        )

        assert "[2] Title: Renewal Playbook" in prompt_input
        assert "Source: renewal-playbook.md" in prompt_input
        assert (
            "Content: The account team should review renewal risk."
            in prompt_input
        )

    @patch("app.rag.OpenAI")
    @patch("app.rag.semantic_search")
    @patch("app.rag.get_settings")
    def test_passes_department_as_none(
        self,
        mock_get_settings,
        mock_semantic_search,
        mock_openai,
    ) -> None:
        hit = self.create_hit()

        mock_get_settings.return_value = SimpleNamespace(
            openai_api_key="test-api-key",
            openai_model="gpt-test-model",
        )
        mock_semantic_search.return_value = [hit]

        mock_client = MagicMock()
        mock_client.responses.create.return_value = SimpleNamespace(
            output_text="Answer [1].",
        )
        mock_openai.return_value = mock_client

        answer_question(
            question="What does the policy say?",
            limit=4,
        )

        mock_semantic_search.assert_called_once_with(
            "What does the policy say?",
            4,
            None,
        )

    @patch("app.rag.OpenAI")
    @patch("app.rag.semantic_search")
    @patch("app.rag.get_settings")
    def test_openai_exception_is_propagated(
        self,
        mock_get_settings,
        mock_semantic_search,
        mock_openai,
    ) -> None:
        hit = self.create_hit()

        mock_get_settings.return_value = SimpleNamespace(
            openai_api_key="test-api-key",
            openai_model="gpt-test-model",
        )
        mock_semantic_search.return_value = [hit]

        mock_client = MagicMock()
        mock_client.responses.create.side_effect = RuntimeError(
            "OpenAI request failed"
        )
        mock_openai.return_value = mock_client

        try:
            answer_question(
                question="What is the renewal policy?",
                limit=5,
            )
            assert False, "Expected RuntimeError"
        except RuntimeError as exc:
            assert str(exc) == "OpenAI request failed" 
