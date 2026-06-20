import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--embedding-model",
        action="store",
        default="gemini",
        help="Specify the embedding model to use: gemini, openai, or st_e5",
        choices=("gemini", "openai", "st_e5"),
    )
    parser.addoption(
        "--llm-provider",
        action="store",
        default="openai",
        help="Specify the LLM provider to use: openai, gemini, anthropic",
        choices=("openai", "gemini", "gemini_vertex", "anthropic"),
    )
    parser.addoption(
        "--llm-model-name",
        action="store",
        default="gpt-5.2",
        help="Specify the LLM model name to use from the specified provider",
    )
