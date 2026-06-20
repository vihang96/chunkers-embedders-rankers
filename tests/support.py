# Copyright (c) 2021-2026 Kognitos, Inc. All rights reserved.
"""Test-only scaffolding.

These helpers are intentionally NOT part of the shipped ``cer`` package. They
provide a concrete tool model and credential lookups that the test suite needs,
without polluting the library namespace.
"""

import json
import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolSchema(BaseModel):
    """Concrete tool model used to exercise the generic retrievers in tests."""

    procedure: str = ""
    book_name: str = ""
    signature: str = ""
    description: str = ""
    skill_type: str = ""
    boolean: bool = False
    hidden: bool = False
    deprecated: bool = False
    good_examples: List[str] = Field(default_factory=list)
    bad_examples: List[str] = Field(default_factory=list)
    key_topics: List[str] = Field(default_factory=list)
    synthetic_queries: List[str] = Field(default_factory=list)


async def get_openai_key() -> Optional[str]:
    """Returns the OpenAI API key from the environment, if configured."""
    return os.getenv("OPENAI_API_KEY")


async def get_gemini_credentials() -> Dict[str, Any]:
    """Returns Gemini/Vertex credentials gathered from the environment.

    Looks for an API key and an optional service-account info blob (either inline
    JSON via ``GOOGLE_SERVICE_ACCOUNT_INFO`` or a path via
    ``GOOGLE_APPLICATION_CREDENTIALS``). Missing values come back as ``None`` so
    integration tests can skip cleanly.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    service_account_info: Optional[Dict[str, Any]] = None

    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_INFO")
    path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if raw:
        try:
            service_account_info = json.loads(raw)
        except json.JSONDecodeError:
            service_account_info = None
    elif path and os.path.exists(path):
        try:
            with open(path) as f:
                service_account_info = json.load(f)
        except (OSError, json.JSONDecodeError):
            service_account_info = None

    return {"api_key": api_key, "service_account_info": service_account_info}
