"""Parser module for loading function definitions and prompts."""

import json
from typing import Any
from pydantic import BaseModel, field_validator


class ParseError(Exception):
    """Custom exception for parsing errors."""

    def __init__(self, cause: str) -> None:
        """Initialize ParseError with a cause message."""
        super().__init__(f"ERROR: {cause}")
        self.cause: str = cause


class PromptItem(BaseModel):
    """Pydantic model for validating a single prompt entry."""

    prompt: str

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_empty(cls, v: str) -> str:
        """Validate that the prompt is not empty."""
        if not v.strip():
            raise ValueError("Prompt must not be empty")
        return v


class FunctionItem(BaseModel):
    """Pydantic model for validating a single function entry."""

    name: str
    parameters: dict[str, Any]
    returns: dict[str, Any]
    description: str

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        """Validate that the function name is not empty."""
        if not v.strip():
            raise ValueError("Function name must not be empty")
        return v

    @field_validator("description")
    @classmethod
    def description_must_not_be_empty(cls, v: str) -> str:
        """Validate that the description is not empty."""
        if not v.strip():
            raise ValueError("Description must not be empty")
        return v


class Parser:
    """Parser for loading function definitions and prompts from JSON files."""

    def _load_json(self, path: str) -> list[Any]:
        """Load and validate a JSON file."""
        try:
            with open(path, 'r') as f:
                data: Any = json.load(f)
        except FileNotFoundError:
            raise ParseError(f"File not found: {path}")
        except json.JSONDecodeError as e:
            raise ParseError(f"Invalid JSON in {path}: {e}")
        except OSError as e:
            raise ParseError(f"Could not read {path}: {e}")

        if not isinstance(data, list):
            raise ParseError(f"Expected a list in {path}")
        if len(data) == 0:
            raise ParseError(f"Empty file: {path}")
        return data

    def load_functions(self, path: str) -> list[dict[str, Any]]:
        """Load function definitions from a JSON file."""
        data: list[Any] = self._load_json(path)
        for i, item in enumerate(data):
            try:
                FunctionItem(**item)
            except Exception as e:
                raise ParseError(
                    f"Invalid function at index {i}: {e}"
                ) from e
        return data

    def load_prompt(self, path: str) -> list[dict[str, Any]]:
        """Load prompts from a JSON file."""
        data: list[Any] = self._load_json(path)
        for i, item in enumerate(data):
            try:
                PromptItem(**item)
            except Exception as e:
                raise ParseError(
                    f"Invalid prompt at index {i}: {e}"
                ) from e
        return data
