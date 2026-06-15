"""Parser module for loading function definitions and prompts."""

import json
from typing import Any


class ParseError(Exception):
    """Custom exception for parsing errors."""

    def __init__(self, cause: str) -> None:
        """Initialize ParseError with a cause message."""
        super().__init__(f"ERROR: {cause}")
        self.cause = cause


class Parser:
    """Parser for loading function definitions and prompts from JSON files."""

    def _load_json(self, path: str) -> Any:
        """Load and validate a JSON file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
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
        data = self._load_json(path)
        for i, item in enumerate(data):
            for key in ("name", "parameters", "returns", "description"):
                if key not in item:
                    raise ParseError(
                        f"Function at index {i} missing key: '{key}'"
                    )
        return data

    def load_prompt(self, path: str) -> list[dict[str, Any]]:
        """Load prompts from a JSON file."""
        data = self._load_json(path)
        for i, item in enumerate(data):
            if "prompt" not in item:
                raise ParseError(
                    f"Prompt at index {i} missing key: 'prompt'"
                )
        return data
