"""Tokenizer module implementing custom BPE encode and decode."""

import json
from typing import Any
from pydantic import BaseModel, field_validator
from .parsing import ParseError


class VocabEntry(BaseModel):
    """Pydantic model for validating the vocabulary structure."""

    vocab: dict[str, int]

    @field_validator("vocab")
    @classmethod
    def vocab_must_not_be_empty(cls, v: dict[str, int]) -> dict[str, int]:
        """Validate that the vocabulary is not empty."""
        if not v:
            raise ValueError("Vocabulary must not be empty")
        return v


class TokenizeInput(BaseModel):
    """Pydantic model for validating tokenizer input."""

    text: str

    @field_validator("text")
    @classmethod
    def text_must_be_string(cls, v: str) -> str:
        """Validate that the input is a non-None string."""
        if v is None:
            raise ValueError("Input text must not be None")
        return v


class Tokenizer:
    """Custom BPE tokenizer using only vocab file."""

    def __init__(self, fun: Any) -> None:
        """Initialize tokenizer and load vocabulary."""
        try:
            self.fun: Any = fun
            self.vocab_path: str = self.fun.model.get_path_to_vocab_file()
            with open(self.vocab_path, "r", encoding="utf-8") as f:
                raw_vocab: Any = json.load(f)
            validated: VocabEntry = VocabEntry(vocab=raw_vocab)
            self.vocab: dict[str, int] = validated.vocab
            self.id_to_token: dict[int, str] = {
                v: k for k, v in self.vocab.items()
            }
        except OSError as e:
            raise ParseError(f"Could not read vocab file: {e}") from e
        except json.JSONDecodeError as e:
            raise ParseError(f"Invalid vocab JSON: {e}") from e
        except Exception as e:
            raise ParseError(f"Failed to initialize tokenizer: {e}") from e

    def my_encode(self, text: str) -> list[int]:
        """Encode text into token ids with greedy longest-match lookup."""
        try:
            TokenizeInput(text=text)
            token_ids: list[int] = []
            processed: str = text.replace(" ", "Ġ").replace("\n", "Ċ")
            i: int = 0
            while i < len(processed):
                found_word: int | None = None
                length_word: int = 0
                for j in range(len(processed), i, -1):
                    word: str = processed[i:j]
                    if word in self.vocab:
                        found_word = self.vocab[word]
                        length_word = len(word)
                        break
                if found_word is not None:
                    token_ids.append(found_word)
                    i += length_word
                else:
                    i += 1
            return token_ids
        except Exception as e:
            raise ParseError(f"Encode error: {e}") from e

    def my_decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back into a string."""
        try:
            result: str = ""
            byte_buffer: bytes = b""

            for token_id in token_ids:
                if token_id not in self.id_to_token:
                    raise ParseError(
                        f"Token ID {token_id} not found in vocabulary"
                    )
                token_str: str = self.id_to_token[token_id]

                if token_str.startswith("<0x") and token_str.endswith(">"):
                    try:
                        byte_val: int = int(token_str[3:-1], 16)
                        byte_buffer += bytes([byte_val])
                    except ValueError:
                        byte_buffer += b"?"
                        continue
                else:
                    if byte_buffer:
                        result += byte_buffer.decode(
                            "utf-8", errors="replace"
                        )
                        byte_buffer = b""
                    result += token_str.replace("Ġ", " ")

            if byte_buffer:
                result += byte_buffer.decode("utf-8", errors="replace")

            return result
        except ParseError:
            raise
        except Exception as e:
            raise ParseError(f"Decode error: {e}") from e
