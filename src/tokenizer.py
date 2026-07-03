"""Tokenizer module implementing custom BPE encode and decode."""

import json
import os
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
            merges_path = os.path.join(
                os.path.dirname(self.vocab_path), "merges.txt")
            self.merges: dict[tuple[str, str], int] = {}
            with open(merges_path, "r", encoding="utf-8") as f:
                for rank, line in enumerate(f):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(" ", 1)
                    if len(parts) == 2:
                        self.merges[(parts[0], parts[1])] = rank
        except OSError as e:
            raise ParseError(f"Could not read vocab file: {e}") from e
        except json.JSONDecodeError as e:
            raise ParseError(f"Invalid vocab JSON: {e}") from e
        except Exception as e:
            raise ParseError(f"Failed to initialize tokenizer: {e}") from e

    def my_encode(self, text: str) -> list[int]:
        """Encode text into a list of token IDs using greedy BPE merges."""
        try:
            processed = text.replace(" ", "Ġ").replace("\n", "Ċ")
            symbols: list[str] = []
            for char in processed:
                if char in self.vocab:
                    symbols.append(char)
                else:
                    for byte in char.encode("utf-8"):
                        symbols.append(f"<0x{byte:02X}>")
            while len(symbols) > 1:
                best_rank = -1
                best_idx = -1

                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i + 1])
                    rank = self.merges.get(pair, -1)
                    if rank != -1 and (best_rank == -1 or rank < best_rank):
                        best_rank = rank
                        best_idx = i

                if best_idx == -1:
                    break

                left = symbols[best_idx]
                right = symbols[best_idx + 1]
                merged = left + right

                new_symbols: list[str] = []
                i = 0
                while i < len(symbols):
                    if (
                        i < len(symbols) - 1
                        and symbols[i] == left
                        and symbols[i + 1] == right
                    ):
                        new_symbols.append(merged)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                symbols = new_symbols

            # step 3 — look up final symbols in vocab
            token_ids: list[int] = []
            for symbol in symbols:
                if symbol in self.vocab:
                    token_ids.append(self.vocab[symbol])
                else:
                    for byte in symbol.encode("utf-8"):
                        key = f"<0x{byte:02X}>"
                        if key in self.vocab:
                            token_ids.append(self.vocab[key])
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
