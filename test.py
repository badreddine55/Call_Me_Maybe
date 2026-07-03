"""Function calling assistant using constrained decoding."""

import sys
import os
from typing import Any
import numpy as np
from pydantic import BaseModel, field_validator

sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), '..', 'llm_sdk'
))
from llm_sdk import Small_LLM_Model  # noqa: E402
from src.parsing import Parser, ParseError  # noqa: E402


NO_FUNCTION_NAME: str = "__no_function__"


class FunctionParam(BaseModel):
    """Pydantic model for validating a single function parameter."""

    type: str

    @field_validator("type")
    @classmethod
    def type_must_be_valid(cls, v: str) -> str:
        """Validate that the parameter type is supported."""
        allowed: set[str] = {"string", "number", "integer", "boolean"}
        if v not in allowed:
            raise ValueError(f"Unsupported parameter type: {v}")
        return v


class FunctionDefinition(BaseModel):
    """Pydantic model for validating a function definition."""

    name: str
    description: str
    parameters: dict[str, FunctionParam]
    returns: dict[str, str]

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        """Validate that the function name is not empty."""
        if not v.strip():
            raise ValueError("Function name must not be empty")
        return v


class FunctionsDict:
    """Function calling assistant using constrained decoding."""

    def __init__(self) -> None:
        """Initialize the FunctionsDict with parser and model."""
        self.parser: Parser = Parser()
        self.model: Small_LLM_Model = Small_LLM_Model()
        self.functions: list[dict[str, Any]] = []
        self.tokenizer: Any = None

    def load_tokenizer(self) -> None:
        """Load tokenizer after model is ready to avoid circular import."""
        try:
            from src.tokenizer import Tokenizer
            self.tokenizer = Tokenizer(self)
        except Exception as e:
            raise ParseError(f"Failed to load tokenizer: {e}") from e

    def _validate_function(
        self, fun: dict[str, Any], index: int
    ) -> FunctionDefinition:
        """Validate a function definition using pydantic."""
        try:
            return FunctionDefinition(**fun)
        except Exception as e:
            raise ParseError(
                f"Invalid function at index {index}: {e}"
            ) from e

    def functions_list(
        self, path: str
    ) -> tuple[list[str], list[list[int]]]:
        """Load and format functions from a JSON file.

        The returned ``functions_names`` list always includes a final
        entry for ``NO_FUNCTION_NAME`` (``__no_function__``), tokenized
        the same way as every real function name, so that "no function
        matches this request" is a genuine candidate the model can
        select during ``extract_function_name`` rather than a fallback
        handled outside the selection logic.
        """
        try:
            new_list_functions: list[str] = []
            functions_names: list[list[int]] = []
            self.functions = self.parser.load_functions(path)

            for i, fun in enumerate(self.functions):
                validated: FunctionDefinition = self._validate_function(
                    fun, i
                )
                params: dict[str, FunctionParam] = validated.parameters
                params_str: str = ", ".join(
                    f"{k}: {v.type}" for k, v in params.items()
                )
                functions_names.append(
                    self.tokenizer.my_encode(validated.name)
                )
                line: str = (
                    f"function_name: {validated.name}({params_str}) "
                    f":returns({validated.returns.get('type', '')}) "
                    f":description({validated.description})"
                )
                new_list_functions.append(line)

            # Add __no_function__ as a real, scoreable candidate.
            functions_names.append(
                self.tokenizer.my_encode(NO_FUNCTION_NAME)
            )

            return new_list_functions, functions_names
        except ParseError:
            raise
        except Exception as e:
            raise ParseError(f"Failed to load functions list: {e}") from e

    def prompt_list(self, path: str) -> list[str]:
        """Load prompts from a JSON file."""
        try:
            new_list_prompt: list[str] = []
            prompts = self.parser.load_prompt(path)
            for prompt in prompts:
                new_list_prompt.append(prompt["prompt"])
            return new_list_prompt
        except ParseError:
            raise
        except Exception as e:
            raise ParseError(f"Failed to load prompt list: {e}") from e

    def build_prompt(
        self, user_request: str, functions_list: list[str]
    ) -> str:
        """Build the prompt string for the LLM."""
        functions_str: str = "\n".join(functions_list)
        return (
            f"Available functions:\n{functions_str}\n\n"
            f"User request: {user_request}\n\n"
            "- For regex parameters, generate a valid regex "
            "pattern (e.g use [...] not ...)\n\n"
            "- Return __no_function__ only if none of the listed "
            "functions can handle the request.\n\n"
            'JSON response:\n{"function_name": "'
        )

    def get_logits(self, input_ids: list[int]) -> list[float]:
        """Get logits from the model for the given input IDs."""
        try:
            return self.model.get_logits_from_input_ids(input_ids)
        except Exception as e:
            raise ParseError(f"Failed to get logits: {e}") from e

    def extract_function_name(
        self,
        list_func_tokens: list[list[int]],
        input_ids: list[int]
    ) -> list[int] | None:
        """Select the function name by scoring full candidate sequences.

        Each candidate in ``list_func_tokens`` (including the
        ``__no_function__`` candidate appended in ``functions_list``)
        is scored independently via teacher forcing: the model is
        walked through each candidate's exact tokens, and the
        log-probability of every token is accumulated and averaged
        over the candidate's length. This avoids the greedy-fork
        problem of committing to a single token at a shared position,
        since every candidate is evaluated fully and in isolation
        before any comparison is made. The highest-scoring candidate
        wins, whether that is a real function or __no_function__.
        """
        try:
            best_score: float = -float('inf')
            best_candidate: list[int] | None = None

            for candidate in list_func_tokens:
                if not candidate:
                    continue

                total_log_prob: float = 0.0
                current_ids: list[int] = list(input_ids)

                for token_id in candidate:
                    logits: np.ndarray = np.array(
                        self.get_logits(current_ids)
                    )
                    log_probs: np.ndarray = (
                        logits - np.logaddexp.reduce(logits)
                    )
                    total_log_prob += float(log_probs[token_id])
                    current_ids.append(token_id)

                avg_log_prob: float = total_log_prob / len(candidate)

                if avg_log_prob > best_score:
                    best_score = avg_log_prob
                    best_candidate = candidate

            return best_candidate
        except ParseError:
            raise
        except Exception as e:
            raise ParseError(
                f"Failed to extract function name: {e}"
            ) from e

    def _get_number_tokens(self) -> set[int]:
        """Get token IDs valid after the first token of a number."""
        try:
            allowed: set[int] = set()
            for c in "0123456789.":
                tokens: list[int] = self.tokenizer.my_encode(c)
                allowed.update(tokens)
            return allowed
        except Exception as e:
            raise ParseError(
                f"Failed to get number tokens: {e}"
            ) from e

    def _extract_string(
        self, input_ids: list[int]
    ) -> str:
        """Extract a string parameter using constrained decoding."""
        try:
            str_value: list[int] = []
            input_ids.extend(self.tokenizer.my_encode(' "'))
            previous_token_text: str | None = None

            while True:
                logits: list[float] = self.get_logits(input_ids)
                token: int = int(np.argmax(logits))
                token_text: str = self.tokenizer.my_decode([token])

                if '"' in token_text:
                    if previous_token_text == "\\":
                        quote_tokens: list[int] = (
                            self.tokenizer.my_encode('"')
                        )
                        input_ids.extend(quote_tokens)
                        str_value.extend(quote_tokens)
                        previous_token_text = token_text
                        continue

                    new_text: str = token_text.split('"')[0]
                    if new_text:
                        tokens: list[int] = (
                            self.tokenizer.my_encode(new_text)
                        )
                        input_ids.extend(tokens)
                        str_value.extend(tokens)
                    break

                input_ids.append(token)
                str_value.append(token)
                previous_token_text = token_text

            result: str = self.tokenizer.my_decode(str_value).strip()
            result = result.replace("\\\\", "\\")
            input_ids.extend(self.tokenizer.my_encode('"'))
            return result
        except ParseError:
            raise
        except Exception as e:
            raise ParseError(f"Failed to extract string: {e}") from e

    def _extract_number(
        self, input_ids: list[int], is_integer: bool
    ) -> int | float:
        """Extract a number parameter using constrained decoding."""
        try:
            value: list[int] = []

            while True:
                logits: list[float] = self.get_logits(input_ids)
                token: int = int(np.argmax(logits))
                token_text: str = self.tokenizer.my_decode([token])

                if any(c in token_text for c in {",", "}"}):
                    digit_part: str = (
                        token_text.split(",")[0]
                        .split("}")[0]
                        .strip()
                    )
                    if digit_part:
                        input_ids.extend(
                            self.tokenizer.my_encode(digit_part)
                        )
                    break

                input_ids.append(token)
                value.append(token)

            raw: str = self.tokenizer.my_decode(value).strip().strip('"')
            return int(raw) if is_integer else float(raw)
        except ParseError:
            raise
        except Exception as e:
            raise ParseError(f"Failed to extract number: {e}") from e

    def _extract_boolean(
        self, input_ids: list[int]
    ) -> bool:
        """Extract a boolean parameter using constrained decoding."""
        try:
            true_tokens: list[int] = self.tokenizer.my_encode("true")
            false_tokens: list[int] = self.tokenizer.my_encode("false")
            logits: np.ndarray = np.array(self.get_logits(input_ids))
            allowed: set[int] = {true_tokens[0], false_tokens[0]}

            for token_id in range(len(logits)):
                if token_id not in allowed:
                    logits[token_id] = -float('inf')

            first_token: int = int(np.argmax(logits))

            if first_token == true_tokens[0]:
                for tok in true_tokens[1:]:
                    input_ids.append(tok)
                return True

            for tok in false_tokens[1:]:
                input_ids.append(tok)
            return False
        except ParseError:
            raise
        except Exception as e:
            raise ParseError(f"Failed to extract boolean: {e}") from e

    def extract_function_params(
        self,
        function_name: str,
        input_ids: list[int]
    ) -> dict[str, Any]:
        """Extract function parameters using constrained decoding."""
        try:
            input_ids.extend(
                self.tokenizer.my_encode('", "parameters": {')
            )
            function: dict[str, Any] = next(
                f for f in self.functions if f["name"] == function_name
            )
            params: list[tuple[str, Any]] = list(
                function["parameters"].items()
            )
            result: dict[str, Any] = {}

            for i, (param_name, param_info) in enumerate(params):
                if i > 0:
                    input_ids.extend(self.tokenizer.my_encode(", "))

                input_ids.extend(
                    self.tokenizer.my_encode(f'"{param_name}":')
                )
                ptype: str = param_info["type"]

                if ptype == "string":
                    result[param_name] = self._extract_string(input_ids)
                elif ptype == "number":
                    result[param_name] = self._extract_number(
                        input_ids, is_integer=False
                    )
                elif ptype == "integer":
                    result[param_name] = self._extract_number(
                        input_ids, is_integer=True
                    )
                elif ptype == "boolean":
                    result[param_name] = self._extract_boolean(input_ids)

            input_ids.extend(self.tokenizer.my_encode("}}"))
            return result
        except ParseError:
            raise
        except Exception as e:
            raise ParseError(
                f"Failed to extract function params: {e}"
            ) from e