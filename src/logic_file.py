"""Function calling assistant using constrained decoding."""

import sys
import os
from typing import Any
import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), '..', 'llm_sdk'
))
from llm_sdk import Small_LLM_Model  # noqa: E402
from src.parsing import Parser  # noqa: E402


class FunctionsDict:
    """Function calling assistant using constrained decoding."""

    def __init__(self) -> None:
        """Initialize the FunctionsDict with parser and model."""
        self.parser = Parser()
        self.model = Small_LLM_Model()
        self.functions: list[dict[str, Any]] = []

    def functions_list(
        self, path: str
    ) -> tuple[list[str], list[list[int]]]:
        """Load and format functions from a JSON file."""
        new_list_functions: list[str] = []
        functions_names: list[list[int]] = []
        self.functions = self.parser.load_functions(path)

        for fun in self.functions:
            params = fun["parameters"]
            params_str = ", ".join(
                f"{k}: {v['type']}" for k, v in params.items()
            )
            functions_names.append(
                self.model.encode(fun["name"]).tolist()[0]
            )
            line = (
                f"function_name: {fun['name']}({params_str}) "
                f":returns({fun['returns']['type']}) "
                f":description({fun['description']})"
            )
            new_list_functions.append(line)

        return new_list_functions, functions_names

    def prompt_list(self, path: str) -> list[str]:
        """Load prompts from a JSON file."""
        new_list_prompt: list[str] = []
        prompts = self.parser.load_prompt(path)
        for prompt in prompts:
            new_list_prompt.append(prompt["prompt"])
        return new_list_prompt

    def build_prompt(
        self, user_request: str, functions_list: list[str]
    ) -> str:
        """Build the prompt string for the LLM."""
        functions_str = "\n".join(functions_list)
        return (
            "You are a function calling assistant. "
            "Your job is to select the correct function and "
            "extract the parameters from the user request. "
            "You must ONLY respond with a JSON object, "
            "nothing else.\n\n"
            f"Available functions:\n{functions_str}\n\n"
            "Rules:\n"
            "- Choose the most appropriate function\n"
            "- Extract exact parameter values\n"
            "- Return ONLY valid JSON\n"
            "- For regex parameters, generate a valid regex "
            "pattern (e.g use [...] not ...)\n\n"
            f"User request: {user_request}\n\n"
            'JSON response:\n{"function_name": "'
        )

    def get_logits(self, input_ids: list[int]) -> list[float]:
        """Get logits from the model for the given input IDs."""
        return self.model.get_logits_from_input_ids(input_ids)

    def extract_function_name(
        self,
        list_func_tokens: list[list[int]],
        input_ids: list[int]
    ) -> list[int] | None:
        """Extract the function name using constrained decoding."""
        alive = set(range(len(list_func_tokens)))
        expected_fun: list[int] = []
        pos = 0

        while alive:
            allowed_tokens: set[int] = set()
            still_alive: set[int] = set()

            for idx in alive:
                if pos < len(list_func_tokens[idx]):
                    allowed_tokens.add(list_func_tokens[idx][pos])
                    still_alive.add(idx)

            if not allowed_tokens:
                break

            if len(allowed_tokens) == 1:
                token = next(iter(allowed_tokens))
            else:
                logits = np.array(self.get_logits(input_ids))
                for token_id in range(len(logits)):
                    if token_id not in allowed_tokens:
                        logits[token_id] = -float('inf')
                token = int(np.argmax(logits))

            expected_fun.append(token)
            input_ids.append(token)

            new_alive: set[int] = set()
            for idx in still_alive:
                if (
                    pos < len(list_func_tokens[idx])
                    and list_func_tokens[idx][pos] == token
                ):
                    new_alive.add(idx)
            alive = new_alive
            pos += 1

            for candidate in list_func_tokens:
                if expected_fun == candidate:
                    return expected_fun

        return expected_fun if expected_fun else None

    def _get_number_tokens(self) -> set[int]:
        """Get token IDs valid after the first token of a number."""
        allowed: set[int] = set()
        for c in "0123456789.":
            tokens = self.model.encode(c).tolist()[0]
            allowed.update(tokens)
        return allowed

    def _extract_string(
        self, input_ids: list[int]
    ) -> str:
        """Extract a string parameter using constrained decoding."""
        str_value: list[int] = []
        input_ids.extend(self.model.encode(' "').tolist()[0])
        previous_token_text = None

        while True:
            logits = self.get_logits(input_ids)
            token = int(np.argmax(logits))
            token_text = self.model.decode([token])

            if '"' in token_text:
                if previous_token_text == "\\":
                    input_ids.append(token)
                    str_value.append(token)
                    previous_token_text = token_text
                    continue
                new_text = token_text.split('"')[0]
                if new_text:
                    tokens = self.model.encode(new_text).tolist()[0]
                    input_ids.extend(tokens)
                    str_value.extend(tokens)
                break

            input_ids.append(token)
            str_value.append(token)
            previous_token_text = token_text

        result = self.model.decode(str_value).strip()
        result = result.replace("\\\\", "\\")
        input_ids.extend(self.model.encode('"').tolist()[0])
        return result

    def _extract_number(
        self, input_ids: list[int], is_integer: bool
    ) -> int | float:
        """Extract a number parameter using constrained decoding."""
        value: list[int] = []

        while True:
            logits = self.get_logits(input_ids)
            token = int(np.argmax(logits))
            token_text = self.model.decode([token])

            if any(c in token_text for c in {",", "}"}):
                digit_part = (
                    token_text.split(",")[0]
                    .split("}")[0]
                    .strip()
                )
                if digit_part:
                    input_ids.extend(
                        self.model.encode(digit_part).tolist()[0]
                    )
                break

            input_ids.append(token)
            value.append(token)

        raw = self.model.decode(value).strip().strip('"')
        return int(raw) if is_integer else float(raw)

    def _extract_boolean(
        self, input_ids: list[int]
    ) -> bool:
        """Extract a boolean parameter using constrained decoding."""
        true_tokens = self.model.encode("true").tolist()[0]
        false_tokens = self.model.encode("false").tolist()[0]
        logits = np.array(self.get_logits(input_ids))
        allowed: set[int] = {true_tokens[0], false_tokens[0]}

        for token_id in range(len(logits)):
            if token_id not in allowed:
                logits[token_id] = -float('inf')

        first_token = int(np.argmax(logits))

        if first_token == true_tokens[0]:
            for tok in true_tokens[1:]:
                input_ids.append(tok)
            return True

        for tok in false_tokens[1:]:
            input_ids.append(tok)
        return False

    def extract_function_params(
        self,
        function_name: str,
        input_ids: list[int]
    ) -> dict[str, Any]:
        """Extract function parameters using constrained decoding."""
        input_ids.extend(
            self.model.encode('", "parameters": {').tolist()[0]
        )
        function = next(
            f for f in self.functions if f["name"] == function_name
        )
        params = list(function["parameters"].items())
        result: dict[str, Any] = {}

        for i, (param_name, param_info) in enumerate(params):
            if i > 0:
                input_ids.extend(self.model.encode(", ").tolist()[0])

            input_ids.extend(
                self.model.encode(f'"{param_name}":').tolist()[0]
            )
            ptype = param_info["type"]

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

        input_ids.extend(self.model.encode("}}").tolist()[0])
        return result
