"""Main entry point for the function calling assistant."""

import os
import sys
import json
import argparse
from typing import Any
from pydantic import BaseModel, field_validator
from src.parsing import ParseError
from src.logic_file import FunctionsDict
from src.tokenizer import Tokenizer


class MainConfig(BaseModel):
    """Pydantic model for validating main configuration."""

    functions_definition: str
    input: str
    output: str

    @field_validator("functions_definition", "input")
    @classmethod
    def must_be_json(cls, v: str) -> str:
        """Validate that the path ends with .json."""
        if not v.endswith(".json"):
            raise ValueError(f"Expected a .json file, got: {v}")
        return v

    @field_validator("output")
    @classmethod
    def output_must_be_json(cls, v: str) -> str:
        """Validate that the output path ends with .json."""
        if not v.endswith(".json"):
            raise ValueError(f"Output must be a .json file, got: {v}")
        return v


class Main:
    """Main class for the function calling assistant."""

    DEFAULT_FUNCTIONS: str = "data/input/functions_definition.json"
    DEFAULT_INPUT: str = "data/input/function_calling_tests.json"
    DEFAULT_OUTPUT: str = "data/output/function_calling_results.json"

    def __init__(self) -> None:
        """Initialize with FunctionsDict and parsed arguments."""
        self.args: argparse.Namespace = self._parse_args()
        self.config: MainConfig = self._validate_config()
        self.fun: FunctionsDict = FunctionsDict()
        self.my_decode: Any = None
        self.my_encode: Any = None

    def _parse_args(self) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description="Function calling assistant"
        )
        parser.add_argument(
            "--functions_definition",
            default=self.DEFAULT_FUNCTIONS,
            help="Path to function definitions JSON file"
        )
        parser.add_argument(
            "--input",
            default=self.DEFAULT_INPUT,
            help="Path to input prompts JSON file"
        )
        parser.add_argument(
            "--output",
            default=self.DEFAULT_OUTPUT,
            help="Path to output JSON file"
        )
        return parser.parse_args()

    def _validate_config(self) -> MainConfig:
        """Validate parsed arguments using pydantic."""
        try:
            return MainConfig(
                functions_definition=self.args.functions_definition,
                input=self.args.input,
                output=self.args.output
            )
        except Exception as e:
            raise ParseError(f"Invalid configuration: {e}") from e

    def _setup_output(self) -> bool:
        """Create output directory if needed."""
        try:
            os.makedirs(
                os.path.dirname(self.config.output), exist_ok=True
            )
            return True
        except OSError as e:
            print(f"ERROR: Could not create output directory: {e}")
            return False

    def _save(self, results: list[Any]) -> bool:
        """Write results to output JSON file."""
        try:
            with open(self.config.output, "w") as f:
                json.dump(results, f, indent=4)
            return True
        except OSError as e:
            print(f"ERROR writing output: {e}")
            return False

    def _process_prompt(
        self,
        prompt: str,
        list_func: list[str],
        functions_names: list[list[int]]
    ) -> dict[str, Any] | None:
        """Process a single prompt and return result or None on error."""
        try:
            input_ids: list[int] = self.my_encode(
                self.fun.build_prompt(prompt, list_func)
            )
        except Exception as e:
            raise ParseError(f"Failed to encode prompt: {e}") from e

        try:
            function_name_tokens: list[int] | None = (
                self.fun.extract_function_name(functions_names, input_ids)
            )
        except Exception as e:
            raise ParseError(
                f"Failed to extract function name: {e}"
            ) from e

        if function_name_tokens is None:
            print(f"ERROR: No matching function for: '{prompt}'")
            return None

        try:
            function_name: str = self.my_decode(function_name_tokens)
        except Exception as e:
            raise ParseError(
                f"Failed to decode function name: {e}"
            ) from e

        try:
            dict_prompt: dict[str, Any] = self.fun.extract_function_params(
                function_name, input_ids
            )
        except Exception as e:
            raise ParseError(f"Failed to extract parameters: {e}") from e

        return {
            "prompt": prompt,
            "name": function_name,
            "parameters": dict_prompt
        }

    def run(self) -> None:
        """Run the function calling assistant."""
        if not self._setup_output():
            sys.exit(1)

        try:
            list_func, functions_names = self.fun.functions_list(
                self.config.functions_definition
            )
        except ParseError as e:
            print(f"ERROR loading functions: {e}")
            sys.exit(1)

        try:
            prompts: list[str] = self.fun.prompt_list(self.config.input)
        except ParseError as e:
            print(f"ERROR loading prompts: {e}")
            sys.exit(1)

        results: list[Any] = []

        for prompt in prompts:
            try:
                result: dict[str, Any] | None = self._process_prompt(
                    prompt, list_func, functions_names
                )
                if result is None:
                    continue
                print(result)
                results.append(result)
            except ParseError as e:
                print(f"ERROR processing prompt '{prompt}': {e}")
                continue

            if not self._save(results):
                sys.exit(1)

    def load_tokenizer(self) -> None:
        """Load tokenizer encode and decode methods."""
        try:
            tc: Tokenizer = Tokenizer(self.fun)
            self.my_decode = tc.my_decode
            self.my_encode = tc.my_encode
            # inputs_1 = self.my_encode("hello this is test €")
            # inputs_2 = self.fun.model.encode(
            # "hello this is test €").tolist()[0]
            # print(inputs_1 == inputs_2)
        except Exception as e:
            raise ParseError(f"Failed to load tokenizer: {e}") from e


if __name__ == "__main__":
    try:
        main = Main()
        main.fun.load_tokenizer()
        main.load_tokenizer()
        main.run()
    except (ParseError, KeyboardInterrupt) as e:
        print(f"ERROR: Keyboard Interrupt or {e}")
        sys.exit(1)
