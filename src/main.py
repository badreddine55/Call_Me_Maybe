"""Main entry point for the function calling assistant."""

import os
import sys
import json
import argparse
from typing import Any
from src.parsing import ParseError
from src.logic_file import FunctionsDict


class Main:
    """Main class for the function calling assistant."""

    DEFAULT_FUNCTIONS = "data/input/functions_definition.json"
    DEFAULT_INPUT = "data/input/function_calling_tests.json"
    DEFAULT_OUTPUT = "data/output/output.json"

    def __init__(self) -> None:
        """Initialize with FunctionsDict and parsed arguments."""
        self.args = self._parse_args()
        self.fun = FunctionsDict()

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

    def _setup_output(self) -> bool:
        """Create output directory if needed."""
        try:
            os.makedirs(
                os.path.dirname(self.args.output), exist_ok=True
            )
            return True
        except OSError as e:
            print(f"ERROR: Could not create output directory: {e}")
            return False

    def _save(self, results: list[Any]) -> bool:
        """Write results to output JSON file."""
        try:
            with open(self.args.output, "w") as f:
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
        input_ids = self.fun.model.encode(
            self.fun.build_prompt(prompt, list_func)
        ).tolist()[0]

        function_name_tokens = self.fun.extract_function_name(
            functions_names, input_ids
        )

        if function_name_tokens is None:
            print(f"ERROR: No matching function for: '{prompt}'")
            return None

        function_name = self.fun.model.decode(function_name_tokens)

        dict_prompt = self.fun.extract_function_params(
            function_name, input_ids
        )

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
                self.args.functions_definition
            )
        except ParseError as e:
            print(f"ERROR loading functions: {e}")
            sys.exit(1)

        try:
            prompts = self.fun.prompt_list(self.args.input)
        except ParseError as e:
            print(f"ERROR loading prompts: {e}")
            sys.exit(1)

        results: list[Any] = []

        for prompt in prompts:
            try:
                result = self._process_prompt(
                    prompt, list_func, functions_names
                )
                if result is None:
                    continue
                print(result)
                results.append(result)

            except Exception as e:
                print(f"ERROR processing prompt '{prompt}': {e}")
                continue

            if not self._save(results):
                sys.exit(1)

    def load_tokenizer(self):
        pass


if __name__ == "__main__":
    Main().run()
