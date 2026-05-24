import sys
import os
import json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'llm_sdk'))
from llm_sdk import Small_LLM_Model
from src.parsing import Parser


class FunctionsDict:
    def __init__(self):
        self.parser = Parser()

    def functions_list(self, path: str):
        new_list_functions = []
        functions = self.parser.load_functions(path)

        for fun in functions:
            params = fun["parameters"]
            params_str = ", ".join(
                f"{k}: {v['type']}" for k, v in params.items()
            )

            line = (
                f"function_name: {fun['name']}({params_str}) "
                f":returns({fun['returns']['type']}) "
                f":description({fun['description']})"
            )

            new_list_functions.append(line)

        return new_list_functions

    def prompt_list(self, path: str):
        new_list_prompt = []
        prompts = self.parser.load_prompt(path)
        for prompt in prompts:
            line = (
                f"{prompt["prompt"]}"
            )
            new_list_prompt.append(line)
        return new_list_prompt
    
    def build_prompt(self, user_request: str, functions_list: list) -> str:
        functions_str = "\n".join(functions_list)
        return f"""You are a function calling assistant. \
            Your job is to select the correct function and extract the parameters from the user request.
            You must ONLY respond with a JSON object, nothing else.

            Available functions:
            {functions_str}

            Rules:
            - Choose the most appropriate function for the request
            - Extract the exact parameter values from the request
            - Return ONLY valid JSON, no explanation

            User request: {user_request}

            JSON response:
            {{\"function_name\": \""""



        