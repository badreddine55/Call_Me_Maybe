import sys
import os
import json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'llm_sdk'))
from llm_sdk import Small_LLM_Model
from src.parsing import Parser
from itertools import zip_longest

class FunctionsDict:
    def __init__(self):
        self.parser = Parser()
        self.model = Small_LLM_Model()
        self.functions = None

    def functions_list(self, path: str):
        new_list_functions = []
        functions_names = []
        self.functions = self.parser.load_functions(path)

        for fun in self.functions:
            params = fun["parameters"]
            params_str = ", ".join(
                f"{k}: {v['type']}" for k, v in params.items()
            )
            functions_names.append(self.model.encode(fun['name']).tolist()[0])
            line = (
                f"function_name: {fun['name']}({params_str}) "
                f":returns({fun['returns']['type']}) "
                f":description({fun['description']})"
            )

            new_list_functions.append(line)

        return (new_list_functions, functions_names)

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


    def get_logits(self, input_ids):
        return self.model.get_logits_from_input_ids(input_ids)

    def extract_function_name(self, list_func_tokens, input_ids):
        expected_fun = []

        for token_group in zip_longest(*list_func_tokens):
            allowed_tokens = set(
                x for x in token_group
                if x is not None
            )

            if len(allowed_tokens) == 1:
                token = next(iter(allowed_tokens))
            else:
                logits = self.get_logits(input_ids)

                for token_id in range(len(logits)):
                    if token_id not in allowed_tokens:
                        logits[token_id] = -float('inf')

                token = np.argmax(logits)

            expected_fun.append(token)
            input_ids.append(token)

            # break if exact match found
            if expected_fun in list_func_tokens:
                break

        return expected_fun
    def extract_function_params(self, function_name, input_ids):
        input_ids.extend(self.model.encode('", "parameters": {').tolist()[0])
        function = next(f for f in self.functions if f["name"] == function_name)
        parameters = function["parameters"]
        params = list(parameters.items())

        for i, (param_name, param_info) in enumerate(params):
            last_param = (i == len(params) - 1)

            if i > 0:
                input_ids.extend(self.model.encode(", ").tolist()[0])

            input_ids.extend(self.model.encode(f'"{param_name}": ').tolist()[0])

            if param_info["type"] == "string":
                input_ids.extend(self.model.encode('"').tolist()[0])
                while True:
                    logits = self.get_logits(input_ids)
                    token = int(np.argmax(logits))
                    token_text = self.model.decode([token])
                    if '"' in token_text:
                        break
                    input_ids.append(token)

                input_ids.extend(self.model.encode('"').tolist()[0])

            elif param_info["type"] == "number":
                while True:
                    logits = self.get_logits(input_ids)
                    token = int(np.argmax(logits))
                    token_text = self.model.decode([token])
                    if any(c in token_text for c in {",", "}"}):
                        digit_part = token_text.split(",")[0].split("}")[0].strip()
                        if digit_part:
                            input_ids.extend(self.model.encode(digit_part).tolist()[0])
                        break
                    input_ids.append(token)

        input_ids.extend(self.model.encode("}}").tolist()[0])
        print(self.model.decode(input_ids))
        return input_ids

    



        