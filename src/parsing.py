import json

class Parser:
    def __init__(self):
        pass

    def load_functions(self, path: str) -> list:
        with open(path, 'r') as f:
            return json.load(f)

    def load_prompt(self, path: str) -> list:
        with open(path, 'r') as f:
            return json.load(f)