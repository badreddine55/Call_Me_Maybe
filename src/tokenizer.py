from src.logic_file import FunctionsDict


class Tokenizer():
    def __init__(self):
        self.fun = FunctionsDict()
        vocab_path = self.fun.model.get_path_to_vocab_file
        print(vocab_path)

    def my_encode(text: str) -> list[int]:
        pass

    def my_decode(token_ids: list[int]) -> str:
        pass


Tc = Tokenizer()
