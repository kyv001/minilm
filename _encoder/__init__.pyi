def encode(vocab: list[str], string: str, unk: int) -> list: ...
def decode(vocab_dict: dict, codes: list[int]) -> str: ...