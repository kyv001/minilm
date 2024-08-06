def encode(vocab_dict: dict, string: str, unk: int) -> list:
    codes = []
    for char in string:
        codes.append(vocab_dict.get(char, unk))
    return codes

def decode(vocab: list[str], codes: list[int]) -> str:
    string = ""
    for code in codes:
        if 0 < code < len(vocab):
            string += vocab[code]
    return string