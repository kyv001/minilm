def encode(vocab: list[str], string: str, unk: int) -> list:
    codes = []
    for char in string:
        for i in range(len(vocab)):
            if vocab[i] == char:
                codes.append(i)
                break
        else:
            codes.append(unk)
    return codes

def decode(vocab_dict: dict, codes: list[int]) -> str:
    string = ""
    for code in codes:
        string += vocab_dict.get(code, "")
    return string