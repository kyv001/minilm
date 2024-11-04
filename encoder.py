import json
import tqdm
try:
    from _encoder import *
except ImportError:
    print("Cython模块尚未编译，编译方法见setup_encoder.sh。")
    exit(1)

class Encoder:
    ## TODO: Implement BPE
    def __init__(self, vocab: list):
        self.vocab = sorted(vocab, key=lambda x: len(x), reverse=True)
        self.vocab_size = len(vocab)
        self.max_token_length = max(len(w) for w in vocab)

    def encode(self, string: str) -> list:
        return encode(self.vocab, self.vocab_size, self.max_token_length, string)

    def decode(self, codes: list) -> str:
        s = ""
        for c in codes:
            if c == self.vocab_size:
                continue
            v = self.vocab[c]
            s += v[:-4] + " " if v[-4:].endswith("</w>") else v
        return s

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"vocab": self.vocab}, f)

    @classmethod
    def from_string(cls, s: str, given_tokens: list, target_vocab_size: int):
        assert len(given_tokens) < target_vocab_size, "给定的词表不能超过目标词表大小"
        vocab_set = build_vocab(s, target_vocab_size - len(given_tokens))
        vocab = given_tokens + list(vocab_set)
        return cls(vocab)

    @classmethod
    def from_path(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        return cls(data["vocab"])

if __name__ == '__main__':
    import sys
    from config import *
    if len(sys.argv) < 3:
        print("Usage: python encoder.py <path> <target_vocab_size>")
        exit(1)
    path = sys.argv[1]
    target_vocab_size = int(sys.argv[2])
    print(f"从{path}构建{target_vocab_size}大小的词表……")
    print("-->读取数据中……")
    with open(path, "r") as f:
        s = f.read().replace("\n\n", "\n")
    with open("hanzi.txt", "r") as f:
        hanzi = list(f.read())
    print("-->构建词表中……")
    encoder = Encoder.from_string(s, SPECIAL_TOKENS + hanzi, target_vocab_size)
    encoder.save("encoder.json")
