import json

class Encoder:
    def __init__(self, vocab: list, special_tokens: list):
        self.vocab = special_tokens + vocab + [""]
        self.unk = len(self.vocab) - 1
        self.vocab_size = len(self.vocab)

    def encode(self, string: str) -> list:
        codes = []
        for char in string:
            for i in range(len(self.vocab)):
                if self.vocab[i] == char:
                    codes.append(i)
                    break
            else:
                codes.append(self.unk)
        return codes

    def decode(self, codes: list) -> str:
        string = ""
        for code in codes:
            if 0 <= code <= self.vocab_size - 1:
                string += self.vocab[code]
        return string

    def dump(self):
        parameters = {
            "vocab": self.vocab
        }
        return json.dumps(parameters)

    @classmethod
    def from_string(cls, s: str, special_tokens: list):
        return cls(vocab=list(set(list(s))), special_tokens=special_tokens)

    @classmethod
    def from_path(cls, path):
        parameters = json.load(open(path))
        return cls(vocab=parameters["vocab"], special_tokens=[])

if __name__ == "__main__": # 构建编码器
    """
    # learning
    import json, tqdm, pickle, config
    with open("WanJuan-WebText/part-000036-a894b46e.jsonl") as f:
        lines = f.readlines()[:5000]
    s = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()`~-_=+[{]}\\|;:'\",<.>/?]\n"
    for line in tqdm.tqdm(lines):
        s += json.loads(line)["content"]
    encoder = Encoder.from_string(s, special_tokens=config.SPECIAL_TOKENS)
    print(codes := encoder.encode("这是一个测试"))
    print(encoder.decode(codes))
    with open("encoder.json", "w") as f:
        f.write(encoder.dump())
        print("Saved to encoder.json")
    print("Loading encoder from encoder.json")
    encoder = Encoder.from_path("encoder.json")
    """
    encoder = Encoder.from_path("encoder.json")
    codes = encoder.encode("这是一个测试")
    print(codes)
    print(encoder.decode(codes + [1]))
    print(encoder.vocab_size)

