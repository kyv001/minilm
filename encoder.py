import json
import tqdm

class Encoder:
    ## TODO: Implement BPE
    def __init__(self, vocab: list):
        self.vocab = sorted(vocab, key=lambda x: len(x), reverse=True)
        self.vocab_size = len(vocab)
        self.max_token_length = max(len(w) for w in vocab)

    def encode(self, string: str) -> list:
        l = string.split(" ") # 保留\n、\r等特殊空白字符
        codes = []
        for w in l:
            w += "</w>"
            while w:
                window = w[:self.max_token_length]
                matches = [i for i, v in enumerate(self.vocab) if window.startswith(v)]
                if not matches:
                    codes.append(self.vocab_size)
                    w = w[1:]
                else:
                    codes.append(matches[0])
                    w = w[len(self.vocab[matches[0]]):]
        return codes

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
        l = (s
            .replace("\n", " \n ")
            .replace("\r", " \r ")
            .replace("\t", " \t ")
            .split(" ")) # 保留\n、\r等特殊空白字符
        f: list[tuple[list[str], int]] = []
        for w in set(l):
            f.append((list(w) + ["</w>"], l.count(w)))
        
        vocab_set = set()
        while True:
            # 计算序列对频率
            pairs: dict[tuple[str, str], int] = {}
            for i in range(len(f)):
                wl, c = f[i]
                if len(wl) == 1:
                    continue
                for j in range(len(wl) - 1):
                    p = wl[j], wl[j+1]
                    pairs[p] = pairs.get(p, 0) + c

            # 如果已合并完毕则退出循环
            if not pairs:
                break

            # 选出频率最高的序列对
            merged_p, merged_c = sorted(list(pairs.items()), key=lambda x: x[1], reverse=True)[0]

            # 合并最高频率的序列对
            new_f = []
            for i in range(len(f)):
                wl, c = f[i]
                if len(wl) == 1:
                    new_f.append((wl, c))
                    continue
                new_wl = []
                j = 0
                while j < len(wl):
                    if tuple(wl[j:j+2]) == merged_p:
                        new_wl.append(merged_p[0] + merged_p[1])
                        j += 2
                    else:
                        new_wl.append(wl[j])
                        j += 1
                new_f.append((new_wl, c))
            f = new_f

            # 更新词表
            for wl, _ in f:
                vocab_set.update(wl)
            vocab_size = len(vocab_set)

            # 达到目标vocab大小则退出循环
            print(f"{vocab_size / target_vocab_size * 100:.2f}%", end="\r", flush=True)
            if vocab_size >= target_vocab_size:
                print()
                break
        
        vocab = given_tokens + list(vocab_set)[:target_vocab_size - len(given_tokens)]
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
    with open(path, "r") as f:
        s = f.read().replace("\n\n", "\n")
    with open("hanzi.txt", "r") as f:
        hanzi = list(f.read())
    encoder = Encoder.from_string(s, SPECIAL_TOKENS + hanzi, target_vocab_size)
    encoder.save("encoder.json")
