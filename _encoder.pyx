def build_vocab(s: str, target_vocab_size: int) -> set[str]:
    l = (s
            .replace("\n", " \n ")
            .replace("\r", " \r ")
            .replace("\t", " \t ")
            .split(" ")) # 保留\n、\r等特殊空白字符
    f: list[tuple[list[str], int]] = []
    for w in set(l):
        f.append((list(w) + ["</w>"], l.count(w)))
    
    vocab_set: set[str] = set()
    cdef int j = 0
    while True:
        # 计算序列对频率
        pairs: dict[tuple[str, str], int] = {}
        for wl, c in f:
            if len(wl) == 1:
                continue
            for j in range(len(wl) - 1):
                p = wl[j], wl[j+1]
                pairs[p] = pairs.get(p, 0) + c

        # 如果已合并完毕则退出循环
        if not pairs:
            break

        # 选出频率最高的序列对
        merged_p = max(pairs, key=lambda x: pairs[x])

        # 合并最高频率的序列对
        new_f = []
        for wl, c in f:
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
        for wl, c in f:
            vocab_set.update(wl)
        vocab_size = len(vocab_set)

        # 达到目标vocab大小则退出循环
        print(f"{vocab_size / target_vocab_size * 100:.2f}%", end="\r", flush=True)
        if vocab_size >= target_vocab_size:
            print()
            break
    
    return vocab_set

def encode(vocab: list[str], vocab_size: int, max_token_length: int, s: str):
    l = s.split(" ") # 保留\n、\r等特殊空白字符
    codes = []
    for w in l:
        w += "</w>"
        while w:
            window = w[:max_token_length]
            matches = [i for i, v in enumerate(vocab) if window.startswith(v)]
            if not matches:
                codes.append(vocab_size)
                w = w[1:]
            else:
                codes.append(matches[0])
                w = w[len(vocab[matches[0]]):]
    return codes
