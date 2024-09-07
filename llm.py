import collections
import torch
import torch.nn.functional as F
from config import *
from encoder import Encoder
from modules import LLM
from train import train

if __name__ == "__main__":
    import os
    if TRAIN:
        DDP = os.environ.get("RANK", None) is not None
        if DDP: # 多GPU
            train(int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"]), True)
        else:
            train(0, 1, False)
    else:
        DEVICE = "cuda"
        encoder = Encoder.from_path("encoder.json")
        llm = LLM(encoder.vocab_size, MODEL_DIM, MAX_LENGTH, N_HEADS, N_BLOCKS, 0).to(DEVICE)
        # 加载预训练模型
        module_state_dict = torch.load(PRETRAINED_STATE_DICT_PATH)
        llm.load_state_dict(module_state_dict)
        llm.eval()
        # 编译模型并设置float32精度以提高推理速度
        torch.set_float32_matmul_precision('high')
        print("Compiling module")
        llm = torch.compile(llm) # torch 2+
        print("Compiled successfully")
        with torch.no_grad():
            while True:
                try:
                    prompt = input(">>> ")
                    # encoded_prompt = [SPECIAL_TOKENS_IDS["<ins>"], *encoder.encode(prompt), SPECIAL_TOKENS_IDS["</ins>"]]
                    encoded_prompt = encoder.encode(prompt)
                    x = torch.tensor(encoded_prompt).unsqueeze(0).to(DEVICE)
                    while True:
                        try:
                            y = F.softmax(llm(x)[:, -1, :], dim=-1).squeeze()
                            probs, indices = torch.topk(y, 30, dim=-1)
                            token = indices[torch.multinomial(probs, 1)].unsqueeze(0)
                            # token = torch.argmax(y.squeeze()).unsqueeze(0).unsqueeze(0)
                            x = torch.cat([x, token], dim=1)[:, -MAX_LENGTH:, ...]
                            code = int(token[0].item())
                            if code == SPECIAL_TOKENS_IDS["<eos>"]:
                                print()
                                break
                            print(encoder.decode([code]), end="", flush=True)
                        except KeyboardInterrupt:
                            print()
                            break
                except EOFError:
                    break
                except KeyboardInterrupt:
                    break

