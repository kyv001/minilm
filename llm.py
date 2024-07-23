import torch
from torch import nn, optim
import torch.nn.functional as F
from config import *
from dataloader import WanJuanLoader
from encoder import Encoder
from modules import LLM
from lr_schedule import get_schedule
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
        llm = LLM(encoder.vocab_size, MODEL_DIM, MAX_LENGTH, N_HEADS, N_BLOCKS, DROPOUT, DEVICE).to(DEVICE)
        state_dict = torch.load(PRETRAINED_STATE_DICT_PATH)
        llm.load_state_dict(state_dict)
        llm.eval()
        with torch.no_grad():
            while True:
                try:
                    prompt = input(">>> ")
                    x = torch.tensor(encoder.encode(prompt)).unsqueeze(0).to(DEVICE)
                    while True:
                        try:
                            y = F.softmax(llm(x)[:, -1, :], dim=-1)
                            probs, indices = torch.topk(y, 15, dim=-1)
                            token = torch.multinomial(probs, 1)
                            x = torch.cat([x, token], dim=1)
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

