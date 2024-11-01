import collections
import torch
import torch.nn.functional as F
from config import *
from encoder import Encoder
from modules import LLM

def infer():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 构建编码器
    encoder = Encoder.from_path("encoder.json")
    # 构建模型
    llm = LLM(encoder.vocab_size, MODEL_DIM, MAX_LENGTH, N_HEADS, N_BLOCKS, 0.0).to(DEVICE)
    path = PRETRAINED_STATE_DICT_PATH if not FINETUNE else FINETUNED_STATE_DICT_PATH
    llm.load_state_dict(torch.load(path, weights_only=True))
    llm.eval()
    # 编译模型并设置float32精度以提高推理速度
    torch.set_float32_matmul_precision('high')
    print("Compiling module")
    llm = torch.compile(llm)
    print("Compiled successfully")
    encoded_prompt = []
    with torch.no_grad():
        while True:
            try:
                prompt = input(">>> ")
                if not FINETUNE:
                    encoded_prompt = encoder.encode(prompt)
                else:
                    encoded_prompt += [
                        *encoder.encode("甲：" + prompt + "\n\n乙："),
                    ]
                x = torch.tensor(encoded_prompt).unsqueeze(0).to(DEVICE)
                while True:
                    try:
                        y = F.softmax(llm(x)[:, -1, :], dim=-1).squeeze()
                        probs, indices = torch.topk(y, 30, dim=-1)
                        token = indices[torch.multinomial(probs, 1)].unsqueeze(0)
                        x = torch.cat([x, token], dim=1)[:, -MAX_LENGTH:, ...]
                        code = int(token[0].item())
                        if not FINETUNE:
                            if code == SPECIAL_TOKENS_IDS["<eos>"]:
                                print()
                                break
                        else:
                            if code == encoder.encode("\n")[0]:
                                print()
                                encoded_prompt = x.squeeze().tolist() + encoder.encode("\n")
                                break
                        print(encoder.decode([code]), end="", flush=True)
                    except KeyboardInterrupt:
                        print()
                        break
            except EOFError:
                break
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    infer()
