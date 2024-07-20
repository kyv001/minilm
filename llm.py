import time
import torch
from torch import nn, optim
import torch.nn.functional as F
from config import *
from dataloader import WanJuanLoader
from encoder import Encoder
from modules import LLM
from lr_schedule import get_schedule

torch.set_float32_matmul_precision('high')
encoder = Encoder.from_path("encoder.json")
llm = LLM(encoder.vocab_size, MODEL_DIM, MAX_LENGTH, N_HEADS, N_BLOCKS, DROPOUT).to(DEVICE)
print("Compiling module")
llm = torch.compile(llm)
print("Compiled successfully")
if TRAIN:
    loader = WanJuanLoader(PRETRAIN_DATA, encoder)
    optimizer = optim.AdamW(llm.parameters(), fused=True)
    schedule = get_schedule(WARMUP_STEPS, MAX_LEARINGRATE, TARGET_STEPS, MIN_LEARINGRATE)
    step = 0
    if PRETRAINED_STATE_DICT_PATH:
        state_dict = torch.load(PRETRAINED_STATE_DICT_PATH)
        llm.load_state_dict(state_dict)
        del state_dict
    llm.train()
    start_time = time.time()
    while not loader.ended:
        t0 = time.time()
        step += 1

        lr = schedule(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        total_loss = 0
        optimizer.zero_grad()
        for i in range(N_BATCHES):
            t1 = time.time()
            x, y, n_tokens = loader.get_data(BATCH_SIZE, MAX_LENGTH)
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            res = llm(x)
            try:
                loss = F.cross_entropy(res.view(-1, res.size(-1)), y.view(-1), reduction="sum") / n_tokens / N_BATCHES
            except RuntimeError:
                break
            loss.backward()
            total_loss += loss.item()
            print(f"{loss.item() * N_BATCHES:.3f} {i + 1}/{N_BATCHES} {time.time() - t1:.3f}s/batch", end="\r")
            del x, y, res, loss
        print()

        nn.utils.clip_grad_norm_(llm.parameters(), 1.0)
        optimizer.step()

        progress = loader.line / loader.total_lines
        step_time = time.time() - t0
        total_time = step_time + t0 - start_time
        print(f"step:{step} loss:{total_loss:.3f} lr:{lr:.8f}")
        print(f"progress:{progress * 100:.3f}% {step_time:.3f}s/step {step / progress * step_time - total_time:.3f}s to go")

        if step % 20 == 0:
            torch.save(llm.state_dict(), f"llm{step}_state_dict{total_loss}.pt")
            print(f"Saved -> llm{step}_state_dict_{total_loss}.pt")
    torch.save(llm.state_dict(), f"llm{step + 1000}_state_dict_{total_loss}.pt")
    print("Training successfully ended.")
else:
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

