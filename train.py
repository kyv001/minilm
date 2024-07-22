import time
import multiprocessing
import torch
from torch import nn, optim
import torch.nn.functional as F
from config import *
from dataloader import WanJuanLoader
from encoder import Encoder
from modules import LLM
from lr_schedule import get_schedule

def train(RANK, WORLD_SIZE, DDP):
    global BATCH_SIZE
    print(f"train({RANK}, {WORLD_SIZE}, {DDP})")
    if DDP:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        dist.init_process_group(backend="nccl")
    DEVICE = f"cuda:{RANK}"
    torch.cuda.set_device(DEVICE)
    IS_MASTER = RANK == 0
    BATCH_SIZE //= WORLD_SIZE
    if IS_MASTER:
        print(f"{BATCH_SIZE} lines per batch.")
        print(f"{N_BATCHES} batches per step.")

    encoder = Encoder.from_path("encoder.json")
    llm = LLM(encoder.vocab_size, MODEL_DIM, MAX_LENGTH, N_HEADS, N_BLOCKS, DROPOUT, DEVICE).to(DEVICE)
    if WORLD_SIZE > 1:
        llm = DDP(llm, device_ids=[RANK])
    data_queue = multiprocessing.JoinableQueue()
    def load_data(fname, encoder, batch_size, max_length):
        loader = WanJuanLoader(fname, encoder, batch_size, max_length)
        while not loader.ended:
            data_queue.put((*loader.get_data(), loader.line, loader.total_lines))
        data_queue.put((0, 0, 0, 0, 0))
        print("\nData fully loaded.\n")
        data_queue.join()
    data_proc = multiprocessing.Process(
        target=load_data,
        name="Data Loader",
        args=(PRETRAIN_DATA[RANK], encoder, BATCH_SIZE, MAX_LENGTH)
    )
    data_proc.start()
    optimizer = optim.AdamW(llm.parameters())# , fused=True) # torch 2+
    schedule = get_schedule(WARMUP_STEPS, MAX_LEARINGRATE, TARGET_STEPS, MIN_LEARINGRATE)
    step = 0
    if PRETRAINED_STATE_DICT_PATH:
        state_dict = torch.load(PRETRAINED_STATE_DICT_PATH)
        llm.load_state_dict(state_dict)
        del state_dict
    llm.train()
    ended = False
    print(f"{RANK + 1}/{WORLD_SIZE} start training.")
    start_time = time.time()
    while not ended:
        t0 = time.time()
        step += 1

        lr = schedule(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        total_loss = 0
        for i in range(N_BATCHES):
            llm.require_backward_grad_sync = (i == N_BATCHES - 1)
            t1 = time.time()
            x, y, n_tokens, current_line, total_lines = data_queue.get()
            data_queue.task_done()
            if isinstance(x, int):
                ended = True
                step -= 1
                break
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            res = llm(x)
            try:
                loss = F.cross_entropy(res.view(-1, res.size(-1)), y.view(-1), reduction="sum") / n_tokens / N_BATCHES
            except RuntimeError:
                break
            loss.backward()
            total_loss += loss.item()
            if IS_MASTER:
                print(f"{loss.item() * N_BATCHES:.3f} {i + 1}/{N_BATCHES} {time.time() - t1:.3f}s/batch", end="\r")
            del x, y, res, loss
        else:
            nn.utils.clip_grad_norm_(llm.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            progress = current_line / total_lines
            step_time = time.time() - t0
            total_time = step_time + t0 - start_time
            if IS_MASTER:
                print()
                print(f"step:{step} loss:{total_loss:.3f} lr:{lr:.8f}")
                print(f"progress:{progress * 100:.3f}% {step_time:.3f}s/step {step / progress * step_time - total_time:.3f}s to go")

                if step % 20 == 0:
                    torch.save(llm.state_dict(), f"llm{step}_state_dict{total_loss}.pt")
                    print(f"Saved -> llm{step}_state_dict_{total_loss}.pt")
    if IS_MASTER:
        torch.save(llm.state_dict(), f"llm{step}_state_dict_{total_loss}.pt")
        print("Training successfully ended.")

    dist.destroy_process_group()
