import time
import multiprocessing
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from config import *
from dataloader_with_mask import BinaryDatasetWithMask, collate_fn_with_mask
from encoder import Encoder
from modules import LLM
from lr_schedule import get_schedule

def finetune(RANK: int, WORLD_SIZE: int, USE_DDP: bool):
    global FINETUNE_BATCH_SIZE
    # 设置进程内超参数和选项
    print(f"train({RANK}, {WORLD_SIZE}, {USE_DDP})")
    # 如果使用DDP，导入并初始化DDP
    if USE_DDP:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        dist.init_process_group(backend="nccl")
    DEVICE = f"cuda:{RANK}"
    torch.cuda.set_device(DEVICE)
    IS_MASTER = (RANK == 0)
    FINETUNE_BATCH_SIZE //= WORLD_SIZE
    if IS_MASTER:
        print(f"{FINETUNE_BATCH_SIZE} lines per batch.")
        print(f"{FINETUNE_N_BATCHES} batches per step.")

    encoder = Encoder.from_path("encoder.json")
    llm = LLM(encoder.vocab_size, MODEL_DIM, MAX_LENGTH, N_HEADS, N_BLOCKS, DROPOUT).to(DEVICE)
    print(f"{sum(para.numel() for para in llm.parameters()) / 1e6:.3f}M parameters.")
    # 如果有的话，加载检查点模型
    if PRETRAINED_STATE_DICT_PATH:
        llm.load_state_dict(torch.load(PRETRAINED_STATE_DICT_PATH, weights_only=True))
    # 微调时只训练后几个block
    if FINETUNE_N_BLOCKS is not None:
        for i in range(N_BLOCKS - FINETUNE_N_BLOCKS):
            for param in llm.blocks[i].parameters():
                param.requires_grad = False
    # 编译模型加快速度
    torch.set_float32_matmul_precision('high')
    print("Compiling module")
    llm.compile()
    print("Compiled successfully")
    # 如果使用DDP，将模型分布到各个显卡上
    if WORLD_SIZE > 1:
        llm = DDP(llm, device_ids=[RANK]) # type: ignore
    llm.train()
    optimizer = optim.AdamW(llm.parameters(), fused=True, weight_decay=0.0)
    schedule = get_schedule(FINETUNE_WARMUP_STEPS, FINETUNE_MAX_LEARINGRATE, FINETUNE_TARGET_STEPS, FINETUNE_MIN_LEARINGRATE)
    loader = DataLoader(
        BinaryDatasetWithMask(FINETUNE_DATA, FINETUNE_MASK, MAX_LENGTH),
        batch_size=FINETUNE_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn_with_mask,
    )

    print(f"{RANK + 1}/{WORLD_SIZE} start training.")
    start_time = time.time()
    step = FINETUNE_START_STEP
    microstep = 0
    total_microsteps = len(loader)
    torch.autograd.set_detect_anomaly(True) # 也许可以在梯度爆炸时发出警告
    for x, y, m, n_tokens in loader:
        if m.sum() < 20:
            continue # 跳过太短的batch
        if microstep % FINETUNE_N_BATCHES == 0: # 一次完整的学习的开始
            t0 = time.time()
            step += 1
            lr = schedule(step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            total_loss = 0.0
            total_tokens = 0
        
        if USE_DDP:                                                                                  # 多卡学习中，这是否是一次完整
            llm.require_backward_grad_sync = (microstep % FINETUNE_N_BATCHES == FINETUNE_N_BATCHES - 1) # type: ignore # 的学习中的最后一次反向传播？
                                                                                                     # 是则需要进行多卡同步
        t1 = time.time() # 开始一次反向传播

        x = x.to(DEVICE)
        y = y.to(DEVICE)
        m = m.to(DEVICE)
        res = llm(x)
        loss = (F.cross_entropy(
            res.view(-1, res.size(-1)),
            y.view(-1),
            reduction="none",
            ignore_index=SPECIAL_TOKENS_IDS["<pad>"]
        ) * m.view(-1)).sum() / m.sum() / FINETUNE_N_BATCHES

        del x, y, m, res
        loss.backward()
        total_loss += loss.detach().item()
        total_tokens += n_tokens.item()

        if IS_MASTER:
            print(f"{loss.item() * FINETUNE_N_BATCHES:.3f} {microstep % FINETUNE_N_BATCHES + 1}/{FINETUNE_N_BATCHES} {time.time() - t1:.3f}s/batch", end="\r")
        del loss # 结束一次反向传播

        microstep += 1
        if microstep % FINETUNE_N_BATCHES == 0: # 一次完整的学习的结束
            nn.utils.clip_grad_norm_(llm.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            llm.normalize()
            torch.cuda.empty_cache()

            progress = microstep / total_microsteps
            step_time = time.time() - t0
            total_time = step_time + t0 - start_time
            if IS_MASTER:
                open(FINETUNE_LOSSES_LOG_PATH, "a").write(f"step{step}: loss={total_loss} @ lr={lr}\n")
                print()
                print(f"step:{step} loss:{total_loss:.3f} lr:{lr:.8f} n_tokens:{total_tokens}")
                print(f"progress:{progress * 100:.3f}% {step_time:.3f}s/step {(step - FINETUNE_START_STEP) / progress * step_time - total_time:.3f}s to go")

                if step % 50 == 0:
                    save_model(llm, f"llm{step}_finetune_state_dict_{total_loss}.pt", USE_DDP)
                    print(f"Saved -> llm{step}_finetune_state_dict_{total_loss}.pt")

    if IS_MASTER:
        save_model(llm, f"llm{step}_finetune_state_dict_{total_loss}.pt", USE_DDP)
        print("Finetuning successfully ended.")
        
    if USE_DDP:
        dist.destroy_process_group()

def save_model(model: nn.Module, path: str, USE_DDP: bool):
    if USE_DDP:
        model = model.module
    model.save(path)

if __name__ == "__main__":
    import os
    DDP = os.environ.get("RANK", None) is not None
    if DDP: # 多GPU
        finetune(int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"]), True)
    else:
        finetune(0, 1, False)
