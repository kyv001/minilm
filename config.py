import torch
import os

# 模型超参数
MAX_LENGTH = 1024
MODEL_DIM = 768
N_HEADS = 8
N_BLOCKS = 12
""" # 如果你有足够的显卡和显存：
MAX_LENGTH = 2048
MODEL_DIM = 4096
N_HEADS = 32
N_BLOCKS = 32
"""
DROPOUT = 0.06

# 训练超参数
TRAIN = False
BATCH_SIZE = 2
N_BATCHES = 60
WARMUP_STEPS = 0
MAX_LEARINGRATE = 3e-4
TARGET_STEPS = 1000
MIN_LEARINGRATE = 1e-5

# 预训练数据路径（*.jsonl.bin）
PRETRAIN_DATA = "COIG-CQIA-full.jsonl.bin" # 这根本不是预训练数据而是微调数据，但是谁在乎呢
WITH_MASK = True

# 特殊token
SPECIAL_TOKENS = ["<pad>", "<eos>", "<ins>", "</ins>"]
SPECIAL_TOKENS_IDS = {
    "<pad>": 0,
    "<eos>": 1,
    "<ins>": 2,
    "</ins>": 3
}
SPECIAL_TOKENS_TENSORS: dict[str, torch.Tensor] = {
    token_name: torch.tensor(SPECIAL_TOKENS_IDS[token_name])
    for token_name in SPECIAL_TOKENS_IDS.keys()
}

# 检查点位置和属性
PRETRAINED_STATE_DICT_PATH = "llm373_state_dict_1.15620494261384.pt"
START_STEP = 373

# Loss数据记录文件
LOSSES_LOG_PATH = "losses.log"

__all__ = [
    "MAX_LENGTH",
    "MODEL_DIM",
    "N_HEADS",
    "N_BLOCKS",
    "DROPOUT",
    "TRAIN",
    "BATCH_SIZE",
    "N_BATCHES",
    "WARMUP_STEPS",
    "MAX_LEARINGRATE",
    "TARGET_STEPS",
    "MIN_LEARINGRATE",
    "PRETRAIN_DATA",
    "WITH_MASK",
    "SPECIAL_TOKENS",
    "SPECIAL_TOKENS_IDS",
    "SPECIAL_TOKENS_TENSORS",
    "PRETRAINED_STATE_DICT_PATH",
    "START_STEP",
    "LOSSES_LOG_PATH",
]
