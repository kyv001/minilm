import torch
import os

# 模型超参数
MAX_LENGTH = 1024
MODEL_DIM = 1024
N_HEADS = 8
N_BLOCKS = 12
""" # 如果你有足够的显卡和显存：
MAX_LENGTH = 2048
MODEL_DIM = 4096
N_HEADS = 32
N_BLOCKS = 32
"""
DROPOUT = 0.02

# 训练超参数
TRAIN = True
BATCH_SIZE = 8
N_BATCHES = 50
WARMUP_STEPS = 100
MAX_LEARINGRATE = 6e-4
TARGET_STEPS = 10000
MIN_LEARINGRATE = 6e-5

# 预训练数据路径（*.jsonl.bin）
PRETRAIN_DATA = "WanJuan-News/part-006853-a894b46e.jsonl.bin"

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
PRETRAINED_STATE_DICT_PATH = None
START_STEP = 0

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
    "SPECIAL_TOKENS",
    "SPECIAL_TOKENS_IDS",
    "SPECIAL_TOKENS_TENSORS",
    "PRETRAINED_STATE_DICT_PATH",
    "START_STEP",
    "LOSSES_LOG_PATH",
]
