import torch
import os
from encoder import Encoder

# 模型超参数
MAX_LENGTH = 1024
MODEL_DIM = 1024
N_HEADS = 8
N_BLOCKS = 16
""" # 如果你有足够的显卡和显存：
MAX_LENGTH = 2048
MODEL_DIM = 4096
N_HEADS = 32
N_BLOCKS = 32
"""
DROPOUT = 0.06

# 训练超参数
TRAIN = False
BATCH_SIZE = 1
N_BATCHES = 200
WARMUP_STEPS = 4000
MAX_LEARINGRATE = 5e-4
TARGET_STEPS = 1000000
MIN_LEARINGRATE = 5e-5

# 预训练数据路径（*.jsonl.bin）
PRETRAIN_DATA = "openwebtext/openwebtext-5000lines.txt.bin"
# 微调数据路径（*.bin）
FINETUNE_DATA = "finetune.bin"
FINETUNE = False
N_FINETUNE_BLOCKS = 10 # 防止爆显存
SYS_PROMPT = ""

# 特殊token
encoder = Encoder.from_path("encoder.json")
SPECIAL_TOKENS = ["<pad>", "<eos>", "<ins>", "</ins>"]
SPECIAL_TOKENS_IDS = {token_name: encoder.encode(token_name)[0] for token_name in SPECIAL_TOKENS}
SPECIAL_TOKENS_TENSORS: dict[str, torch.Tensor] = {
    token_name: torch.tensor(SPECIAL_TOKENS_IDS[token_name])
    for token_name in SPECIAL_TOKENS_IDS.keys()
}

# 检查点位置和属性
PRETRAINED_STATE_DICT_PATH = None
FINETUNED_STATE_DICT_PATH = None
START_STEP = 0

# Loss数据记录文件
LOSSES_LOG_PATH = "losses_finetune.log"

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
    "FINETUNE_DATA",
    "FINETUNE",
    "N_FINETUNE_BLOCKS",
    "SYS_PROMPT",
    "SPECIAL_TOKENS",
    "SPECIAL_TOKENS_IDS",
    "SPECIAL_TOKENS_TENSORS",
    "PRETRAINED_STATE_DICT_PATH",
    "FINETUNED_STATE_DICT_PATH",
    "START_STEP",
    "LOSSES_LOG_PATH",
]
