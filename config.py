import torch
import os

# DEVICE = "cpu"
MAX_LENGTH = 1500
MODEL_DIM = 768
N_HEADS = 12
N_BLOCKS = 16
DROPOUT = 0.1

TRAIN = False
BATCH_SIZE = 8
N_BATCHES = 24
WARMUP_STEPS = 100
MAX_LEARINGRATE = 6e-4
TARGET_STEPS = 30000
MIN_LEARINGRATE = 6e-5
USE_TORCH2 = True # 如果安装了requirements.txt而不是requirements_old.txt，改为True
# 以上除DEVICE外皆为超参数

# Pretraining data
PRETRAIN_DATA = "WanJuan-News/part-006853-a894b46e.jsonl.contents.txt.lines.txt.encoded.bin"

SPECIAL_TOKENS = ["<pad>", "<eos>", "<ins>", "</ins>"]

SPECIAL_TOKENS_IDS = {
    "<pad>": 0,
    "<eos>": 1,
    "<ins>": 2,
    "</ins>": 3
}
SPECIAL_TOKENS_TENSORS = {
    token_name: torch.tensor(SPECIAL_TOKENS_IDS[token_name])
    for token_name in SPECIAL_TOKENS_IDS.keys()
}
LINE_SEP = chr(9000)

PRETRAINED_STATE_DICT_PATH = None

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
    "USE_TORCH2",
    "NEWS_FILES",
    "WEBT_FILES",
    "PRETRAIN_DATA",
    "SPECIAL_TOKENS",
    "SPECIAL_TOKENS_IDS",
    "SPECIAL_TOKENS_TENSORS",
    "LINE_SEP",
    "PRETRAINED_STATE_DICT_PATH",
]
