import torch
import os

# DEVICE = "cpu"
MAX_LENGTH = 1500
MODEL_DIM = 768
N_HEADS = 12
N_BLOCKS = 16
DROPOUT = 0.1

TRAIN = True
BATCH_SIZE = 2
N_BATCHES = 100
WARMUP_STEPS = 1000
MAX_LEARINGRATE = 6e-4
TARGET_STEPS = 600000
MIN_LEARINGRATE = 6e-5
USE_TORCH2 = True # 如果安装了requirements.txt而不是requirements_old.txt，改为True
# 以上除DEVICE外皆为超参数

# Pretraining data
NEWS_FILES = []
for name in os.listdir("WanJuan-News/"):
    if not name.endswith(".parts"):
        NEWS_FILES.append("WanJuan-News/" + name)
WEBT_FILES = []
for name in os.listdir("WanJuan-WebText/"):
    if not name.endswith(".parts"):
        WEBT_FILES.append("WanJuan-WebText/" + name)
PRETRAIN_DATA = NEWS_FILES

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

PRETRAINED_STATE_DICT_PATH = ""

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
    "PRETRAINED_STATE_DICT_PATH"
]
