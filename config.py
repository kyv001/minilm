import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
MAX_LENGTH = 1024
MODEL_DIM = 768
N_HEADS = 12
N_BLOCKS = 20
DROPOUT = 0.1

TRAIN = True
BATCH_SIZE = 1
N_BATCHES = 256
WARMUP_STEPS = 10
MAX_LEARINGRATE = 6e-4
TARGET_STEPS = 2000
MIN_LEARINGRATE = 1e-4

# Pretraining data
NEWS_FILES = list(map(lambda x: "WanJuan-News/" + x, os.listdir("WanJuan-News/")))
WEBT_FILES = list(map(lambda x: "WanJuan-WebText/" + x, os.listdir("WanJuan-WebText/")))
PRETRAIN_DATA = NEWS_FILES[0]

SPECIAL_TOKENS = ["<pad>", "<eos>", "<ins>", "</ins>"]

SPECIAL_TOKENS_IDS = {
    "<pad>": 0,
    "<eos>": 1,
    "<ins>": 2,
    "</ins>": 3
}
SPECIAL_TOKENS_TENSORS = {
    token_name: torch.tensor(SPECIAL_TOKENS_IDS[token_name]).to(DEVICE)
    for token_name in SPECIAL_TOKENS_IDS.keys()
}

__all__ = [
    "DEVICE",
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
    "NEWS_FILES",
    "WEBT_FILES",
    "PRETRAIN_DATA",
    "SPECIAL_TOKENS",
    "SPECIAL_TOKENS_IDS",
    "SPECIAL_TOKENS_TENSORS"
]
