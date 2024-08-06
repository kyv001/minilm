import torch
import os

# DEVICE = "cpu"
MAX_LENGTH = 800
MODEL_DIM = 256
N_HEADS = 4
N_BLOCKS = 8
DROPOUT = 0.1

TRAIN = False
BATCH_SIZE = 8
N_BATCHES = 4
WARMUP_STEPS = 0
MAX_LEARINGRATE = 6e-4
TARGET_STEPS = 3
MIN_LEARINGRATE = 6e-4
USE_TORCH2 = True # 如果安装了requirements.txt而不是requirements_old.txt，改为True
# 以上除DEVICE外皆为超参数

# Pretraining data
PRETRAIN_DATA = "tiny-example-news.jsonl.contents.txt.lines.txt.encoded.bin"

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

PRETRAINED_STATE_DICT_PATH = "llm513_state_dict_5.627837300300598.pt"

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
    "PRETRAIN_DATA",
    "SPECIAL_TOKENS",
    "SPECIAL_TOKENS_IDS",
    "SPECIAL_TOKENS_TENSORS",
    "LINE_SEP",
    "PRETRAINED_STATE_DICT_PATH",
]
