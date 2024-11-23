import torch
import os
from encoder import Encoder

# 模型超参数
MAX_LENGTH = 1024
MODEL_DIM = 1024
N_HEADS = 16
N_BLOCKS = 10
""" # 如果你有足够的显卡和显存：
MAX_LENGTH = 2048
MODEL_DIM = 4096
N_HEADS = 32
N_BLOCKS = 32
"""
DROPOUT = 0.06

# 训练超参数
BATCH_SIZE = 1
N_BATCHES = 256
WARMUP_STEPS = 0
MAX_LEARINGRATE = 2e-3
TARGET_STEPS = 600000
MIN_LEARINGRATE = 0

# 微调超参数
FINETUNE_BATCH_SIZE = 1
FINETUNE_N_BATCHES = 128
FINETUNE_WARMUP_STEPS = 0
FINETUNE_MAX_LEARINGRATE = 6e-4
FINETUNE_TARGET_STEPS = 1000
FINETUNE_MIN_LEARINGRATE = 3e-4

# 预训练数据路径
PRETRAIN_DATA = "openwebtext/openwebtext.txt.bin.part1"
# 微调数据路径
FINETUNE_DATA = "ultrachat/ultrachat_release_230407.json.bin"
FINETUNE_MASK = "ultrachat/ultrachat_release_230407.json.mask.bin"
FINETUNE_N_BLOCKS = None # 防止爆显存
SYS_PROMPT = ""
ROLE_PREFIXES = ["Assistant: ", "User: "]

# 特殊token
SPECIAL_TOKENS = ["<pad>", "<eos>", "<ins>", "</ins>"]
try:
    encoder = Encoder.from_path("encoder.json")
    SPECIAL_TOKENS_IDS: dict[str, int] = {
        token_name: encoder.encode(token_name)[0]
        for token_name in SPECIAL_TOKENS
    }
    SPECIAL_TOKENS_TENSORS: dict[str, torch.Tensor] = {
        token_name: torch.tensor(SPECIAL_TOKENS_IDS[token_name])
        for token_name in SPECIAL_TOKENS_IDS.keys()
    }
except FileNotFoundError:
    print("没有找到encoder.json")
    SPECIAL_TOKENS_IDS = {}
    SPECIAL_TOKENS_TENSORS = {}

# 检查点位置和属性
PRETRAINED_STATE_DICT_PATH = "llm1500_pretrain_state_dict_3.8536354270763695.pt"
FINETUNED_STATE_DICT_PATH = "llm900_finetune_state_dict_1.9746221052482724.pt"
START_STEP = 1500
FINETUNE_START_STEP = 0

# Loss数据记录文件
LOSSES_LOG_PATH = "losses.log"
FINETUNE_LOSSES_LOG_PATH = "finetune-losses.log"
