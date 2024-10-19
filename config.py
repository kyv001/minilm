import torch
import os

# 模型超参数
MAX_LENGTH = 1024
MODEL_DIM = 768
LORA_DIM = 32
N_HEADS = MODEL_DIM // 64
N_BLOCKS = 4
DROPOUT = 0.06

# 训练超参数
TRAIN = False
BATCH_SIZE = 1
N_BATCHES = 60
WARMUP_STEPS = 0
MAX_LEARNINGRATE = 4e-5
TARGET_STEPS = 1
MIN_LEARNINGRATE = 4e-5

# 预训练数据路径（*.jsonl.bin）
PRETRAIN_DATA = "WuDaoCorpus2.0_base_200G/part_0.bin"
# 微调数据路径（*.bin）
FINETUNE_DATA = "instruct_finetune.bin"
FINETUNE = False
N_FINETUNE_BLOCKS = 8 # 只训练最后8层防止爆显存
SYS_PROMPT = ""

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
PRETRAINED_STATE_DICT_PATH = "ckpt.pt"
FINETUNED_STATE_DICT_PATH = None
START_STEP = 0

# Loss数据记录文件
LOSSES_LOG_PATH = "losses_finetune.log"

__all__ = [
    "MAX_LENGTH",
    "MODEL_DIM",
    "LORA_DIM",
    "N_HEADS",
    "N_BLOCKS",
    "DROPOUT",
    "TRAIN",
    "BATCH_SIZE",
    "N_BATCHES",
    "WARMUP_STEPS",
    "MAX_LEARNINGRATE",
    "TARGET_STEPS",
    "MIN_LEARNINGRATE",
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
