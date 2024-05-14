import random
import numpy as np
import torch
from transformers import LlamaForCausalLM, LlamaConfig


def seed_everything(seed):
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

config = LlamaConfig(
    vocab_size=100,
    hidden_size=512,
    intermediate_size=1024,
    num_hidden_layers=3,
    num_attention_heads=8,
    num_key_value_heads=4,
    hidden_act="silu",
    max_position_embeddings=2048,
    initializer_range=0.02,
    rms_norm_eps=1e-6,
    use_cache=True,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
    pretraining_tp=1,
    tie_word_embeddings=False,
    rope_theta=10000.0,
    rope_scaling=None,
    attention_bias=False,
    attention_dropout=0.0,
)

# CUDA设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用纯python实现的注意力机制,而不是算子库
config._attn_implementation = "eager"
official_model = LlamaForCausalLM(config).to(device)
print(official_model)

torch.save(official_model.state_dict(), "official_model.pth")

# 假设下面的输入是:
# <pad><pad><bos>今天天气很好<eos>
input_seq = torch.tensor([0, 0, 1, 10, 20, 20, 30, 40, 50, 2])
input_ids = input_seq.unsqueeze(0).to(device)
attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1]]).to(device)

past_key_values = None
use_cache = True

output = official_model.generate(**{"input_ids": input_ids, "attention_mask": attention_mask}, max_length=20, use_cache=use_cache)
print(output)

while True:
    output = official_model(input_ids, attention_mask, past_key_values=past_key_values, use_cache=use_cache)
    logits = output.logits
    past_key_values = output.past_key_values
    next_token_logits = logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1)
    input_seq = torch.cat([input_seq, next_token.cpu()], dim=-1)
    input_ids = torch.tensor(next_token).unsqueeze(0).to(device)
    attention_mask = torch.cat([attention_mask, torch.ones(1, 1).to(device)], dim=-1)
    if next_token.item() == 2:
        break
    if len(input_seq) >= 20:
        break
print(input_seq)
