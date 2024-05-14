import random
import numpy as np
import torch
from MyLlama import LlamaForCausalLM

class LlamaConfig:
    def __init__(self, vocab_size, hidden_size, intermediate_size, num_hidden_layers, num_attention_heads,
                 num_key_value_heads, hidden_act, max_position_embeddings, initializer_range, rms_norm_eps,
                 use_cache, pad_token_id, bos_token_id, eos_token_id, pretraining_tp, tie_word_embeddings,
                 rope_theta, rope_scaling, attention_bias, attention_dropout, output_attentions, output_hidden_states,
                 use_return_dict):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pretraining_tp = pretraining_tp
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.use_return_dict = use_return_dict


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
    output_attentions=False,
    output_hidden_states=False,
    use_return_dict=True
)

def seed_everything(seed):
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LlamaForCausalLM(config).to(device)
print(model)

model.load_state_dict(torch.load("official_model.pth"))

# 假设下面的输入是:
# <pad><pad><bos>今天天气很好<eos>
input_seq = torch.tensor([0, 0, 1, 10, 20, 20, 30, 40, 50, 2])
input_ids = input_seq.unsqueeze(0).to(device)
attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1]]).to(device)

past_key_values = None
use_cache = True

while True:
    output = model(input_ids, attention_mask, past_key_values=past_key_values, use_cache=use_cache)
    logits = output[1]
    past_key_values = output[2]
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
