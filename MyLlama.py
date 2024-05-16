import math
from typing import Optional, Tuple, List, Dict, Any, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm
        先将每个词向量长度缩小到1
        再乘以可学习参数得到新的长度
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # 计算每个隐藏状态向量的长度
        variance = hidden_states.pow(2).mean(-1, keepdim=True)

        # 应用标准化公式,epsilon用于防止除以零
        # rsqrt(x) = 1 / sqrt(x)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # 乘以可学习的权重参数,对隐藏状态进行调整
        return self.weight * hidden_states


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        # Swish(x) = x * sigmoid(beta * x)
        # beta = 1 时 Swish(x) = SiLU(x)
        # SiLU(x) = x * sigmoid(x)
        # SwiGLU(x, W, V) = Swish(xW) * (xV)
        # W: gate_proj, V: up_proj
        # 先经过门控, 再经过SiLU激活函数, 与上投影的结果求哈达玛积, 最后经过下投影
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        # 这里的dim是head_dim,单个注意力头的词向量维度
        # head_dim = hidden_size // num_attention_heads
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # i是词向量中每个维度的位置下标
        # 计算公式中的 1/(10000^(2i/dim)),即 theta
        # 得到一个(dim/2,)维的向量
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))

        # 如果一个参数不参与梯度下降,但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        # 初始化一个tensor [0, 1, 2, 3, ...],(max_seq_len_cached,)维
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        # 如果使用线性放缩(LinearScaling)实现对llama的长度扩展
        # 在这里用t除以放缩因子scaling_factor
        # 比如使llama的上下文长度从2048扩展到4096,t = t / 2

        # (max_seq_len_cached,) X (dim/2,) -> (max_seq_len_cached, dim/2)
        # freqs第0行是inv_freq*0,第1行是inv_freq*1,第2行是inv_freq*2,...
        # freqs即公式中的 m*theta
        freqs = torch.outer(t, self.inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # (max_len, dim/2) -> (max_len, dim), 这里的cat操作相当于把freqs的每一行复制一遍
        # freqs的每一行是: m * [theta_0, theta_1, ..., theta_(dim/2-1), theta_0, theta_1, ..., theta_(dim/2-1)]
        emb = torch.cat((freqs, freqs), dim=-1)

        # 取emb的cos和sin,保存
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # 这里的x是split head并transpose后的输入
        # x: [bs, num_attention_heads, seq_len, head_size]

        # 如果输入的seq_len大于之前缓存的最大长度,重新计算
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # 取前seq_len个位置的cos和sin
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    # 因为线性层是无序的,不依赖维度顺序信息,并不一定要求按照维度顺序一正一负两两组合对词向量进行取负数
    # [q_0, q_1, ..., q_{d/2-1}, q_{d/2}, q_{d/2+1}, ..., q_{d-1}] ->
    # [-q_{d/2}, -q_{d/2+1}, ..., -q_{d-1}, q_0, q_1, ..., q_{d/2-1}]
    # 具体原理与公式见本项目中的RoPE.ipynb
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # 按照position_ids取出对应位置的cos和sin
    # 这样实现便于KV-cache机制中仅对新增的位置进行RoPE
    # unsqueeze_dim是head所在的维度,使cos和sin能广播到每个head
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    # 执行RoPE编码,具体原理与公式见本项目中的RoPE.ipynb
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # (batch, num_key_value_heads, seqlen, head_dim) -> (batch, num_key_value_heads, 1, seqlen, head_dim)
    # -> (batch, num_key_value_heads, n_rep, seqlen, head_dim)
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)

    # -> (batch, num_key_value_heads * n_rep, seqlen, head_dim)
    # 其中 num_key_value_heads * n_rep = num_heads
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Cache:
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        # 返回指定层已缓存的序列的长度
        # 如果指定层还没有缓存,则返回0
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        # 返回已缓存的layer层数
        return len(self.key_cache)

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        # 每次新的forward开始时更新seen_tokens
        if layer_idx == 0:
            self.seen_tokens += key_states.shape[-2]

        # Update the cache
        # 第一次forward时,直接将key_states和value_states加入cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # 之后的forward,将新的key_states和value_states加入cache
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        # 返回更新(cat)后的key_states和value_states
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    # 我也不知道为什么,保存和加载KV-Cache的时候必须用这两个函数为Tuple一下再变回去
    # 直接把Cache类原样传递给下一轮forward不行吗？明明相关的调用代码中判断都写好了
    # 调试发现,第一次forward时,past_key_values是None,导致LlamaModel的forward函数中的变量use_legacy_cache为True
    # 该变量使得第一次forward结束时,触发了to_legacy_cache,将next_decoder_cache从Cache类转为了Tuple,传递给下一次forward
    # 之后的forward,又会因为past_key_value是Tuple,继续use_legacy_cache设置True,之后触发from_legacy_cache函数,将Tuple转换回Cache类
    # 绕这么一圈子,可能是为了兼容旧版本的transformers库,也可能是新版pytorch算子的兼容性问题
    # 毕竟,即使目前最新版的源码中,KV-Cache的机制优化还处于TODO待重构状态
    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        # Cache类转为Tuple
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None):
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        # Tuple转为Cache类
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        # 对输入进行线性变换得到Q、K、V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 将Q、K、V分成多个head
        # head_dim = hidden_size // num_attention_heads
        # 一个hidden_size维的词向量可以被视作是num_attention_heads个head_dim维的词向量的拼接
        # (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, num_attention_heads, head_dim)
        # -> (batch_size, num_attention_heads, seq_len, head_dim)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # KV-Cache机制中新增的seq_len,除第一次forward外,一般是1,即只有一个新的token
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            # 如果有KV-Cache,需要将KV-Cache中的seq_len加到当前的seq_len上
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        # 根据当前的seq_len,仅对新增的位置进行RoPE
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # 缓存的是执行了RoPE之后的key_states和value_states
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            # 更新KV-Cache,将新的key_states和value_states加入cache,返回包括之前的cache在内的所有key_states和value_states
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # GQA: 复制num_key_value_heads份key_states和value_states
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 计算 Q@K.T/sqrt(d_k)
        # key_states.transpose: (batch_size, num_heads, head_dim, seq_len)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # attn_weights (batch_size, num_heads, seq_len, seq_len)
        # attention_mask (batch_size, 1, seq_len, seq_len)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        # 计算softmax(Q@K.T/sqrt(d_k)), 得到注意力权重
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # 对注意力权重进行dropout, 默认为0, 即不进行dropout
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        # 基于注意力权重对V进行加权求和
        attn_output = torch.matmul(attn_weights, value_states)

        # 将多头注意力的结果拼接起来
        # attn_output (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, seq_len, hidden_size)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        # 最后的线性变换,将多头注意力的结果整合,映射回hidden_size
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        # 保存hidden_states, 用于残差连接
        residual = hidden_states
        # 先对输入进行RMSNorm
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        # 残差连接
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # 这里返回值的数量与形状存在不确定和不严格的问题
        # 个人认为像LlamaAttention的forward方法那样,将不需要返回的值设为None是更合理的方法
        # 但是transformers库源码就是这么写的,后面的代码则是写了一个判断来适配
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


def prepare_4d_causal_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length):
    batch_size, seq_length = input_shape
    # 生成一个上三角矩阵, 用于mask掉未来的信息, 需要被mask的位置为torch.finfo(torch.float32).min, 大约是-3.4028234663852886e+38
    # 不能用-inf,否则如果某一行全是-inf,softmax之后会变成nan
    causal_mask = torch.triu(
        torch.full((seq_length, seq_length), torch.finfo(inputs_embeds.dtype).min, dtype=torch.float32,
                   device=attention_mask.device),
        diagonal=1)
    # causal_mask扩展到(batch_size, 1, seq_length, seq_length)
    causal_mask = causal_mask.expand(batch_size, 1, seq_length, seq_length)
    # 如果attention_mask是None,则初始化一个全1的mask
    if attention_mask is None:
        attention_mask = torch.ones((batch_size, seq_length), device=inputs_embeds.device)
    # attention_mask扩展到(batch_size, 1, seq_length, seq_length)
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).repeat(1, 1, seq_length, 1)
    # 用torch.finfo(torch.float32).min替换掉attention_mask中的0
    attention_mask = torch.where(attention_mask == 0, torch.finfo(inputs_embeds.dtype).min, float(0.0))

    # 两个mask相加, 即为最终的attention_mask
    # causal_mask = attention_mask + causal_mask
    # 将相加导致溢出为-inf的位置重新置为torch.finfo(torch.float32).min
    # 这个问题是我使用加法实现导致的,transformers库中并没有这个问题
    # causal_mask = torch.where(attention_mask == float('-inf'), torch.finfo(inputs_embeds.dtype).min, attention_mask)

    # 这是transformers库中的实现,直接用attention_mask中非0的部分作为mask,在causal_mask中置换, 不会出现溢出问题
    causal_mask = causal_mask.masked_fill(attention_mask.bool(), torch.finfo(inputs_embeds.dtype).min)
    return causal_mask


class LlamaModel(nn.Module):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        # 词向量的embedding层,padding_idx会被编码成全0的向量
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # 总共num_hidden_layers层的DecoderLayer
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple]:
        # 关于返回值的设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # 获取输入的batch_size和seq_length
        batch_size, seq_length = input_ids.shape[:2]

        # 如果是第一次forward,则初始化KV-Cache
        # 如果不是第一次forward,则加载之前的KV-Cache
        past_key_values_length = 0
        if use_cache:
            # 如果past_key_values是Cache类或子类,则直接使用
            use_legacy_cache = not isinstance(past_key_values, Cache)
            # 如果是第一次forward,past_key_values是None,from_legacy_cache会初始化KV-Cache
            # 如果past_key_values是兼容旧版本的Tuple,则转换为Cache类
            # 这里的代码实现存在一些问题,详细分析见本文件231行到237行的注释
            if use_legacy_cache:
                past_key_values = Cache().from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        # position_ids是KV-Cache机制中新增的seq_len,除第一次forward外,一般是1,即只有一个新的token
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        # 编码input_ids得到词向量
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # 4d mask is passed through the layers
        # 整理attention_mask的形状,在此期间生成causal mask,用于mask掉未来的信息
        # 生成的attention_mask的形状为(batch_size, 1, seq_length, seq_length)
        """
        attention_mask用于在 Q@K.T/sqrt(d_k) 得到的attention score矩阵中,将<pad> token对应的 列 全部置为-inf
        -inf在softmax之后就会变为0,之后基于attention score矩阵对 V 的加权求和时,就不会加入<pad> token的信息(乘以0)
        由于大模型都采用left pad, 即<pad> token补充在句子的前面, 所以经过上三角的causal mask后,<pad> token所在的 行 也会被全部mask掉
        但是一整行的-inf会导致softmax之后的attention score是 1/seq_length, 这意味着<pad> token也能够平均权重地接收其他token的信息,更新自己的词向量
        不过,<pad> token的这些数值变化由于mask的存在,不会对其他token产生影响,可以忽略
        代码中可以直接在embedding层设置padding_idx,将<pad> token的词向量置为全0
        """
        attention_mask = prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # 逐层调用decoder layer
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            # 保存本次forward的KV-Cache,用于下次forward
            if use_cache:
                # 这里的判断是因为LlamaDecoderLayer的返回变量数量不确定
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # 层与层之间的RMSNorm
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            # 保存本次forward的KV-Cache,用于下次forward
            # 注意这里的use_legacy_cache,它的值的异常导致了一些问题,详细分析见本文件231行到237行的注释
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        # hidden_states, past_key_values, hidden_states, attentions
        return hidden_states, next_cache, all_hidden_states, all_self_attns


class LlamaForCausalLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple]:

        # 关于返回值的设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # 调用基础模型
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 将基座模型的输出传入全连接层,映射到vocab_size维
        # 相当于一个类别数为vocab_size的分类任务
        # logits值最大的维度对应的即是next token
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        # 计算loss,用于训练
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            # 用前n-1个token(下标[0,seq_len-1])
            shift_logits = logits[..., :-1, :].contiguous()
            # 预测后n-1个token(下标[1,seq_len])
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        # loss, logits, past_key_values, hidden_states, attentions
        return loss, logits, outputs[1], outputs[2], outputs[3]
