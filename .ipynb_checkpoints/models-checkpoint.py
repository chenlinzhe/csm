"""
CSM (条件语音模型) 神经网络架构
本模块实现了CSM模型的核心神经网络架构，负责从文本输入生成富有表现力的语音。
该架构由两个主要组件组成：

1. 骨干网络：基于LLaMA的transformer，处理文本输入并生成高级特征
2. 解码器网络：较小的transformer，将骨干网络的特征转换为音频令牌

主要组件：
- 文本处理：使用LLaMA架构理解文本语义
- 音频生成：采用多码本方法实现高质量语音合成
- 注意力机制：为文本和音频生成提供因果注意力
- 嵌入层：分别为文本和音频令牌提供嵌入
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torchtune
from torchtune.models import llama3_2


def llama3_2_1B() -> torchtune.modules.transformer.TransformerDecoder:
    """
    创建1B参数的LLaMA模型作为骨干网络。
    该模型负责处理文本输入并生成高级特征。
    
    架构：
    - 16层transformer
    - 32个注意力头（8个KV头用于高效注意力）
    - 2048维嵌入维度
    - 8192维中间层维度用于MLP
    - 128K词汇表大小
    - 2048最大序列长度
    
    返回：
        TransformerDecoder：配置好的LLaMA模型实例
    """
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


def llama3_2_100M() -> torchtune.modules.transformer.TransformerDecoder:
    """
    创建100M参数的LLaMA模型作为解码器网络。
    这个较小的模型将骨干网络的特征转换为音频令牌。
    
    架构：
    - 4层transformer
    - 8个注意力头（2个KV头）
    - 1024维嵌入维度
    - 8192维中间层维度
    - 128K词汇表大小
    - 2048最大序列长度
    
    返回：
        TransformerDecoder：配置好的LLaMA模型实例
    """
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
        embed_dim=1024,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


# 模型类型到其构造函数的映射
FLAVORS = {
    "llama-1B": llama3_2_1B,
    "llama-100M": llama3_2_100M,
}


def _prepare_transformer(model):
    """
    通过移除令牌嵌入和输出层来准备transformer模型。
    这使得模型可以作为特征提取器使用。
    
    参数：
        model：要准备的transformer模型
        
    返回：
        tuple：(处理后的模型, 嵌入维度)
    """
    embed_dim = model.tok_embeddings.embedding_dim
    model.tok_embeddings = nn.Identity()
    model.output = nn.Identity()
    return model, embed_dim


def _create_causal_mask(seq_len: int, device: torch.device):
    """
    为transformer注意力创建因果掩码。
    该掩码确保每个位置只能关注之前的位置。
    
    参数：
        seq_len：序列长度
        device：创建掩码的设备
        
    返回：
        torch.Tensor：形状为(seq_len, seq_len)的因果注意力掩码
    """
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def _index_causal_mask(mask: torch.Tensor, input_pos: torch.Tensor):
    """
    基于输入位置索引因果掩码。
    这用于生成过程中的高效注意力计算。
    
    参数：
        mask：形状为(max_seq_len, max_seq_len)的基础因果掩码
        input_pos：形状为(batch_size, seq_len)的输入位置
        
    返回：
        torch.Tensor：形状为(batch_size, seq_len, max_seq_len)的索引掩码
    """
    r = mask[input_pos, :]
    return r


def _multinomial_sample_one_no_sync(probs):
    """
    执行无需CUDA同步的多项式采样。
    通过避免不必要的GPU同步来提高生成速度。
    
    参数：
        probs：要从中采样的概率分布
        
    返回：
        torch.Tensor：采样的令牌索引
    """
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample_topk(logits: torch.Tensor, topk: int, temperature: float):
    """
    使用温度缩放从top-k令牌中采样。
    这实现了核采样以实现多样化的生成。
    
    参数：
        logits：模型的原始logits
        topk：要考虑的top令牌数量
        temperature：用于缩放logits的温度（越高越随机）
        
    返回：
        torch.Tensor：采样的令牌索引
    """
    logits = logits / temperature

    filter_value: float = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    scores_processed = torch.nn.functional.log_softmax(scores_processed, dim=-1)
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)

    sample_token = _multinomial_sample_one_no_sync(probs)
    return sample_token


@dataclass
class ModelArgs:
    """
    CSM模型的配置参数。
    
    属性：
        backbone_flavor：骨干模型类型（"llama-1B"或"llama-100M"）
        decoder_flavor：解码器模型类型（"llama-1B"或"llama-100M"）
        text_vocab_size：文本词汇表大小
        audio_vocab_size：每个码本的音频词汇表大小
        audio_num_codebooks：用于高质量合成的音频码本数量
    """
    backbone_flavor: str
    decoder_flavor: str
    text_vocab_size: int
    audio_vocab_size: int
    audio_num_codebooks: int


class Model(nn.Module):
    """
    主要的CSM模型，结合文本理解和音频生成。
    
    架构：
    1. 骨干网络：使用LLaMA架构处理文本输入
    2. 解码器网络：从骨干特征生成音频令牌
    3. 嵌入层：为文本和音频令牌提供单独的嵌入
    4. 投影层：将骨干特征映射到解码器维度
    5. 音频头：多个头用于生成音频令牌
    
    该模型使用多码本方法进行高质量语音合成，
    每个码本捕获音频信号的不同方面。
    """
    
    def __init__(self, args: ModelArgs):
        """
        使用指定配置初始化CSM模型。
        
        参数：
            args：模型配置参数
        """
        super().__init__()
        self.args = args

        # 初始化骨干网络和解码器网络
        self.backbone, backbone_dim = _prepare_transformer(FLAVORS[args.backbone_flavor]())
        self.decoder, decoder_dim = _prepare_transformer(FLAVORS[args.decoder_flavor]())

        # 初始化嵌入层
        self.text_embeddings = nn.Embedding(args.text_vocab_size, backbone_dim)
        self.audio_embeddings = nn.Embedding(args.audio_vocab_size * args.audio_num_codebooks, backbone_dim)

        # 初始化投影层和输出层
        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)
        self.codebook0_head = nn.Linear(backbone_dim, args.audio_vocab_size, bias=False)
        self.audio_head = nn.Parameter(torch.empty(args.audio_num_codebooks - 1, decoder_dim, args.audio_vocab_size))

    def setup_caches(self, max_batch_size: int) -> torch.Tensor:
        """
        设置用于高效生成的键值缓存。
        这些缓存存储中间键和值张量以避免重复计算。
        
        参数：
            max_batch_size：生成的最大批次大小
            
        返回：
            torch.Tensor：用于注意力的初始因果掩码
        """
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        with device:
            self.backbone.setup_caches(max_batch_size, dtype)
            self.decoder.setup_caches(max_batch_size, dtype, decoder_max_seq_len=self.args.audio_num_codebooks)

        self.register_buffer("backbone_causal_mask", _create_causal_mask(self.backbone.max_seq_len, device))
        self.register_buffer("decoder_causal_mask", _create_causal_mask(self.args.audio_num_codebooks, device))

    def generate_frame(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: float,
        topk: int,
    ) -> torch.Tensor:
        """
        生成单个音频令牌帧。
        这是核心生成函数，它：
        1. 通过骨干网络处理输入令牌
        2. 生成第一个音频令牌
        3. 使用解码器生成剩余的音频令牌
        
        参数：
            tokens：形状为(batch_size, seq_len, audio_num_codebooks+1)的输入令牌
            tokens_mask：输入令牌的掩码
            input_pos：每个令牌的位置索引
            temperature：采样温度
            topk：用于采样的top令牌数量
            
        返回：
            torch.Tensor：形状为(batch_size, audio_num_codebooks)的生成音频令牌
        """
        dtype = next(self.parameters()).dtype
        b, s, _ = tokens.size()

        # 通过骨干网络处理
        assert self.backbone.caches_are_enabled(), "backbone caches are not enabled"
        curr_backbone_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)
        embeds = self._embed_tokens(tokens)
        masked_embeds = embeds * tokens_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)
        h = self.backbone(h, input_pos=input_pos, mask=curr_backbone_mask).to(dtype=dtype)

        # 生成第一个音频令牌
        last_h = h[:, -1, :]
        c0_logits = self.codebook0_head(last_h)
        c0_sample = sample_topk(c0_logits, topk, temperature)
        c0_embed = self._embed_audio(0, c0_sample)

        # 生成剩余的音频令牌
        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
        curr_sample = c0_sample.clone()
        curr_pos = torch.arange(0, curr_h.size(1), device=curr_h.device).unsqueeze(0).repeat(curr_h.size(0), 1)

        # 为新帧重置解码器缓存
        self.decoder.reset_caches()
        for i in range(1, self.args.audio_num_codebooks):
            curr_decoder_mask = _index_causal_mask(self.decoder_causal_mask, curr_pos)
            decoder_h = self.decoder(self.projection(curr_h), input_pos=curr_pos, mask=curr_decoder_mask).to(
                dtype=dtype
            )
            ci_logits = torch.mm(decoder_h[:, -1, :], self.audio_head[i - 1])
            ci_sample = sample_topk(ci_logits, topk, temperature)
            ci_embed = self._embed_audio(i, ci_sample)

            curr_h = ci_embed
            curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
            curr_pos = curr_pos[:, -1:] + 1

        return curr_sample

    def reset_caches(self):
        """
        重置骨干网络和解码器的键值缓存。
        在生成新序列之前应该调用此函数。
        """
        self.backbone.reset_caches()
        self.decoder.reset_caches()

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        """
        嵌入特定码本的音频令牌。
        
        参数：
            codebook：码本索引
            tokens：令牌索引
            
        返回：
            torch.Tensor：嵌入的音频令牌
        """
        return self.audio_embeddings(tokens + codebook * self.args.audio_vocab_size)

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        嵌入文本和音频令牌。
        文本令牌直接嵌入，而音频令牌根据其码本索引进行偏移嵌入。
        
        参数：
            tokens：包含文本和音频的输入令牌
            
        返回：
            torch.Tensor：组合的嵌入
        """
        text_embeds = self.text_embeddings(tokens[:, :, -1]).unsqueeze(-2)

        audio_tokens = tokens[:, :, :-1] + (
            self.args.audio_vocab_size * torch.arange(self.args.audio_num_codebooks, device=tokens.device)
        )
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.args.audio_num_codebooks, -1
        )

        return torch.cat([audio_embeds, text_embeds], dim=-2)
