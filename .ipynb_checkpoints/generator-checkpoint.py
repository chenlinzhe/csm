"""
CSM (Conditional Speech Model) Generator
这个文件实现了语音生成的高层接口，负责将文本转换为语音。
主要功能包括：
1. 文本和音频的标记化处理
2. 语音生成的核心逻辑
3. 模型加载和初始化
4. 上下文管理和处理
"""

from dataclasses import dataclass
from typing import List, Tuple
import os
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from models import Model, ModelArgs
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
from watermarking import CSM_1B_GH_WATERMARK, load_watermarker, watermark


@dataclass
class Segment:
    """
    表示一个语音段落的数据结构
    Args:
        speaker: 说话者ID（0或1，用于区分不同说话者）
        text: 文本内容
        audio: 音频张量，采样率为24kHz
    """
    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: torch.Tensor


def load_llama3_tokenizer():
    """
    加载并配置LLaMA-3分词器
    这个分词器用于处理文本输入，是模型理解文本语义的基础
    Returns:
        配置好的LLaMA-3分词器
    """
    tokenizer_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
    )

    return tokenizer


class Generator:
    """
    语音生成器的主类
    负责协调文本处理、语音生成和后处理等所有步骤
    """
    def __init__(
        self,
        model: Model,
    ):
        """
        初始化生成器
        Args:
            model: CSM模型实例
        """
        self._model = model
        self._model.setup_caches(1)

        self._text_tokenizer = load_llama3_tokenizer()

        device = next(model.parameters()).device
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=device)
        mimi.set_num_codebooks(32)
        self._audio_tokenizer = mimi

        self._watermarker = load_watermarker(device=device)

        self.sample_rate = mimi.sample_rate
        self.device = device

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将文本段落转换为模型可以处理的标记
        Args:
            text: 输入文本
            speaker: 说话者ID
        Returns:
            标记张量和掩码张量
        """
        frame_tokens = []
        frame_masks = []

        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True

        frame_tokens.append(text_frame.to(self.device))
        frame_masks.append(text_frame_mask.to(self.device))

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将音频数据转换为标记
        Args:
            audio: 输入音频张量
        Returns:
            音频标记张量和掩码张量
        """
        frame_tokens = []
        frame_masks = []

        # (K, T)
        audio = audio.to(self.device)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), 33).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool().to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        处理完整的语音段落（包含文本和音频）
        Args:
            segment: 语音段落对象
        Returns:
            组合后的标记张量和掩码张量
        """
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)

        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
    ) -> torch.Tensor:
        """
        生成语音的主要函数
        Args:
            text: 要转换为语音的文本
            speaker: 说话者ID
            context: 上下文语音段落列表
            max_audio_length_ms: 最大音频长度（毫秒）
            temperature: 生成的随机性参数
            topk: top-k采样参数
        Returns:
            生成的音频张量
        """
        self._model.reset_caches()

        max_audio_frames = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        samples = []
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        max_seq_len = 2048 - max_audio_frames
        if curr_tokens.size(1) >= max_seq_len:
            raise ValueError(f"Inputs too long, must be below max_seq_len - max_audio_frames: {max_seq_len}")

        for _ in range(max_audio_frames):
            sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            if torch.all(sample == 0):
                break  # eos

            samples.append(sample)

            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat(
                [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

        audio = self._audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0)).squeeze(0).squeeze(0)

        # This applies an imperceptible watermark to identify audio as AI-generated.
        # Watermarking ensures transparency, dissuades misuse, and enables traceability.
        # Please be a responsible AI citizen and keep the watermarking in place.
        # # If using CSM 1B in another application, use your own private key and keep it secret.

        
        # audio, wm_sample_rate = watermark(self._watermarker, audio, self.sample_rate, CSM_1B_GH_WATERMARK)
        # audio = torchaudio.functional.resample(audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate)

        return audio


def load_csm_1b(ckpt_path: str = "ckpt.pt", device: str = "cuda") -> Generator:
    """
    加载CSM-1B模型并创建生成器
    Args:
        ckpt_path: 模型检查点路径
        device: 运行设备
    Returns:
        配置好的Generator实例
    """
    model_args = ModelArgs(
        backbone_flavor="llama-1B",  # 使用LLaMA-1B作为主干网络
        decoder_flavor="llama-100M", # 使用较小的解码器
        text_vocab_size=128256,      # 文本词汇表大小
        audio_vocab_size=2051,       # 音频词汇表大小
        audio_num_codebooks=32,      # 音频码本数量
    )
    model = Model(model_args).to(device=device, dtype=torch.bfloat16)
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict)

    generator = Generator(model)
    return generator



# def load_csm_1b(model_path: str = "/root/autodl-tmp/csm/model", device: str = "cuda") -> Generator:
#     model_args = ModelArgs(
#         backbone_flavor="llama-1B",
#         decoder_flavor="llama-100M",
#         text_vocab_size=128256,
#         audio_vocab_size=2051,
#         audio_num_codebooks=32,
#     )
#     model = Model(model_args).to(device=device, dtype=torch.bfloat16)

#     # 构建模型权重文件的路径
#     ckpt_path = os.path.join(model_path, "model.safetensors")  # 假设你的权重文件是 model.safetensors
#     if not os.path.exists(ckpt_path):
#         ckpt_path = os.path.join(model_path, "pytorch_model.bin")  # 如果是 pytorch_model.bin

#     if os.path.exists(ckpt_path):
#         state_dict = torch.load(ckpt_path)
#         model.load_state_dict(state_dict)
#     else:
#         raise FileNotFoundError(f"模型权重文件在路径 {model_path} 下未找到 (尝试了 model.safetensors 和 pytorch_model.bin)")

#     generator = Generator(model)
#     return generator
