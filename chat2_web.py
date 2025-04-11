import os
import torch
import torchaudio
import gradio as gr
import time
import base64
import whisper
import numpy as np
import requests
import json
from datetime import datetime
from pathlib import Path
from generator import Segment, Generator, load_csm_1b
from io import BytesIO

# 创建 wav 目录（如果不存在）
os.makedirs('wav', exist_ok=True)

# 初始化设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 初始化 Whisper 模型用于语音识别
print("Loading Whisper model...")
whisper_model = whisper.load_model("turbo")
print("Whisper model loaded")

# 初始化 CSM 模型用于语音生成
print("Loading CSM model...")
model_path = "/root/autodl-tmp/csm/model/ckpt.pt"
generator = load_csm_1b(model_path, device)
print("CSM model loaded")

# DeepSeek 配置
DEEPSEEK_API_KEY = "sk-"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 加载音频函数
def load_audio(audio_path, target_sample_rate):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=target_sample_rate
    )
    return audio_tensor

# 调用 DeepSeek Chat API 生成文本回复
def get_deepseek_response(user_message, history):
    try:
        print(f"Calling DeepSeek API for user message: '{user_message}'")
        
        # 构建消息历史
        messages = []
        for item in history:
            messages.append({"role": item["role"], "content": item["content"]})
        
        # 添加当前用户消息
        messages.append({"role": "user", "content": user_message})
        
        # 系统提示词：简短温柔的女生设定
        system_prompt = """You are a gentle and lovely girl, we are in a relationship of lovers, in communication:
1. The reply should be short, preferably no more than 30 words
2. The tone should be soft and sweet, full of emotion.
3. Reply in English, no emoticons allowed
You're fun to be around, a little provocative
6. Do not display your reply in the form of AI.
7. Ensure that users feel present and immersed."""
                
        # 准备请求数据
        payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "system": system_prompt,
            "temperature": 0.9,  # 增加创意性
            "max_tokens": 100,    # 限制输出长度
            "top_p": 0.9        # 保持适度的随机性
        }
        
        # 发送请求到 DeepSeek API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        }
        
        response = requests.post(
            f"{DEEPSEEK_BASE_URL}/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result["choices"][0]["message"]["content"]
            print(f"DeepSeek response: '{ai_response}'")
            return ai_response
        else:
            print(f"DeepSeek API error: {response.status_code}")
            print(response.text)
            return f"抱歉，我现在无法回应，请稍后再试。"
            
    except Exception as e:
        print(f"Error calling DeepSeek API: {e}")
        return "抱歉，我遇到了一些问题~"

# 初始化对话管理器
class ConversationManager:
    def __init__(self, max_history=5):
        self.max_history = max_history
        self.history = []
        self.chatbot_history = []
        
        # 准备参考语音段落（使用女声作为参考）
        self.reference_text = "And Lake turned round upon me, a little abruptly, with his odd yellowish eyes, a little like those of the sea eagle, and the ghost of his smile that flickered on his singularly pale face with a stern and insidious look confronted me."
        self.reference_path = "/root/autodl-tmp/csm/model/prompts/read_speech_a.wav"  # 女声参考
        
        # 加载参考语音
        self.reference_segment = Segment(
            text=self.reference_text,
            speaker=0,  # 女声
            audio=load_audio(self.reference_path, generator.sample_rate)
        )
        
        print(f"已初始化女声参考语音: {self.reference_text}")
    
    def add_message(self, text, audio_tensor, is_user=True):
        speaker_id = 1 if is_user else 0
        segment = Segment(text=text, speaker=speaker_id, audio=audio_tensor)
        
        self.history.append(segment)
        
        # 更新聊天机器人历史为新的消息格式
        message = {
            "role": "user" if is_user else "assistant",
            "content": text
        }
        self.chatbot_history.append(message)
        
        # 保持最大历史记录长度
        if len(self.history) > self.max_history * 2:  # 每轮对话包含用户和系统消息
            self.history = self.history[-self.max_history * 2:]
            self.chatbot_history = self.chatbot_history[-self.max_history * 2:]
            
        return segment
    
    def get_context(self):
        # 返回参考语音段落加上历史记录
        return [self.reference_segment] + self.history
    
    def get_chatbot_history(self):
        # 直接返回消息历史记录列表
        return self.chatbot_history

# 初始化对话管理器
conv_manager = ConversationManager()

# 使用Whisper将用户音频转换为文本
def transcribe_audio(audio_path):
    print(f"转写音频: {audio_path}")
    try:
        result = whisper_model.transcribe(audio_path)
        transcribed_text = result["text"]
        print(f"转写结果: '{transcribed_text}'")
        return transcribed_text
    except Exception as e:
        print(f"转写过程中出错: {e}")
        return f"[转写错误]"




# 生成系统回复-------------------------------------------------------------------------------------------------
def generate_response(user_text, user_audio_tensor):
    print(f"处理用户输入: '{user_text}'")
    
    # 添加用户消息到历史记录
    user_segment = conv_manager.add_message(user_text, user_audio_tensor, is_user=True)
    
    # 使用DeepSeek生成AI回复文本
    system_text = get_deepseek_response(user_text, conv_manager.chatbot_history[:-1])
    
    print(f"为AI回复生成语音: '{system_text}'")
    
    # 获取完整上下文（参考语音 + 历史记录）
    context = conv_manager.get_context()
    
    # 生成系统回复的语音
    system_audio = generator.generate(
        text=system_text,  # 系统回复文本
        speaker=0,         # 使用女声
        context=context,   # 包含参考语音和对话历史的上下文
        max_audio_length_ms=20_000,  # 增加最大长度以适应较长回复
    )
    
    # 添加系统回复到历史记录
    system_segment = conv_manager.add_message(system_text, system_audio, is_user=False)
    
    # 将音频转换为16位整数格式
    audio_numpy = (system_audio.cpu().numpy() * 32767).astype(np.int16)
    
    # 使用BytesIO和soundfile保存音频
    wav_io = BytesIO()
    import soundfile as sf
    sf.write(wav_io, audio_numpy, samplerate=generator.sample_rate, format="WAV")
    wav_io.seek(0)
    wav_bytes = wav_io.getvalue()
    
    # 将WAV数据转换为base64编码
    audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')
    
    # 创建一个带有自动播放的HTML音频元素
    html_audio = f'<audio controls autoplay src="data:audio/wav;base64,{audio_b64}"></audio>'
    
    print(f"生成的系统回复音频长度: {len(audio_numpy)} 采样点")
    
    # 返回系统回复文本、HTML音频元素和格式化的对话历史
    return system_text, gr.HTML(html_audio), conv_manager.get_chatbot_history()

# 清除对话历史
def clear_history():
    global conv_manager
    conv_manager = ConversationManager()
    return "", gr.HTML(""), []  # 返回空的聊天历史

# 处理用户输入（音频录制）
def process_user_input(audio_file):
    if audio_file is None:
        return "请先录制音频消息。", gr.HTML(""), []
    
    # 加载并转换用户音频
    audio_tensor, sample_rate = torchaudio.load(audio_file)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    
    # 使用Whisper将音频转换为文本
    # 创建临时文件用于Whisper处理
    temp_path = "temp_whisper_input.wav"
    torchaudio.save(temp_path, audio_tensor.unsqueeze(0), generator.sample_rate)
    user_text = transcribe_audio(temp_path)
    os.remove(temp_path)  # 删除临时文件
    
    # 生成系统响应
    return generate_response(user_text, audio_tensor)

# 创建Gradio界面
demo = gr.Blocks(theme=gr.themes.Soft())
with demo:
    gr.Markdown("# 🎙️ AI语音对话系统")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["microphone"], 
                type="filepath",
                label="录制消息"
            )
            submit_btn = gr.Button("发送", variant="primary")
            clear_btn = gr.Button("清除对话历史", variant="secondary")
            
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="对话历史", 
                height=400,
                show_label=True,
                type="messages"
            )
            with gr.Row():
                system_text = gr.Textbox(
                    label="AI回复文本", 
                    interactive=False
                )
            audio_output = gr.HTML(
                label="AI语音回复"
            )
    
    # 设置事件处理
    submit_btn.click(
        fn=process_user_input,
        inputs=[audio_input],
        outputs=[system_text, audio_output, chatbot]
    )
    
    # 清除对话历史
    def clear_history():
        global conv_manager
        conv_manager = ConversationManager()
        return "", gr.HTML(""), []
    
    clear_btn.click(
        fn=clear_history,
        inputs=[],
        outputs=[system_text, audio_output, chatbot]
    )

    # 自定义CSS提高界面美观度
    demo.load(js="""
    function checkAudioVisibility() {
        const audioResponse = document.getElementById('audio-response');
        if (audioResponse) {
            const waveform = audioResponse.querySelector('.audio-waveform');
            if (waveform) {
                waveform.style.display = 'block';
                waveform.style.height = '60px';
            }
        }
        setTimeout(checkAudioVisibility, 1000);
    }
    document.addEventListener('DOMContentLoaded', checkAudioVisibility);
    """)

# 启动应用
if __name__ == "__main__":
    # 确保缓存目录存在
    os.makedirs("gradio_cache", exist_ok=True)
    demo.launch(server_name="0.0.0.0", server_port=6006, share=True)