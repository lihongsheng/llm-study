import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from gtts import gTTS
import os
import numpy as np
print(np.__version__)
# 步骤 1: 汉语语音转汉语文字
model = whisper.load_model("base")
result = model.transcribe("/Users/lhs/Downloads/test1.mp3")
chinese_text = result["text"]
print("识别出的中文文本:", chinese_text)
# 创建语音识别管道
# transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base")
# result = transcriber("/Users/lhs/Downloads/test1.mp3")
# print(result["text"])
# chinese_text = result["text"]
# 步骤 2: 汉语文字翻译成英文
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
inputs = tokenizer(chinese_text, return_tensors="pt")
outputs = model.generate(**inputs)
english_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("翻译后的英文文本:", english_text)

# 步骤 3: 英文生成语音
tts = gTTS(text=english_text, lang='en')
tts.save("/Users/lhs/Downloads/test2.mp3")
# os.system("mpg321 output_audio.mp3")
